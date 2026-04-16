import gc
import os
import math
import random
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

# Assuming these are in your local directory
from gap.purification.data_loaders import get_pets_data, get_cifar10_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class DatasetConfig:
    name: str
    root: str
    img_size: int
    in_channels: int
    num_workers: int = 0

@dataclass
class HyperConfig:
    base_channels: int
    channel_mults: tuple
    num_res_blocks: int
    dropout: float
    lr: float
    batch_size: int
    num_timesteps: int

# ============================================================
#  UNet + Diffusion (Properly Conditioned)
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, time_emb_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)

class UNetLite(nn.Module):
    def __init__(self, in_channels, base_channels, channel_mults=(1, 2, 4), num_res_blocks=2, dropout=0.1):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = TimeEmbedding(base_channels)
        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.downs = nn.ModuleList([])
        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
            if i != len(channel_mults) - 1:
                self.downs.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))

        # Middle
        self.mid1 = ResidualBlock(ch, ch, time_dim, dropout)
        self.mid2 = ResidualBlock(ch, ch, time_dim, dropout)

        # Decoder
        self.ups = nn.ModuleList([])
        for i, mult in enumerate(reversed(channel_mults)):
            out_ch = base_channels * mult
            for j in range(num_res_blocks + 1):
                # The first block at each decoder level takes the concatenated skip
                in_ch = ch + out_ch if j == 0 else ch
                self.ups.append(ResidualBlock(in_ch, out_ch, time_dim, dropout))
                ch = out_ch
            if i != len(channel_mults) - 1:
                self.ups.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x = self.in_conv(x)
        
        # --- ENCODER ---
        hs = []
        curr = x
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                curr = layer(curr, t_emb)
            else: # Downsample
                hs.append(curr) # Save features before shrinking
                curr = layer(curr)
        hs.append(curr) # Save final encoder state
        
        # --- MIDDLE ---
        curr = self.mid1(curr, t_emb)
        curr = self.mid2(curr, t_emb)
        
        # --- DECODER ---
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                # If the layer expects a skip connection
                if layer.conv1.in_channels > curr.shape[1]:
                    skip = hs.pop()
                    # Defensive spatial matching
                    if curr.shape[-2:] != skip.shape[-2:]:
                        curr = F.interpolate(curr, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                    curr = torch.cat([curr, skip], dim=1)
                curr = layer(curr, t_emb)
            else: # Upsample
                curr = layer(curr)
                
        return self.out_conv(F.silu(self.out_norm(curr)))

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 4)
        self.lin2 = nn.Linear(dim * 4, dim * 4)

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        # Handle cases where dim might be odd
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
            
        return self.lin2(F.silu(self.lin1(emb)))

class GaussianDiffusion(nn.Module):
    def __init__(self, model, num_timesteps=1000):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        x_noisy = sqrt_ac * x0 + sqrt_om * noise
        return F.mse_loss(self.model(x_noisy, t), noise)

# ============================================================
#  Data & Training Infrastructure
# ============================================================

def get_dataloaders(cfg, batch_size):
    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*cfg.in_channels, [0.5]*cfg.in_channels),
    ])
    
    # Load dataset once
    if cfg.name == "cifar10": 
        ds = get_cifar10_data()
    else: 
        ds = get_pets_data(binary=("binary" in cfg.name))
        
    ds.transform = tfm
    v_sz = int(0.1 * len(ds))
    t_ds, v_ds = random_split(ds, [len(ds)-v_sz, v_sz])
    
    # FORCED num_workers=0 to prevent Windows Deadlocks
    return (
        DataLoader(t_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(v_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    )

def train_model(dataset_cfg, hcfg, max_epochs, seed, save_dir=None):
    torch.manual_seed(seed)
    train_loader, val_loader = get_dataloaders(dataset_cfg, hcfg.batch_size)
    
    model = UNetLite(dataset_cfg.in_channels, hcfg.base_channels, hcfg.channel_mults, hcfg.num_res_blocks, hcfg.dropout).to(device)
    diffusion = GaussianDiffusion(model, hcfg.num_timesteps).to(device)
    opt = torch.optim.AdamW(diffusion.parameters(), lr=hcfg.lr)
    
    best_val = float("inf")
    try:
        for epoch in range(max_epochs):
            diffusion.train()
            # Disable inner tqdm if you're still seeing freezes
            for x, _ in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False, disable=False):
                t = torch.randint(0, hcfg.num_timesteps, (x.size(0),), device=device)
                loss = diffusion.p_losses(x.to(device), t)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            
            # Simple print instead of nested bars
            diffusion.eval()
            v_loss = 0.0
            with torch.no_grad():
                for x, _ in val_loader:
                    t = torch.randint(0, hcfg.num_timesteps, (x.size(0),), device=device)
                    v_loss += diffusion.p_losses(x.to(device), t).item() * x.size(0)
            
            v_loss /= len(val_loader.dataset)
            print(f"  - {dataset_cfg.name} Epoch {epoch+1} Val Loss: {v_loss:.6f}")
            
            if v_loss < best_val:
                best_val = v_loss
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({"model": model.state_dict(), "hcfg": asdict(hcfg)}, f"{save_dir}/best_model.pt")
                    
    finally:
        # CLEANUP: This is the most important part for sweeps
        del train_loader
        del val_loader
        del model
        del diffusion
        del opt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
    return best_val

# ============================================================
#  Search & Sweep Logic
# ============================================================

def get_baseline_config():
    return HyperConfig(base_channels=32, channel_mults=(1, 2, 4), num_res_blocks=2, dropout=0.1, lr=3e-4, batch_size=64, num_timesteps=400)

def sensitivity_sweep(dcfg, baseline, sweep_space, epochs=2):
    results = {}
    for param, values in sweep_space.items():
        param_res = []
        for v in values:
            h = HyperConfig(**asdict(baseline))
            setattr(h, param, v)

            print("Testing:", h)

            val = train_model(dcfg, h, epochs, seed=0)
            param_res.append((v, val))
        param_res.sort(key=lambda x: x[1])
        results[param] = {"importance": param_res[-1][1] - param_res[0][1], "best": [x[0] for x in param_res[:2]]}
    return results

def sample_from_space(rng, space):
    return HyperConfig(
        base_channels=rng.choice(space["base_channels"]),
        channel_mults=rng.choice(space["channel_mults"]),
        num_res_blocks=rng.choice(space["num_res_blocks"]),
        dropout=rng.choice(space["dropout"]),
        lr=rng.choice(space["lr"]),
        batch_size=rng.choice(space["batch_size"]),
        num_timesteps=rng.choice(space["num_timesteps"]),
    )

# ============================================================
#  Main Execution Path
# ============================================================

def main():
    d_cfgs = [
        DatasetConfig("pets_binary", "../datasets/oxford-iiit-pet", 128, 3),
        DatasetConfig("cifar10", "../datasets", 32, 3),
        DatasetConfig("pets", "../datasets/oxford-iiit-pet", 128, 3), 
    ]
    
    sweep_space = {
        "base_channels": [16,32],
        "channel_mults": [(1,2,4), (1,2,2)],
        "num_res_blocks": [1, 2],
        "dropout": [0.0, 0.1],
        "lr": [1e-4, 3e-4],
        "batch_size": [32, 64],
        "num_timesteps": [200, 400]
    }
    
    rng = random.Random(42)
    
    for dcfg in d_cfgs:
        print(f"\n=== 1. Sensitivity Sweep: {dcfg.name} ===")
        sens = sensitivity_sweep(dcfg, get_baseline_config(), sweep_space)
        
        # Build a narrowed search space based on sensitivity
        narrow_space = {}
        for p, info in sens.items():
            # If a parameter was important, keep the top 2 values; otherwise keep the best 1.
            narrow_space[p] = info["best"] if info["importance"] > 0.005 else [info["best"][0]]

        print(f"=== 2. Randomized Search: {dcfg.name} ===")
        best_hcfg = None
        best_loss = float("inf")
        
        for i in range(5): # Adjust trials as needed
            hcfg = sample_from_space(rng, narrow_space)
            print(f"Trial {i+1}: {hcfg}")
            val = train_model(dcfg, hcfg, max_epochs=3, seed=i)
            if val < best_loss:
                best_loss = val
                best_hcfg = hcfg

        print(f"=== 3. Final Training: {dcfg.name} ===")
        train_model(dcfg, best_hcfg, max_epochs=30, seed=123, save_dir=f"final_{dcfg.name}")

if __name__ == "__main__":
    main()