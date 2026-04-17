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

# ============================================================
#  Configs
# ============================================================

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
    num_timesteps: int  # used for sampling discretization

# ============================================================
#  UNet backbone with continuous-time conditioning
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

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.lin1 = nn.Linear(dim, dim * 4)
        self.lin2 = nn.Linear(dim * 4, dim * 4)

    def forward(self, t):
        # t is continuous in [0, 1]
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.lin2(F.silu(self.lin1(emb)))

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
            else:
                hs.append(curr)
                curr = layer(curr)
        hs.append(curr)

        # --- MIDDLE ---
        curr = self.mid1(curr, t_emb)
        curr = self.mid2(curr, t_emb)

        # --- DECODER ---
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                if layer.conv1.in_channels > curr.shape[1]:
                    skip = hs.pop()
                    if curr.shape[-2:] != skip.shape[-2:]:
                        curr = F.interpolate(curr, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                    curr = torch.cat([curr, skip], dim=1)
                curr = layer(curr, t_emb)
            else:
                curr = layer(curr)

        return self.out_conv(F.silu(self.out_norm(curr)))

# ============================================================
#  Continuous-time VE-SDE + Langevin dynamics
#  (following "Score-Based Generative Modeling through SDEs")
# ============================================================

class ScoreSDEContinuous(nn.Module):
    """
    VE-SDE:
      d x = sqrt{d[σ(t)^2]} d w,   t in [0, 1]
      x(t) | x(0) ~ N(x(0), σ(t)^2 I)

    We train s_θ(x(t), t) to approximate ∇_x log p_t(x(t)),
    using denoising score matching with a time-dependent weight.
    """
    def __init__(self, model, sigma_min=0.01, sigma_max=50.0):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    # continuous noise scale
    def sigma(self, t):
        # Exponential schedule: σ(t) = σ_min * (σ_max / σ_min)^t
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    # training loss (continuous-time SDE objective)
    def loss(self, x0):
        b = x0.size(0)
        device = x0.device

        # sample continuous time t ~ Uniform(0, 1)
        t = torch.rand(b, device=device)

        # compute σ(t)
        sigma_t = self.sigma(t).view(b, 1, 1, 1)

        # perturb data
        noise = torch.randn_like(x0)
        x_t = x0 + sigma_t * noise

        # target score: ∇_x log p_t(x_t | x0) = -(x_t - x0) / σ(t)^2 = -noise / σ(t)
        target = -noise / sigma_t

        # model prediction
        s_theta = self.model(x_t, t)

        # time-dependent weighting λ(t) = σ(t)^2 (VE-SDE objective)
        weight = (sigma_t ** 2)

        return torch.mean(weight * (s_theta - target) ** 2)

    # discrete noise levels for sampling
    def get_discrete_sigmas(self, num_steps):
        # log-spaced from σ_max down to σ_min
        return torch.exp(
            torch.linspace(
                math.log(self.sigma_max),
                math.log(self.sigma_min),
                num_steps,
                device=device,
            )
        )

    # Annealed Langevin Dynamics (corrector) at a fixed noise level
    @torch.no_grad()
    def langevin_step(self, x, t, sigma_t, snr=0.16, n_steps=1):
        for _ in range(n_steps):
            grad = self.model(x, t)
            noise = torch.randn_like(x)
            # step size from SNR heuristic (Song et al.)
            step_size = (snr * sigma_t) ** 2 * 2.0
            x = x + step_size * grad + torch.sqrt(2.0 * step_size) * noise
        return x

    # Simple annealed Langevin sampler (no predictor, only corrector)
    @torch.no_grad()
    def sample_ald(self, shape, num_steps=1000, snr=0.16, n_inner=1):
        sigmas = self.get_discrete_sigmas(num_steps)
        x = torch.randn(*shape, device=device) * sigmas[0]

        for i in range(num_steps):
            sigma_t = sigmas[i]
            # map discrete index to continuous t in [0, 1]
            t = torch.full((shape[0],), i / max(num_steps - 1, 1), device=device)
            x = self.langevin_step(x, t, sigma_t, snr=snr, n_steps=n_inner)

        return x

# ============================================================
#  Data & Training Infrastructure
# ============================================================

def get_dataloaders(cfg, batch_size):
    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*cfg.in_channels, [0.5]*cfg.in_channels),
    ])

    if cfg.name == "cifar10":
        ds = get_cifar10_data()
    else:
        ds = get_pets_data(binary=("binary" in cfg.name))

    ds.transform = tfm
    v_sz = int(0.1 * len(ds))
    t_ds, v_ds = random_split(ds, [len(ds)-v_sz, v_sz])

    return (
        DataLoader(t_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
        DataLoader(v_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    )

def train_model(dataset_cfg, hcfg, max_epochs, seed, save_dir=None):
    torch.manual_seed(seed)
    train_loader, val_loader = get_dataloaders(dataset_cfg, hcfg.batch_size)

    model = UNetLite(
        dataset_cfg.in_channels,
        hcfg.base_channels,
        hcfg.channel_mults,
        hcfg.num_res_blocks,
        hcfg.dropout,
    ).to(device)

    sde = ScoreSDEContinuous(model).to(device)
    opt = torch.optim.AdamW(sde.parameters(), lr=hcfg.lr)

    best_val = float("inf")
    try:
        for epoch in range(max_epochs):
            sde.train()
            for x, _ in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False, disable=False):
                x = x.to(device)
                loss = sde.loss(x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            sde.eval()
            v_loss = 0.0
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    v_loss += sde.loss(x).item() * x.size(0)

            v_loss /= len(val_loader.dataset)
            print(f"  - {dataset_cfg.name} Epoch {epoch+1} Val Loss: {v_loss:.6f}")

            if v_loss < best_val:
                best_val = v_loss
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "hcfg": asdict(hcfg),
                            "sigma_min": sde.sigma_min,
                            "sigma_max": sde.sigma_max,
                        },
                        f"{save_dir}/best_model.pt"
                    )

    finally:
        del train_loader
        del val_loader
        del model
        del sde
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
    return HyperConfig(
        base_channels=32,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        dropout=0.1,
        lr=3e-4,
        batch_size=64,
        num_timesteps=500,  # used for sampling discretization
    )

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
        results[param] = {
            "importance": param_res[-1][1] - param_res[0][1],
            "best": [x[0] for x in param_res[:2]],
        }
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
        DatasetConfig("pets", "../datasets/oxford-iiit-pet", 128, 3),
        DatasetConfig("cifar10", "../datasets", 32, 3),
    ]

    sweep_space = {
        "base_channels": [16, 32],
        "channel_mults": [(1, 2, 4), (1, 2, 2)],
        "num_res_blocks": [1, 2],
        "dropout": [0.0, 0.1],
        "lr": [1e-4, 3e-4],
        "batch_size": [32, 64],
        "num_timesteps": [250, 500, 1000],  # for sampling resolution
    }

    rng = random.Random(42)

    for dcfg in d_cfgs:
        print(f"\n=== 1. Sensitivity Sweep: {dcfg.name} ===")
        sens = sensitivity_sweep(dcfg, get_baseline_config(), sweep_space)

        narrow_space = {}
        for p, info in sens.items():
            narrow_space[p] = info["best"] if info["importance"] > 0.005 else [info["best"][0]]

        print(f"=== 2. Randomized Search: {dcfg.name} ===")
        best_hcfg = None
        best_loss = float("inf")

        for i in range(5):
            hcfg = sample_from_space(rng, narrow_space)
            print(f"Trial {i+1}: {hcfg}")
            val = train_model(dcfg, hcfg, max_epochs=3, seed=i)
            if val < best_loss:
                best_loss = val
                best_hcfg = hcfg

        print(f"=== 3. Final Training: {dcfg.name} ===")
        train_model(dcfg, best_hcfg, max_epochs=30, seed=123, save_dir=f"saved_models/{dcfg.name}_sde")

if __name__ == "__main__":
    main()
