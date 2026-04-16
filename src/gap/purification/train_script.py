import gc
import os
import math
import random
import json
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from gap.purification.data_loaders import get_pets_data, get_cifar10_data

# ============================================================
#  Config
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("HEREDevice:", device)

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
#  UNet + Diffusion
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim * 4)
        self.lin2 = nn.Linear(dim * 4, dim * 4)

    def forward(self, t):
        half_dim = self.lin1.in_features // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, device=t.device) / half_dim
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        emb = self.lin2(F.silu(self.lin1(emb)))
        return emb

class UNetLite(nn.Module):
    def __init__(self, in_channels, base_channels,
                 channel_mults=(1, 2, 4),
                 num_res_blocks=2,
                 dropout=0.1):
        super().__init__()

        self.time_mlp = TimeEmbedding(base_channels)

        self.in_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # -------------------------
        # Encoder (down path)
        # -------------------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult

            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(ch, out_ch, dropout))
                ch = out_ch

            self.down_blocks.append(blocks)

            # downsample between levels (except after last)
            if i != len(channel_mults) - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.downsamples.append(None)

        # -------------------------
        # Middle
        # -------------------------
        self.mid1 = ResidualBlock(ch, ch, dropout)
        self.mid2 = ResidualBlock(ch, ch, dropout)

        # -------------------------
        # Decoder (up path)
        # -------------------------
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        # We traverse channel_mults in reverse for the decoder
        for i, mult in enumerate(reversed(channel_mults)):
            skip_ch = base_channels * mult
            out_ch = skip_ch  # final channels at this level

            blocks = nn.ModuleList()

             # First block takes concatenated input
            blocks.append(ResidualBlock(ch + skip_ch, out_ch, dropout))
            ch = out_ch

            # Remaining blocks just take ch
            for _ in range(num_res_blocks - 1):
                blocks.append(ResidualBlock(ch, out_ch, dropout))

            self.up_blocks.append(blocks)

            # upsample between levels (except after last)
            if i != len(channel_mults) - 1:
                self.upsamples.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.upsamples.append(None)

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)

    def forward(self, x, t):
        # t is currently unused in the blocks, but kept for future conditioning
        _ = self.time_mlp(t)

        h = self.in_conv(x)
        skips = []

        # -------------------------
        # Down path
        # -------------------------
        for blocks, downsample in zip(self.down_blocks, self.downsamples):
            for block in blocks:
                h = block(h)
            # one skip per resolution level
            skips.append(h)

            if downsample is not None:
                h = downsample(h)

        # -------------------------
        # Middle
        # -------------------------
        h = self.mid1(h)
        h = self.mid2(h)

        # -------------------------
        # Up path
        # -------------------------
        for blocks, upsample in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)

            for block in blocks:
                h = block(h)

            if upsample is not None:
                h = upsample(h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h

class GaussianDiffusion(nn.Module):
    def __init__(self, model, num_timesteps=1000):
        super().__init__()
        self.model = model
        self.num_timesteps = num_timesteps

        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_om * noise, noise

    def p_losses(self, x0, t):
        x_noisy, noise = self.q_sample(x0, t)
        noise_pred = self.model(x_noisy, t)
        return F.mse_loss(noise_pred, noise)


# ============================================================
#  Data
# ============================================================

def get_dataset(cfg: DatasetConfig):
    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * cfg.in_channels, [0.5] * cfg.in_channels),
    ])

    if cfg.name == "cifar10":
        ds = get_cifar10_data()
    elif cfg.name == "pets":
        ds = get_pets_data(binary=False)
    elif cfg.name == "pets_binary":
        ds = get_pets_data(binary=True)
    else:
        raise ValueError(f"Unknown dataset: {cfg.name}")

    ds.transform = tfm
    return ds

def get_dataloaders(cfg: DatasetConfig, batch_size):
    ds = get_dataset(cfg)
    val_size = int(0.1 * len(ds))
    train_size = len(ds) - val_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True,
                              persistent_workers=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            persistent_workers=False)
    return train_loader, val_loader

# ============================================================
#  Training utilities
# ============================================================

def train_model(dataset_cfg, hcfg, max_epochs, seed, save_dir=None):
    torch.manual_seed(seed)
    random.seed(seed)

    train_loader, val_loader = get_dataloaders(dataset_cfg, hcfg.batch_size)

    model = UNetLite(
        in_channels=dataset_cfg.in_channels,
        base_channels=hcfg.base_channels,
        channel_mults=hcfg.channel_mults,
        num_res_blocks=hcfg.num_res_blocks,
        dropout=hcfg.dropout,
    ).to(device)

    diffusion = GaussianDiffusion(model, num_timesteps=hcfg.num_timesteps).to(device)
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=hcfg.lr)

    best_val = float("inf")

    for epoch in range(max_epochs):
        diffusion.train()
        total_loss = 0.0

        for x, _ in tqdm(train_loader, desc=f"Train {dataset_cfg.name} ep: {epoch+1}/{max_epochs}", leave=False):
            x = x.to(device)
            t = torch.randint(0, hcfg.num_timesteps, (x.size(0),), device=device)
            loss = diffusion.p_losses(x, t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        # Validation
        diffusion.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in tqdm(val_loader, desc=f"Val {dataset_cfg.name} ep: {epoch+1}/{max_epochs}", leave=False):
                x = x.to(device)
                t = torch.randint(0, hcfg.num_timesteps, (x.size(0),), device=device)
                loss = diffusion.p_losses(x, t)
                val_loss += loss.item() * x.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"[{dataset_cfg.name}] Epoch {epoch+1}/{max_epochs} | "
              f"train={train_loss:.4f} val={val_loss:.4f}")

        # Save only during final training
        if save_dir is not None and val_loss < best_val:
            best_val = val_loss
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "best_model.pt")

            torch.save({
                "model_state": model.state_dict(),
                "hyperparams": asdict(hcfg),
                "dataset_cfg": asdict(dataset_cfg),
                "val_loss": best_val,
            }, save_path)

            print(f"  -> Saved best model to {save_path}")

    return best_val


# ============================================================
#  Hyperparameter search utilities (NEW)
# ============================================================

def get_baseline_config():
    return HyperConfig(
        base_channels=48,
        channel_mults=(1, 2, 4),
        num_res_blocks=2,
        dropout=0.1,
        lr=3e-4,
        batch_size=64,
        num_timesteps=400,
    )


def sensitivity_sweep(dataset_cfg, baseline, sweep_space, epochs=2):
    """
    Sweep one parameter at a time while keeping others fixed.
    Returns importance scores + best values.
    """
    print(f"\n=== Sensitivity Sweep: {dataset_cfg.name} ===")

    results = {}

    for param, values in sweep_space.items():
        print(f"\n-- Sweeping {param} --")

        param_results = []

        for v in values:
            hcfg = HyperConfig(**asdict(baseline))
            setattr(hcfg, param, v)

            val = train_model(dataset_cfg, hcfg, epochs, seed=0, save_dir=None)
            param_results.append((v, val))

            torch.cuda.empty_cache()
            gc.collect()

        # Sort by performance (lower is better)
        param_results.sort(key=lambda x: x[1])

        best_val = param_results[0][1]
        worst_val = param_results[-1][1]

        importance = worst_val - best_val

        results[param] = {
            "results": param_results,
            "importance": importance,
            "best_values": [v for v, _ in param_results[:2]]  # top 2
        }

        print(f"{param} importance: {importance:.4f}")
        print(f"Best values: {[v for v, _ in param_results[:2]]}")

    return results


def build_search_space_from_sensitivity(sensitivity_results):
    """
    Narrow search space based on sensitivity results.
    """
    search_space = {}

    for param, info in sensitivity_results.items():
        if info["importance"] < 0.01:
            # Not important → keep default only
            search_space[param] = info["best_values"][:1]
        else:
            # Important → keep top performers
            search_space[param] = info["best_values"]

    return search_space


def sample_from_search_space(rng, space):
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
#  Main pipeline (REWRITTEN)
# ============================================================

if __name__ == "__main__":
    datasets_cfg = [
        DatasetConfig("pets", "../datasets/oxford-iiit-pet", 128, 3),
        DatasetConfig("pets_binary", "../datasets/oxford-iiit-pet", 128, 3),
        DatasetConfig("cifar10", "../datasets", 32, 3),
    ]

    rng = random.Random(42)

    # -------------------------
    # 1. Sensitivity sweep (coarse search)
    # -------------------------
    sweep_space = {
        "base_channels": [32, 48, 64],
        "channel_mults": [(1, 2, 4), (1, 2, 2)],
        "num_res_blocks": [1, 2],
        "dropout": [0.0, 0.1, 0.2],
        "lr": [1e-4, 3e-4, 5e-4],
        "batch_size": [32, 64],
        "num_timesteps": [200, 400, 600],
    }

    sensitivity_epochs = 2

    sensitivity_results_all = {}
    search_spaces = {}

    for dcfg in datasets_cfg:
        baseline = get_baseline_config()

        sens_results = sensitivity_sweep(
            dcfg,
            baseline,
            sweep_space,
            epochs=sensitivity_epochs
        )

        sensitivity_results_all[dcfg.name] = sens_results

        search_space = build_search_space_from_sensitivity(sens_results)
        search_spaces[dcfg.name] = search_space

    # -------------------------
    # 2. Randomized search (refined)
    # -------------------------
    print("\n=== Refined Random Search ===")

    search_trials = 10
    search_epochs = 3

    best_cfgs = {}

    for dcfg in datasets_cfg:
        print(f"\nDataset: {dcfg.name}")

        best_loss = float("inf")
        best_hcfg = None

        space = search_spaces[dcfg.name]

        for i in range(search_trials):
            hcfg = sample_from_search_space(rng, space)

            print(f"Trial {i+1}/{search_trials}: {hcfg}")

            val = train_model(dcfg, hcfg, search_epochs, seed=i, save_dir=None)

            if val < best_loss:
                best_loss = val
                best_hcfg = hcfg

        best_cfgs[dcfg.name] = best_hcfg

    # -------------------------
    # 3. Final training
    # -------------------------
    print("\n=== Final Training ===")

    final_epochs = 30

    for dcfg in datasets_cfg:
        print(f"\n[{dcfg.name}] Final training")

        train_model(
            dcfg,
            best_cfgs[dcfg.name],
            final_epochs,
            seed=123,
            save_dir=f"final_{dcfg.name}"
        )

    print("\nAll done.")