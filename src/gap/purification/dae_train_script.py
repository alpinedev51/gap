import gc
import os
import random
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

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
    bottleneck_dim: int
    dropout: float
    lr: float
    batch_size: int
    noise_std: float

# ============================================================
#  Shallow Convolutional DAE
# ============================================================

class ConvEncoder(nn.Module):
    def __init__(self, in_channels, base_channels, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class ConvDecoder(nn.Module):
    def __init__(self, out_channels, base_channels, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(base_channels, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class ShallowDAE(nn.Module):
    def __init__(self, in_channels, base_channels, bottleneck_dim, dropout, noise_std):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, base_channels, dropout)
        self.bottleneck = nn.Conv2d(base_channels * 4, bottleneck_dim, 1)
        self.decoder_input = nn.Conv2d(bottleneck_dim, base_channels * 4, 1)
        self.decoder = ConvDecoder(in_channels, base_channels, dropout)
        self.noise_std = noise_std

    def forward(self, x):
        z = self.encoder(x)
        z = self.bottleneck(z)
        z = self.decoder_input(z)
        return self.decoder(z)

    def loss(self, clean_x):
        noise = torch.randn_like(clean_x) * self.noise_std
        noisy_x = clean_x + noise
        recon = self.forward(noisy_x)
        return F.mse_loss(recon, clean_x)

# ============================================================
#  Data & Training
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

    dae = ShallowDAE(
        dataset_cfg.in_channels,
        hcfg.base_channels,
        hcfg.bottleneck_dim,
        hcfg.dropout,
        hcfg.noise_std
    ).to(device)

    opt = torch.optim.AdamW(dae.parameters(), lr=hcfg.lr)

    best_val = float("inf")
    try:
        for epoch in range(max_epochs):
            dae.train()
            for x, _ in tqdm(train_loader, desc=f"Ep {epoch+1}", leave=False):
                x = x.to(device)
                loss = dae.loss(x)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            dae.eval()
            v_loss = 0.0
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    v_loss += dae.loss(x).item() * x.size(0)

            v_loss /= len(val_loader.dataset)
            print(f"  - {dataset_cfg.name} Epoch {epoch+1} Val Loss: {v_loss:.6f}")

            if v_loss < best_val:
                best_val = v_loss
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({"model": dae.state_dict(), "hcfg": asdict(hcfg)}, f"{save_dir}/best_model.pt")

    finally:
        del train_loader, val_loader, dae, opt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    return best_val

# ============================================================
#  Sweep Logic
# ============================================================

def get_baseline_config():
    return HyperConfig(
        base_channels=32,
        bottleneck_dim=128,
        dropout=0.1,
        lr=3e-4,
        batch_size=64,
        noise_std=0.1
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
            "best": [x[0] for x in param_res[:2]]
        }
    return results

def sample_from_space(rng, space):
    return HyperConfig(
        base_channels=rng.choice(space["base_channels"]),
        bottleneck_dim=rng.choice(space["bottleneck_dim"]),
        dropout=rng.choice(space["dropout"]),
        lr=rng.choice(space["lr"]),
        batch_size=rng.choice(space["batch_size"]),
        noise_std=rng.choice(space["noise_std"])
    )

# ============================================================
#  Main
# ============================================================

def main():
    d_cfgs = [
        DatasetConfig("pets_binary", "../datasets/oxford-iiit-pet", 128, 3),
        DatasetConfig("cifar10", "../datasets", 32, 3),
        DatasetConfig("pets", "../datasets/oxford-iiit-pet", 128, 3),
    ]

    sweep_space = {
        "base_channels": [16, 32],
        "bottleneck_dim": [64, 128, 256],
        "dropout": [0.0, 0.1],
        "lr": [1e-4, 3e-4],
        "batch_size": [32, 64],
        "noise_std": [0.05, 0.1, 0.2],
    }

    rng = random.Random(42)

    for dcfg in d_cfgs:
        print(f"\n=== 1. Sensitivity Sweep: {dcfg.name} ===")
        sens = sensitivity_sweep(dcfg, get_baseline_config(), sweep_space)

        narrow_space = {
            p: info["best"] if info["importance"] > 0.005 else [info["best"][0]]
            for p, info in sens.items()
        }

        print(f"=== 2. Randomized Search: {dcfg.name} ===")
        best_hcfg = None
        best_loss = float("inf")

        for i in range(10):
            hcfg = sample_from_space(rng, narrow_space)
            print(f"Trial {i+1}: {hcfg}")
            val = train_model(dcfg, hcfg, max_epochs=3, seed=i)
            if val < best_loss:
                best_loss = val
                best_hcfg = hcfg

        print(f"=== 3. Final Training: {dcfg.name} ===")
        train_model(dcfg, best_hcfg, max_epochs=30, seed=123, save_dir=f"saved_models/{dcfg.name}_dae")

if __name__ == "__main__":
    main()
