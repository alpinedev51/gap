import torch
from sklearn.datasets import make_swiss_roll


def get_swiss_roll(n=5000):
    x, _ = make_swiss_roll(n_samples=n, noise=0.1, random_state=0)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    return torch.tensor((x - x_mean) / x_std, dtype=torch.float32)
