import numpy as np
import torch
from torch import nn


class ScoreModel3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256), 
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 3) 
        )

    def forward(self, x, sigma):
        # passing log(sigma) helps the network learn
        # exponential scales of noise better than raw sigma
        log_sigma = torch.log(sigma).view(-1, 1).expand(x.shape[0], 1)
        inputs = torch.cat([x, log_sigma], dim=-1)
        return self.net(inputs)


class GaussianFourierProjection(nn.Module):
    """Embeds scalars into a high-dimensional sinusoidal space."""
    def __init__(self, embed_dim=256, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class GScoreModel3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = GaussianFourierProjection(embed_dim=256, scale=30.0)

        self.input_proj = nn.Linear(3, 256)

        self.net = nn.Sequential(
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x, sigma):
        embed_sigma = self.embed(sigma.view(-1)) # [Batch, 256]
        embed_x = self.input_proj(x)              # [Batch, 256]

        h = torch.cat([embed_x, embed_sigma], dim=-1)

        return self.net(h) / sigma.view(-1, 1)
