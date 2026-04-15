import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import random
import numpy as np
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ScalableResNetLite(nn.Module):
    def __init__(self, channels=64, depth=3, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        layers = []
        c = channels
        for i in range(depth):
            stride = 1 if i == 0 else 2
            out_c = c if i == 0 else c * 2
            layers.append(ResidualBlock(c, out_c, stride))
            c = out_c

        self.res_layers = nn.Sequential(*layers)
        self.linear = nn.Linear(c, num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.res_layers(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return self.linear(out)
    
def load_model_from_checkpoint(weights_path, metadata_path, device):
    """
    Reconstructs a ScalableResNetLite model from saved weights + metadata.

    Args:
        weights_path (str): Path to .pth file containing state_dict.
        metadata_path (str): Path to JSON metadata file.
        device (torch.device): cpu or cuda.

    Returns:
        model (nn.Module): Loaded and ready-to-use model.
        metadata (dict): Parsed metadata dictionary.
    """
    # --- Load metadata ---
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    cfg = metadata["model_config"]
    channels = cfg["channels"]
    depth = cfg["depth"]
    num_classes = cfg["num_classes"]

    # --- Rebuild model ---
    model = ScalableResNetLite(
        channels=channels,
        depth=depth,
        num_classes=num_classes,
    ).to(device)

    # --- Load weights ---
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, metadata
