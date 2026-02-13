import torch
from sklearn.datasets import make_swiss_roll
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from gap.utils import get_default_device


def get_data_loader(dataset_name, batch_size=32, train=True, download=True):
    """
    Factory function to get DataLoader for any dataset for the project.
    """

    device = get_default_device()
    use_pin_memory = True if device.type == "cuda" else False

    if dataset_name.lower() == "swiss_roll":
        dataset = _load_swiss_roll(train)
    elif dataset_name.lower() == "cifar10":
        dataset = _load_cifar10(train, download)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        pin_memory=use_pin_memory,
        num_workers=2 if device.type == "cuda" else 0
        )
    return loader

def _load_swiss_roll(train=True, total_samples=6000, train_size=5000):
    """
    Generates the full dataset and explicitly slices it to ensure
    Train and Test sets are disjoint.
    """
    x_all, t_all = make_swiss_roll(n_samples=total_samples, noise=0.1, random_state=42)
    
    x_all = x_all.astype("float32")
    t_all = t_all.astype("float32")

    train_x = x_all[:train_size]
    test_x  = x_all[train_size:]
    
    train_t = t_all[:train_size]
    test_t  = t_all[train_size:]

    x_mean = train_x.mean(axis=0)
    x_std  = train_x.std(axis=0)

    if train:
        x_norm = (train_x - x_mean) / x_std
        t_out  = train_t
    else:
        x_norm = (test_x - x_mean) / x_std
        t_out  = test_t

    return TensorDataset(torch.from_numpy(x_norm), torch.from_numpy(t_out))

def _load_cifar10(train=True, download=True):
    training_set_for_stats = datasets.CIFAR10(root="./data", train=True, download=download, transform=transforms.ToTensor())

    print("Calculating CIFAR10 stats...")
    imgs = torch.stack([img for img, _ in training_set_for_stats], dim=0)

    mean = imgs.mean(dim=(0, 2, 3))
    std = imgs.std(dim=(0, 2, 3))

    print(f"Computed Mean: {mean}")
    print(f"Computed Std: {std}")

    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    if train:
        training_set_for_stats.transform = final_transform
        return training_set_for_stats
    else:
        return datasets.CIFAR10(root="./data", train=False, download=download, transform=final_transform)

def get_swiss_roll(n=5000):
    x, _ = make_swiss_roll(n_samples=n, noise=0.1, random_state=0)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    return torch.tensor((x - x_mean) / x_std, dtype=torch.float32)
