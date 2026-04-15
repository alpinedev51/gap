import os
from sklearn import datasets
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Russian_Blue",
    "Siamese", "Sphynx"
}

class OxfordPetDataset(Dataset):
    def __init__(self, binary=False, transform=None):
        self.root = "../datasets/oxford-iiit-pet/images/images"
        self.transform = transform

        self.files = [f for f in os.listdir(self.root) if f.lower().endswith(".jpg")]
        self.files.sort()

        if binary:
            for fname in self.files:
                breed = self._extract_class_name(fname)
                label = 0 if breed in CAT_BREEDS else 1
                self.labels.append(label)
            
            self.classes = ["cat", "dog"]
        else:
            # Extract class names
            class_names = [self._extract_class_name(f) for f in self.files]
            self.classes = sorted(list(set(class_names)))

            # Map class → index
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

            # Build label list
            self.labels = [self.class_to_idx[self._extract_class_name(f)] for f in self.files]

    def _extract_class_name(self, filename):
        # Remove trailing "_<number>.jpg"
        base = filename.rsplit("_", 1)[0]
        return base

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.files[idx])
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label
    
def get_pets_data(binary=False):
    return OxfordPetDataset(binary=binary)

def get_cifar10_data():
    return datasets.CIFAR10(root="../datasets", train=True, download=True, transform=None)

