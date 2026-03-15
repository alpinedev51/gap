import os
import re

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class PetDataset(Dataset):
    """Custom PyTorch Dataset for the Oxford-IIIT Pets."""
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx]['filename'])
        image = Image.open(img_name).convert('RGB')
        label = self.df.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)
        return image, label

class PetDataProcessor:
    """Manages data loading for both Deep Learning and Classical ML."""
    def __init__(self, data_dir, test_size=0.2, random_state=42):
        self.data_dir = data_dir
        self.encoder = LabelEncoder()
        self.train_df, self.test_df = self._prepare_dataframe(test_size, random_state)
        self.num_classes = len(self.encoder.classes_)

    def _prepare_dataframe(self, test_size, random_state):
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.jpg')]
        # Extract breed name
        labels = [re.findall(r'^(.*)_\d+.jpg$', f)[0] for f in all_files]

        df = pd.DataFrame({'filename': all_files, 'label_name': labels})
        df['label'] = self.encoder.fit_transform(df['label_name'])

        return train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['label'])

    def get_cnn_dataloaders(self, batch_size=32, img_size=224):
        """Returns PyTorch DataLoaders for CNNs."""
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        train_ds = PetDataset(self.train_df, self.data_dir, transform)
        test_ds = PetDataset(self.test_df, self.data_dir, transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    def get_flattened_data(self, img_size=64):
        """Returns flattened X, y NumPy arrays for Scikit-Learn.
           Downscales images heavily to prevent RAM overload."""
        def _process_df(df):
            X, y = [], []
            for _, row in df.iterrows():
                img_path = os.path.join(self.data_dir, row['filename'])
                # Resize and flatten to a 1D array
                img = Image.open(img_path).convert('RGB').resize((img_size, img_size))
                X.append(np.array(img).flatten())
                y.append(row['label'])
            return np.array(X), np.array(y)

        print("Flattening images for classical ML (this may take a minute)...")
        X_train, y_train = _process_df(self.train_df)
        X_test, y_test = _process_df(self.test_df)

        # scale pixels to 0-1 range
        return X_train / 255.0, X_test / 255.0, y_train, y_test
