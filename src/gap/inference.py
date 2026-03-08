import json
import os

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from gap.classifiers import CustomCNN


class UniversalPredictor:
    def __init__(self, model_folder):
        self.model_folder = model_folder
        self.meta = self._load_meta()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self._load_model()

    def _load_meta(self):
        with open(os.path.join(self.model_folder, "metadata.json"), "r") as f:
            return json.load(f)

    def _load_model(self):
        model_type = self.meta['model_type']

        if model_type == "CustomCNN":
            # Reconstruct the architecture using metadata
            model = CustomCNN(
                num_classes=len(self.meta.get('classes', [0]*37)), 
                input_size=self.meta.get('image_size', 128)
            )
            weights_path = os.path.join(self.model_folder, "weights.pth")
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            return model

        elif model_type == "LogisticRegressionCV":
            weights_path = os.path.join(self.model_folder, "model.pkl")
            return joblib.load(weights_path)

        else:
            raise ValueError(f"Unknown model type in metadata: {model_type}")

    def predict(self, image_path):
        """Takes a path to a new image and returns the predicted breed."""
        img = Image.open(image_path).convert('RGB')

        if self.meta['model_type'] == "CustomCNN":
            # CNN Pipeline
            size = self.meta.get('image_size', 128)
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                _, pred_idx = torch.max(output, 1)
                class_idx = pred_idx.item()

        else:
            # Logistic Regression Pipeline
            size = self.meta.get('image_size_flattened', 64)
            img_flat = np.array(img.resize((size, size))).flatten() / 255.0
            class_idx = self.model.predict([img_flat])[0]

        # Map index back to name
        breed_name = self.meta['classes'][class_idx]
        return breed_name

if __name__ == "__main__":
    MY_MODEL_RUN = "../saved_models/CustomCNN_10Epochs_20260306_1620"

    predictor = UniversalPredictor(MY_MODEL_RUN)

    # TODO: Change path and test
    test_image = "path/to/some_random_dog_pic.jpg"
    if os.path.exists(test_image):
        result = predictor.predict(test_image)
        print(f"The model thinks this is a: {result}")
