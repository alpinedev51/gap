import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report

import gap.visualize as visualize


class ModelEvaluator:
    def __init__(self, model, dataloader, classes, device):
        self.model = model
        self.dataloader = dataloader
        self.classes = classes
        self.device = device
        self.num_classes = len(classes)

        print("Running inference on full dataset for evaluation...")
        self.y_true, self.y_pred, self.y_prob = self._get_predictions()

    def _get_predictions(self):
        """Runs the entire dataloader through the model once to gather arrays."""
        self.model.eval()
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in self.dataloader:
                images = images.to(self.device)
                outputs = self.model(images)

                # Get probabilities using Softmax (required for ROC-AUC)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        return np.array(all_labels), np.array(all_preds), np.array(all_probs)

    def analyze_class_difficulty(self):
        """Prints text-based rankings of the easiest and hardest classes."""
        report = classification_report(self.y_true, self.y_pred, target_names=self.classes, output_dict=True)
        class_scores = {k: v['f1-score'] for k, v in report.items() if k in self.classes}

        sorted_classes = sorted(class_scores.items(), key=lambda x: x[1])
        hardest = sorted_classes[:5]
        easiest = sorted_classes[-5:]

        print("\n--- 🚨 TOP 5 HARDEST CLASSES (Lowest F1-Score) ---")
        for name, score in hardest:
            print(f"{name:.<25} {score:.4f}")

        print("\n--- 🏆 TOP 5 EASIEST CLASSES (Highest F1-Score) ---")
        for name, score in reversed(easiest):
            print(f"{name:.<25} {score:.4f}")

    def plot_confusion_matrix(self):
        visualize.plot_confusion_matrix(self.y_true, self.y_pred, self.classes)

    def plot_roc_auc(self):
        visualize.plot_roc_auc(self.y_true, self.y_prob, self.num_classes)

    def visualize_random_predictions(self, num_images=6):
        visualize.visualize_random_predictions(
            self.model, self.dataloader.dataset, self.classes, self.device, num_images
        )

    def generate_full_report(self):
        """Runs all evaluations and visualizations sequentially."""
        print("Generating Full Evaluation Report...\n")
        self.analyze_class_difficulty()
        print("\nGenerating Confusion Matrix...")
        self.plot_confusion_matrix()
        print("\nGenerating ROC-AUC Curve...")
        self.plot_roc_auc()
        print("\nVisualizing Sample Predictions...")
        self.visualize_random_predictions()
