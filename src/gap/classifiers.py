import copy

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report


class CustomCNN(nn.Module):
    """
    A 7-Layer CNN (5 Conv + 2 Linear) optimized for fine-grained classification.

    --- CONVOLUTIONAL MATH ---
    For every Conv2d layer, the output spatial dimension (W_out) is calculated as:
    $$W_{out} = \\left\\lfloor\\frac{W_{in} - K + 2P}{S}\\right\\rfloor + 1$$
    Where:
      - W_in = Input width/height
      - K = Kernel size (3)
      - P = Padding (1)
      - S = Stride (1)
    Because K=3, P=1, and S=1, the spatial dimension remains unchanged by the convolutions:
    $$W_{out} = W_{in} - 3 + 2 + 1 = W_{in}$$

    --- POOLING MATH ---
    The MaxPool2d(2, 2) layer halves the spatial dimensions:
    $$W_{out} = \\left\\lfloor\\frac{W_{in} - 2}{2}\\right\\rfloor + 1 = \\frac{W_{in}}{2}$$

    --- TENSOR LIFECYCLE (Assuming 224x224 input) ---
    Input:       (Batch, 3, 224, 224)
    Block 1:     (Batch, 32, 112, 112)
    Block 2:     (Batch, 64, 56, 56)
    Block 3:     (Batch, 128, 28, 28)
    Block 4:     (Batch, 256, 14, 14)
    Block 5:     (Batch, 512, 7, 7)
    Adapt. Pool: (Batch, 512, 2, 2)
    Flatten:     (Batch, 2048)
    Linear 1:    (Batch, 512)
    Linear 2:    (Batch, num_classes)
    """

    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()

        # --- FEATURE EXTRACTOR (5 Convolutional Layers) ---
        self.features = nn.Sequential(
            # Block 1 (Layer 1)
            # Input: (B, 3, 224, 224) -> Conv -> (B, 32, 224, 224) -> Pool -> (B, 32, 112, 112)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2 (Layer 2)
            # Input: (B, 32, 112, 112) -> Conv -> (B, 64, 112, 112) -> Pool -> (B, 64, 56, 56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3 (Layer 3)
            # Input: (B, 64, 56, 56) -> Conv -> (B, 128, 56, 56) -> Pool -> (B, 128, 28, 28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4 (Layer 4)
            # Input: (B, 128, 28, 28) -> Conv -> (B, 256, 28, 28) -> Pool -> (B, 256, 14, 14)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 5 (Layer 5) - Extracts highly complex semantic features
            # Input: (B, 256, 14, 14) -> Conv -> (B, 512, 14, 14) -> Pool -> (B, 512, 7, 7)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Adaptive Pooling: Safely forces any input to exactly 512 channels x 2 x 2 spatial size
        # Input: (B, 512, 7, 7) [if 224x224] -> Output: (B, 512, 2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # --- CLASSIFIER (2 Linear Layers) ---
        self.classifier = nn.Sequential(
            # Input: (B, 512, 2, 2) -> Output: (B, 2048)
            nn.Flatten(),
            # Layer 6
            # Input: (B, 2048) -> Output: (B, 512)
            nn.Linear(512 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            # Layer 7 (Output)
            # Input: (B, 512) -> Output: (B, num_classes)
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x starts as (Batch, 3, H, W)
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        # x ends as (Batch, num_classes) containing raw logits
        return x


def train_cnn(model, train_loader, val_loader, epochs, learning_rate, device):
    """
    Trains the PyTorch CNN and evaluates on the validation set per epoch.
    Automatically restores the best model weights based on validation accuracy.
    Returns the best model and a dictionary of training history.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    print(f"Starting Training for {epochs} Epochs...\n")

    for epoch in range(epochs):
        # ========================================
        # 1. TRAINING PHASE
        # ========================================
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()

        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct / len(train_loader.dataset)

        # ========================================
        # 2. VALIDATION PHASE
        # ========================================
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = val_correct / len(val_loader.dataset)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch [{epoch + 1:02d}/{epochs:02d}] | "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}"
        )

        # ========================================
        # 3. SAVE BEST MODEL
        # ========================================
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"   Validation accuracy improved to {best_acc:.4f}!")

    print(
        f"\nTraining Complete. Restoring best model weights (Val Acc: {best_acc:.4f})"
    )
    model.load_state_dict(best_model_wts)

    return model, history


def evaluate_cnn(model, test_loader, device):
    """Evaluates the PyTorch CNN."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"CNN Test Accuracy: {acc:.4f}")
    return all_labels, all_preds


# ==========================================
# 2. CLASSICAL MACHINE LEARNING (Scikit-Learn)
# ==========================================


def train_logistic_regression_cv(X_train, y_train, cv_folds=3, max_iter=500):
    """
    Trains a Logistic Regression model using Cross-Validation to automatically
    find the best 'C' (inverse regularization strength) hyperparameter.
    """
    print(f"Training LogisticRegressionCV with {cv_folds} folds...")
    lrcv = LogisticRegressionCV(
        Cs=10,
        cv=cv_folds,
        max_iter=max_iter,
        n_jobs=-1,
        random_state=42,  # Added seed for reproducibility here too
    )
    lrcv.fit(X_train, y_train)
    print(f"Best C parameters found (sample): {lrcv.C_[0]}")
    return lrcv


def evaluate_classical_model(model, X_test, y_test, target_names=None):
    """Evaluates a Scikit-Learn model."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Classical Model Test Accuracy: {acc:.4f}")

    if target_names is not None:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

    return y_test, y_pred
