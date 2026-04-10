import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, classes):
    """Generates a large, readable confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 16))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation="vertical", colorbar=False)

    plt.title("Confusion Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_roc_auc(y_true, y_prob, num_classes):
    """Calculates and plots the Macro-Average ROC curve."""
    # Binarize labels for One-vs-Rest (OvR)
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    fpr, tpr, roc_auc = dict(), dict(), dict()

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(num_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= num_classes
    macro_auc = auc(all_fpr, mean_tpr)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.plot(
        all_fpr,
        mean_tpr,
        color="darkorange",
        linewidth=4,
        label=f"Macro-average ROC curve (AUC = {macro_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], "k--", linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


def visualize_random_predictions(model, dataset, classes, device, num_images=6):
    """Plots a grid of images with their predicted and actual labels."""
    model.eval()
    fig = plt.figure(figsize=(15, 10))

    # pick random indices from the validation set
    indices = np.random.choice(len(dataset), num_images, replace=False)

    for i, idx in enumerate(indices):
        image_tensor, actual_label_idx = dataset[idx]

        # run inference
        input_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            _, pred_idx = torch.max(output, 1)

        # prepare image for display (un-normalize using our standard 0.5 values)
        img = image_tensor.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        # labeling
        actual_name = classes[actual_label_idx]
        pred_name = classes[pred_idx.item()]
        color = "green" if actual_name == pred_name else "red"

        ax = fig.add_subplot(2, 3, i + 1, xticks=[], yticks=[])
        ax.imshow(img)
        ax.set_title(f"Pred: {pred_name}\nActual: {actual_name}", color=color)
    plt.show()
