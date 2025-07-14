import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tqdm import tqdm

def print_confusion_matix(y_test, y_pred):
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')

def reconstruct_mask(flat, valid_mask, H, W, fill_value=-1):
    """Reconstructs full mask from flat valid pixel predictions."""
    full = np.full(H * W, fill_value)
    full[valid_mask] = flat
    return full.reshape(H, W)

def create_colormap_and_legend(class_names, colors):
    """Creates colormap, normalization, and legend for plotting."""
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(class_names) + 1) - 0.5, ncolors=len(class_names))
    legend_elements = [Patch(facecolor=colors[i], edgecolor='k', label=class_names[i]) for i in range(len(class_names))]
    return cmap, norm, legend_elements

def plot_ml_predictions(dataset, clf, class_names, colors, num_samples=5):
    """
    Predicts and visualizes RF results for a dataset using custom class labels and colors.
    
    Parameters:
        dataset: torch Dataset with 'image' and 'mask'
        clf: trained RandomForestClassifier
        class_names: list of class names
        colors: list of color hex codes (same length as class_names)
        num_samples: number of samples to visualize
    """
    cmap, norm, legend_elements = create_colormap_and_legend(class_names, colors)
    plt.figure(figsize=(12, 4 * num_samples))

    for i in range(num_samples):
        sample = dataset[i]
        image_tensor = sample['image']  # shape: (C, T, H, W)
        mask_tensor = sample['mask']    # shape: (H, W)
        H, W = mask_tensor.shape
        C, T = image_tensor.shape[0:2]

        # Flatten
        image = image_tensor.permute(2, 3, 1, 0).reshape(H * W, T * C)
        mask = mask_tensor.reshape(H * W)

        # Filter valid pixels
        valid = mask != -1
        X = image[valid].numpy()
        y = mask[valid].numpy()

        # Predict
        y_pred = clf.predict(X)

        # Reconstruct
        gt_mask = reconstruct_mask(y, valid.numpy(), H, W)
        pred_mask = reconstruct_mask(y_pred, valid.numpy(), H, W)

        # Plot
        plt.subplot(num_samples, 2, i * 2 + 1)
        plt.imshow(gt_mask, cmap=cmap, norm=norm)
        plt.title(f"[{i}] Ground Truth")
        plt.axis('off')

        plt.subplot(num_samples, 2, i * 2 + 2)
        plt.imshow(pred_mask, cmap=cmap, norm=norm)
        plt.title(f"[{i}] RF Prediction")
        plt.axis('off')

    plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title="Classes")
    plt.tight_layout()
    plt.show()
