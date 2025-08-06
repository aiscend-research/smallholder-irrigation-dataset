import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from tqdm import tqdm

MASK_CLASS_NAMES = [["No irrigation", "Small-scale", "Tree crop", "Industrial", "Lawn", "Covered"], ["No irrigation", "Irrigation"]]
MASK_COLORS = [["#cccccc", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"], ["#cccccc", "#1f77b4"]]

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

def plot_ml_predictions(dataset, clf, class_names=MASK_CLASS_NAMES, colors=MASK_COLORS, num_samples=5, save_path=None):
    """
    Predicts and visualizes ML results for a dataset using custom class labels and colors.
    Supports both single- and multi-band masks.

    Parameters:
        dataset: torch Dataset with 'image' and 'mask'
        clf: trained classifier (single or multioutput)
        class_names: list of class names (or list of lists if different per band)
        colors: list of color hex codes (same length as class_names, or list of lists)
        num_samples: number of samples to visualize
        save_path: optional file path to save the resulting plot
    """
    # Determine single-band or multi-band mask
    sample = dataset[0]
    mask_tensor = sample['mask']
    is_multiband = (mask_tensor.ndim == 3)
    num_bands = mask_tensor.shape[0] if is_multiband else 1

    # Standardize class_names/colors (handle per-band or same for all)
    if isinstance(class_names[0], list):
        class_names_list = class_names
    else:
        class_names_list = [class_names] * num_bands
    if isinstance(colors[0], list) or isinstance(colors[0], tuple):
        colors_list = colors
    else:
        colors_list = [colors] * num_bands

    fig, axes = plt.subplots(num_samples, num_bands * 2, figsize=(4 * num_bands * 2, 4 * num_samples))

    for i in range(num_samples):
        sample = dataset[i]
        image_tensor = sample['image']
        mask_tensor = sample['mask']
        if not is_multiband:
            mask_tensor = mask_tensor.unsqueeze(0)
        H, W = mask_tensor.shape[1:]
        C, T = image_tensor.shape[0:2]

        # Convert tensors to numpy arrays for compatibility with sklearn and plotting
        image = image_tensor.permute(2, 3, 1, 0).reshape(H * W, T * C)
        mask = mask_tensor.reshape(num_bands, H * W)

        # Convert to numpy if tensor
        if hasattr(image, 'numpy'):
            image = image.numpy()
        if hasattr(mask, 'numpy'):
            mask = mask.numpy()

        # Filter valid pixels for *all* bands
        valid = np.all(mask != -1, axis=0)
        X = image[valid]
        y_true = mask[:, valid].T  # shape: (num_pixels, num_bands)

        # Predict (multi-output if needed)
        y_pred = clf.predict(X)
        if num_bands == 1:
            y_pred = y_pred.reshape(-1, 1)
        # Only plot as many bands as are predicted (e.g., if model only predicts 2 bands, plot 2)
        n_pred_bands = y_pred.shape[1]

        for b in range(n_pred_bands):
            # Reconstruct masks
            gt_mask = reconstruct_mask(y_true[:, b], valid, H, W)
            pred_mask = reconstruct_mask(y_pred[:, b], valid, H, W)

            # Colormap and legend
            cmap, norm, legend_elements = create_colormap_and_legend(class_names_list[b], colors_list[b])

            # Axes indexing
            if num_samples > 1:
                ax_gt = axes[i, b * 2]
                ax_pred = axes[i, b * 2 + 1]
            else:
                ax_gt = axes[b * 2]
                ax_pred = axes[b * 2 + 1]

            ax_gt.imshow(gt_mask, cmap=cmap, norm=norm)
            ax_gt.set_title(f"Sample {i} Band {b} GT")
            ax_gt.axis('off')

            ax_pred.imshow(pred_mask, cmap=cmap, norm=norm)
            ax_pred.set_title(f"Sample {i} Band {b} Pred")
            ax_pred.axis('off')

            # Only show legends once per band (on first sample row)
            if i == 0:
                ax_pred.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', title=f"Classes (Band {b})")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close()
    else:
        plt.show()