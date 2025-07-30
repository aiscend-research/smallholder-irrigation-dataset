import os
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio

class MultiTemporalCropDataset(Dataset):
    def __init__(self, image_dir, label_dir, sample_file_list, label_bands=list(range(1, 9))):
        """
        Args:
            image_dir (str): Path to directory containing Sentinel-2 input .tif files.
            label_dir (str): Path to directory containing label .tif files.
            sample_file_list (list of str): List of sample base filenames (no extension).
            label_bands (list of int): List of band indices (1-based) from the 8-band label .tif to use as target(s).
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.sample_file_list = sample_file_list
        self.label_bands = label_bands
        self.num_bands = 14
        self.num_timesteps = 37
        self.image_band_count = self.num_bands * self.num_timesteps  # = 518

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, idx):
        sample_name = self.sample_file_list[idx]
        image_path = os.path.join(self.image_dir, f"{sample_name}.tif")
        label_path = os.path.join(self.label_dir, f"{sample_name}.tif")

        # Load Sentinel-2 image
        with rasterio.open(image_path) as src:
            full_array = src.read()  # shape: (518+, H, W)
            image_tensor = torch.from_numpy(full_array[:self.image_band_count]).float()
            H, W = image_tensor.shape[1:]
            image_tensor = image_tensor.reshape(self.num_timesteps, self.num_bands, H, W).permute(1, 0, 2, 3)  # (14, 37, H, W)

        # Load mask (single band or stacked)
        with rasterio.open(label_path) as label_src:
            mask_array = label_src.read(self.label_bands)  # shape: (B, H, W)
            if mask_array.shape[0] == 1:
                mask_tensor = torch.from_numpy(mask_array[0]).long()  # (H, W)
            else:
                mask_tensor = torch.from_numpy(mask_array).long()      # (B, H, W)

        return {
            "image": image_tensor,
            "mask": mask_tensor
        }

    @staticmethod 
    def plot_mask_tensor(mask_tensor):
        band_titles = [
            "Band 1: Irrigation Type",
            "Band 2: Irrigation Presence",
            "Band 3: Unclear signs of agriculture",
            "Band 4: Only slightly green",
            "Band 5: Uneven",
            "Band 6: May naturally be green",
            "Band 7: May be a fishpond",
            "Band 8: Certainty Score"
        ]

        # Color/label settings
        band1_colors = ['#e0e0e0', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#d62728']
        band1_labels = [
            '0: No irrigation', '1: Small-scale', '2: Tree crop',
            '3: Industrial', '4: Lawn', '5: Covered'
        ]
        band8_colors = ['#e0e0e0', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e', '#d62728']
        band8_labels = [
            '0: No irrigation', '1: Probably not irrigated', '2: Probably not irrigated',
            '3: May be irrigated', '4: Probably irrigated', '5: Irrigated'
        ]
        binary_colors = ['#e0e0e0', '#1f77b4']
        binary_labels = ['0: No', '1: Yes']

        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        axes = axes.flatten()

        for i in range(8):
            ax = axes[i]
            band = mask_tensor[i].numpy()
            if i == 0:
                im = ax.imshow(band, cmap=ListedColormap(band1_colors), vmin=0, vmax=5)
                legend_handles = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in zip(band1_colors, band1_labels)]
            elif i == 7:
                im = ax.imshow(band, cmap=ListedColormap(band8_colors), vmin=0, vmax=5)
                legend_handles = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in zip(band8_colors, band8_labels)]
            else:
                im = ax.imshow(band, cmap=ListedColormap(binary_colors), vmin=0, vmax=1)
                legend_handles = [Patch(facecolor=c, edgecolor='k', label=l) for c, l in zip(binary_colors, binary_labels)]

            ax.set_title(band_titles[i], fontsize=14)
            ax.axis('off')

            # Place the legend to the right of each subplot
            ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
                    borderaxespad=0., fontsize=10, frameon=False)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.4)
        plt.show()

    @staticmethod
    def plot_all_bands_at_time( image_tensor, time_idx=0, band_names=None, band_cmaps=None):
        """
        Plot all 14 bands for a specific time index, each with a distinct colormap.

        Args:
            image_tensor: Tensor (14, 37, H, W)
            time_idx: Index of the timepoint to plot (0-based)
            band_names: List of 14 band names for titles
            band_cmaps: List of 14 colormap names (e.g., ['Blues', 'Greens', ...])
        """
        n_bands = image_tensor.shape[0]
        n_cols = 4
        n_rows = 4
        default_cmaps = [
            'Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'Greys', 'cividis',
            'YlGn', 'YlOrBr', 'PuRd', 'viridis', 'plasma', 'magma', 'cubehelix'
        ]
        if band_cmaps is None:
            band_cmaps = default_cmaps
        if band_names is None:
            band_names = [f"Band {i+1}" for i in range(n_bands)]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 14))
        axes = axes.flatten()
        for b in range(n_bands):
            band_img = image_tensor[b, time_idx].numpy()
            band_img = np.where(band_img == -9999, np.nan, band_img)
            ax = axes[b]
            im = ax.imshow(band_img, cmap=band_cmaps[b])
            ax.set_title(f"{band_names[b]} (t={time_idx})")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Hide unused axes
        for ax in axes[n_bands:]:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    # Usage example:
    band_names = [
        "B2", "B3", "B4", "B5", "B6", "B7", "B8",
        "B8A", "B11", "B12", "NDVI", "EVI", "NDWI", "SCL"
    ]    

    @staticmethod
    def plot_band_over_time(image_tensor, band_idx=0, band_name=None, time_indices=None):
        """
        Plot one band (e.g. NDVI) for all 37 timepoints.

        Args:
            image_tensor: Tensor of shape (14, 37, H, W)
            band_idx: Which band to plot (0–13)
            band_name: Optional string for title
            time_indices: Optional list of timepoints to plot (default: all 37)
        """
        n_time = image_tensor.shape[1]
        if time_indices is None:
            time_indices = range(n_time)
        n_cols = 7
        n_rows = int(np.ceil(len(time_indices) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        axes = axes.flatten()
        band_data = image_tensor[band_idx]  # shape: (37, H, W)
        for i, t in enumerate(time_indices):
            img = band_data[t].numpy()
            img = np.where(img == -9999, np.nan, img)
            ax = axes[i]
            im = ax.imshow(img, cmap='viridis')
            ax.set_title(f"Time {t}")
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # Hide unused axes
        for ax in axes[len(time_indices):]:
            ax.axis('off')
        plt.suptitle(f"{band_name or f'Band {band_idx+1}'} Over Time", fontsize=16)
        plt.tight_layout()
        plt.show()     


#---testing code---
# --- 1. Setup paths and sample list ---
image_dir = "multi-temporal-crop-classification-subset"  
label_dir = "multi-temporal-crop-classification-subset"
sample_file_list = ["image_test"]  # No .tif extension

# --- 2. Instantiate dataset ---
dataset = MultiTemporalCropDataset(
    image_dir=image_dir,
    label_dir=label_dir,
    sample_file_list=sample_file_list,
    label_bands=list(range(1, 9)),  # or [1], [2], etc.
)

# --- 3. Fetch one sample and print shapes ---
sample = dataset[0]
image, mask = sample["image"], sample["mask"]

print("Image tensor shape:", image.shape)  # Should be (14, 37, H, W)
print("Mask tensor shape:", mask.shape)    # (8, H, W) if label_bands=[1,2,3,4,5,6,7,8], else (H, W)

print("Image stats: min =", image.min().item(), "max =", image.max().item())
print("Mask unique values:", torch.unique(mask))

# --- Visualize all 8 mask bands ---
#MultiTemporalCropDataset.plot_mask_tensor(mask)   # Will plot all bands by default
MultiTemporalCropDataset.plot_all_bands_at_time(image)
