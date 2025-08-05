import os
import json
import re
import torch
import numpy as np
import logging
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio

# Configure logging
logger = logging.getLogger(__name__)

# Sentinel-2 and derived band names for 14-band cubes
S2_BAND_NAMES = [
    "B2 (Blue)",
    "B3 (Green)",
    "B4 (Red)",
    "B5 (Vegetation Red Edge 1)",
    "B6 (Vegetation Red Edge 2)",
    "B7 (Vegetation Red Edge 3)",
    "B8 (Near Infrared, NIR)",
    "B8A (Narrow NIR)",
    "B11 (Short Wave Infrared 1, SWIR 1)",
    "B12 (Short Wave Infrared 2, SWIR 2)",
    "NDVI (Normalized Difference Vegetation Index)",
    "EVI (Enhanced Vegetation Index)",
    "NDWI (Normalized Difference Water Index)",
    "SCL (Scene Classification Layer)"
]

# Band name to index mapping (both short code and full name)
S2_BAND_NAME_TO_INDEX = {}
for i, name in enumerate(S2_BAND_NAMES):
    short = name.split(" ")[0].split("(")[0].strip()
    S2_BAND_NAME_TO_INDEX[short] = i
    S2_BAND_NAME_TO_INDEX[name] = i

def get_band_indices(band_names):
    """
    Converts a list of band names or indices to index list for slicing tensors.
    """
    indices = []
    for b in band_names:
        if isinstance(b, int):
            indices.append(b)
        elif isinstance(b, str):
            idx = S2_BAND_NAME_TO_INDEX.get(b)
            if idx is None:
                raise ValueError(f"Unknown band name: {b}")
            indices.append(idx)
        else:
            raise ValueError(f"Band identifier must be str or int, got {type(b)}")
    return indices

class MultiTemporalCropDataset(Dataset):
    def __init__(self, data_dir, sample_file_list, label_bands=list(range(1, 9))):
        """
        Args:
            data_dir (str): Path to directory containing Sentinel-2 .tif files and corresponding .json metadata files.
            sample_file_list (list of str): List of sample base filenames (no extension).
            label_bands (list of int): List of band indices (1-based) from the 8-band label .tif to use as target(s).
        """
        self.data_dir = data_dir
        self.sample_file_list = sample_file_list
        self.label_bands = label_bands
        self.num_bands = 14  # 10 Sentinel-2 + NDVI + EVI + NDWI + SCL
        self.num_timesteps = 37
        self.image_band_count = self.num_bands * self.num_timesteps  # = 518
        
        # Set default value for image band names (can be None or specific band list)
        self.image_band_names = None
        
        # Validate files and create list of actually existing samples
        self.valid_files = []
        for sample_name in sample_file_list:
            tif_path = os.path.join(data_dir, f"{sample_name}.tif")
            json_path = os.path.join(data_dir, f"{sample_name}.json")
            if os.path.exists(tif_path) and os.path.exists(json_path):
                self.valid_files.append(sample_name)
            else:
                logger.warning(f"Missing .tif or .json for {sample_name}. Skipping.")
        
        logger.info(f"Found {len(self.valid_files)} valid files out of {len(sample_file_list)} requested")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        sample_name = self.valid_files[idx]
        tif_path = os.path.join(self.data_dir, f"{sample_name}.tif")
        json_path = os.path.join(self.data_dir, f"{sample_name}.json")

        # Load metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Load Sentinel-2 time series stack
        with rasterio.open(tif_path) as src:
            # Read the full stack: shape (T*B, H, W) = (518, 100, 100)
            full_array = src.read()  # shape: (518, H, W)
            
            # Reshape to (T, B, H, W) = (37, 14, 100, 100)
            full_array = full_array.reshape(self.num_timesteps, self.num_bands, full_array.shape[1], full_array.shape[2])
            
            # Convert to torch tensor and permute to (B, T, H, W) = (14, 37, 100, 100)
            image_tensor = torch.from_numpy(full_array).permute(1, 0, 2, 3).float()

        # Load mask/label data
        # Extract unique_id and site_id from sample name
        # Sample name format: "1_5168346_2023.09.06_image"
        parts = sample_name.split('_')
        if len(parts) >= 3:
            unique_id = parts[0]
            site_id_number = parts[1]
        else:
            # Fallback if pattern doesn't match
            site_id_number = "5168346"
        
        # Construct mask filename using the new consistent naming convention
        # From image name: "1_5168346_2023.09.06_image.tif" -> mask: "1_5168346_2023.09.06_label.tif"
        mask_filename = f"{unique_id}_{site_id_number}_2023.09.06_label.tif"
        mask_path = os.path.join(self.data_dir, mask_filename)
        
        # Check if mask file exists
        if os.path.exists(mask_path):
            # Load the actual mask file
            with rasterio.open(mask_path) as label_src:
                mask_array = label_src.read(self.label_bands)
                if mask_array.shape[0] == 1:
                    mask_tensor = torch.from_numpy(mask_array[0]).long()  # (H, W)
                else:
                    mask_tensor = torch.from_numpy(mask_array).long()      # (B, H, W)
            logger.debug(f"Loaded mask from: {mask_filename}")
        else:
            # Fallback to placeholder if mask file doesn't exist
            logger.warning(f"Mask file not found: {mask_path}")
            mask_tensor = torch.zeros((100, 100), dtype=torch.long)  # Placeholder mask

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "metadata": metadata
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
    def plot_all_bands_at_time(image_tensor, time_idx=0, band_names=None, band_cmaps=None):
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
        band_names = S2_BAND_NAMES
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
      

    @staticmethod
    def plot_band_over_time(image_tensor, band_idx=0, band_name=None, time_indices=None, band_names=S2_BAND_NAMES):
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
        plt.suptitle(f"{band_name or band_names[band_idx]} Over Time", fontsize=16)
        plt.tight_layout()
        plt.show()