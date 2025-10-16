import os
import json
import re
import torch
import glob
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

# Short band names
SHORT_BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI", "EVI", "NDWI", "SCL"]

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
    def __init__(self, image_dir=None, label_dir=None, data_dir=None, label_bands=None, image_band_names=None, time_step_selection=None, drop_cloud_images=True):
        """
        Args:
            image_dir (str): Path to directory containing Sentinel-2 input .tif files.
            label_dir (str): Path to directory containing label .tif files.
            data_dir (str, optional): If provided, overrides both image_dir and label_dir.
            label_bands (list of int): List of band indices (1-based) from the label .tif to use as target(s).
            image_band_names (list of str or int, optional): List of band names or indices to select from the image tensor.
                If None, all bands are returned. Band names can be short codes like 'B2' or full names like 'B2 (Blue)'.
            time_step_selection (list, optional): If set, a list where each element is either
                - int: selects that time step (0-based)
                - list of ints: averages those time steps.
                Example: [0, [1,2,3], 4] => output will have three time slices per band:
                    1st: time 0; 2nd: average of times 1,2,3; 3rd: time 4
            drop_cloud_images (bool, optional): True by default. If true, drops images with any invalid (cloudy) pixels. If false,
            keeps images with any valid pixels, while dropping only invalid pixels

        Note:
            If data_dir is provided, it overrides both image_dir and label_dir.

        Naming expected:
            {unique_id}_image.tif
            {unique_id}_label.tif
        """
        # Override image_dir and label_dir if data_dir is provided
        if data_dir is not None:
            image_dir = data_dir
            label_dir = data_dir

        self.image_dir = image_dir
        self.label_dir = label_dir
        if not image_band_names:
            self.image_band_names = S2_BAND_NAMES
        else:
            self.image_band_names = image_band_names
        if not label_bands:
            self.label_bands = list(range(1, 9))
        else:
            self.label_bands = label_bands
        self.num_bands = 14
        self.num_timesteps = 37
        self.image_band_count = self.num_bands * self.num_timesteps
        self.time_step_selection = time_step_selection
        self.drop_cloud_images = drop_cloud_images

        # === Integrated block (strict *_image.tif + *_label.tif) ===
        # Find files according to the naming convention: *_image.tif and *_label.tif
        image_files = sorted(glob.glob(os.path.join(self.image_dir, "*_image.tif")))
        label_files = sorted(glob.glob(os.path.join(self.label_dir, "*_label.tif")))  # only label.tif

        # Extract unique IDs for images: handle new standardized naming {base}_image.tif
        image_id_to_file = {}
        for f in image_files:
            base = os.path.splitext(os.path.basename(f))[0]
            if base.endswith('_image'):
                unique_id = base[:-6]  # strip '_image'
            else:
                unique_id = base
            image_id_to_file[unique_id] = f

        # Extract unique IDs for labels: only *_label.tif supported
        mask_id_to_file = {}
        for f in label_files:
            base = os.path.splitext(os.path.basename(f))[0]
            if base.endswith('_label'):
                unique_id = base[:-6]  # strip '_label'
            else:
                unique_id = base
            mask_id_to_file[unique_id] = f

        # Intersect IDs so we only keep samples that have both image and label
        paired_ids = sorted(set(image_id_to_file.keys()) & set(mask_id_to_file.keys()))

        logger.info(f"Found {len(image_files)} image files, {len(label_files)} label files")
        logger.info(f"Extracted {len(image_id_to_file)} image IDs, {len(mask_id_to_file)} label IDs")
        logger.info(f"Matched {len(paired_ids)} paired IDs")

        self.paired_image_files = []
        self.paired_mask_files = []
        self.paired_unique_ids = []
        for i, uid in enumerate(paired_ids):
            self.paired_image_files.append(image_id_to_file[uid])
            self.paired_mask_files.append(mask_id_to_file[uid])
            self.paired_unique_ids.append(int(uid) if uid.isdigit() else i)
        # === End integrated block ===

    def __len__(self):
        return len(self.paired_image_files)

    def __getitem__(self, idx):
        image_path = self.paired_image_files[idx]
        mask_path = self.paired_mask_files[idx]
        unique_id = self.paired_unique_ids[idx]

        # Load metadata (assume .json with same basename as image)
        sample_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(self.image_dir, f"{sample_name}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Load Sentinel-2 time series stack
        with rasterio.open(image_path) as src:
            full_array = src.read()  # shape: (518, H, W)
            image_tensor = torch.from_numpy(full_array).float()
            H, W = image_tensor.shape[1:]
            image_tensor = image_tensor.reshape(self.num_timesteps, self.num_bands, H, W).permute(1, 0, 2, 3)  # (14, 37, H, W)

            # Replace -9999 (no-data)
            if torch.any(image_tensor == -9999):
                invalid_count = torch.sum(image_tensor == -9999)
                total_pixels = image_tensor.numel()
                invalid_percentage = (invalid_count / total_pixels) * 100
                print(f"Sample {idx}: Found {invalid_count} -9999 values ({invalid_percentage:.2f}% of pixels)")
                if self.drop_cloud_images and torch.any(image_tensor == -9999):
                    image_tensor = torch.zeros(image_tensor.shape)
                else:
                    image_tensor = torch.where(image_tensor == -9999, torch.tensor(0.0), image_tensor)

        # --- Time step selection/averaging ---
        if self.time_step_selection is not None:
            selected = []
            for sel in self.time_step_selection:
                if isinstance(sel, int):
                    selected.append(image_tensor[:, sel:sel+1, :, :])  # (C, 1, H, W)
                elif isinstance(sel, list):
                    selected.append(image_tensor[:, sel, :, :].mean(dim=1, keepdim=True))  # (C, 1, H, W)
                else:
                    raise ValueError(f"time_step_selection element must be int or list, got {type(sel)}")
            image_tensor = torch.cat(selected, dim=1)  # (C, S, H, W)

        if self.image_band_names is not None:
            band_indices = get_band_indices(self.image_band_names)
            image_tensor = image_tensor[band_indices, ...]

        # Load mask/label
        with rasterio.open(mask_path) as label_src:
            mask_array = label_src.read(self.label_bands)  # shape: (B, H, W)
            mask_tensor = torch.from_numpy(mask_array).float()

            if torch.any(mask_tensor == -9999):
                invalid_count = torch.sum(mask_tensor == -9999)
                total_pixels = mask_tensor.numel()
                invalid_percentage = (invalid_count / total_pixels) * 100
                print(f"Sample {idx}: Mask has {invalid_count} -9999 values ({invalid_percentage:.2f}% of pixels)")
                mask_tensor = torch.where(mask_tensor == -9999, torch.tensor(0.0), mask_tensor)

            if mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor[0]

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "metadata": metadata,
            "id": unique_id
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
            band_cmaps: List of 14 colormap names
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
        for ax in axes[len(time_indices):]:
            ax.axis('off')
        plt.suptitle(f"{band_name or band_names[band_idx]} Over Time", fontsize=16)
        plt.tight_layout()
        plt.show()