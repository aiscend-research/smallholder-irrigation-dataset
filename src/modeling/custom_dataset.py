import os
import torch
import glob
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio

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
    def __init__(self, image_dir, label_dir, label_bands=list(range(1, 9)), image_band_names=None, time_step_selection=None):
        """
        Args:
            image_dir (str): Path to directory containing Sentinel-2 input .tif files.
            label_dir (str): Path to directory containing label .tif files.
            label_bands (list of int): List of band indices (1-based) from the label .tif to use as target(s).
            image_band_names (list of str or int, optional): List of band names or indices to select from the image tensor.
                If None, all bands are returned. Band names can be short codes like 'B2' or full names like 'B2 (Blue)'.
            time_step_selection (list, optional): If set, a list where each element is either
                - int: selects that time step (0-based)
                - list of ints: averages those time steps.
                Example: [0, [1,2,3], 4] => output will have three time slices per band:
                    1st: time 0; 2nd: average of times 1,2,3; 3rd: time 4

        This dataset finds all .tif files in image_dir and label_dir, extracts unique IDs from filenames
        (for images: last underscore-separated field before '.tif'; for masks: first underscore-separated field),
        and only keeps pairs that have both an image and a mask for the same unique ID.
        It stores lists of paired image filenames, mask filenames, and integer unique IDs.

        When fetching a sample, returns a dict with:
            "image": Tensor of shape (bands, timesteps, H, W)
            "mask": Tensor of shape (B, H, W) or (H, W)
            "unique_id": Tensor of shape (H, W) filled with the integer unique ID of the sample
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_bands = label_bands
        self.num_bands = 14
        self.num_timesteps = 37
        self.image_band_count = self.num_bands * self.num_timesteps  # = default: 518
        self.image_band_names = image_band_names
        self.time_step_selection = time_step_selection

        # Find all image and label .tif files
        image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.tif")))
        label_files = sorted(glob.glob(os.path.join(self.label_dir, "*.tif")))

        # Extract unique IDs for images: last underscore-separated field before .tif
        image_id_to_file = {}
        for f in image_files:
            base = os.path.splitext(os.path.basename(f))[0]
            parts = base.split('_')
            if len(parts) < 2:
                continue
            unique_id = parts[-1]
            image_id_to_file[unique_id] = f

        # Extract unique IDs for masks: first underscore-separated field
        mask_id_to_file = {}
        for f in label_files:
            base = os.path.splitext(os.path.basename(f))[0]
            parts = base.split('_')
            if len(parts) < 1:
                continue
            unique_id = parts[0]
            mask_id_to_file[unique_id] = f

        # Find intersection of IDs that have both image and mask (matching by unique_id)
        paired_ids = sorted(set(image_id_to_file.keys()) & set(mask_id_to_file.keys()))
        # Store lists of paired image filenames, mask filenames, and integer unique IDs
        self.paired_image_files = []
        self.paired_mask_files = []
        self.paired_unique_ids = []
        for i, uid in enumerate(paired_ids):
            self.paired_image_files.append(image_id_to_file[uid])
            self.paired_mask_files.append(mask_id_to_file[uid])
            self.paired_unique_ids.append(int(uid) if uid.isdigit() else i)

    def __len__(self):
        return len(self.paired_image_files)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample.
        Returns:
            dict with:
                "image": Tensor of shape (bands, timesteps, H, W)
                "mask": Tensor of shape (B, H, W) or (H, W)
                "unique_id": Tensor of shape (H, W) filled with the integer unique ID
        """
        image_path = self.paired_image_files[idx]
        label_path = self.paired_mask_files[idx]
        unique_id = self.paired_unique_ids[idx]

        # Load Sentinel-2 image
        with rasterio.open(image_path) as src:
            full_array = src.read()  # shape: (518+, H, W)
            if full_array.shape[0] != self.image_band_count:
                raise ValueError(f"Image file {image_path} has {full_array.shape[0]} bands, expected {self.image_band_count}.")
            image_tensor = torch.from_numpy(full_array).float()
            H, W = image_tensor.shape[1:]
            image_tensor = image_tensor.reshape(self.num_timesteps, self.num_bands, H, W).permute(1, 0, 2, 3)  # (14, 37, H, W)
            # Convert -9999 values to NaN in image tensor
            image_tensor[image_tensor == -9999] = float('nan')

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
            image_tensor = torch.cat(selected, dim=1)  # (C, S, H, W) where S=len(selection)

        if self.image_band_names is not None:
            band_indices = get_band_indices(self.image_band_names)
            image_tensor = image_tensor[band_indices, ...]

        # Load mask (single band or stacked)
        with rasterio.open(label_path) as label_src:
            mask_array = label_src.read(self.label_bands)  # shape: (B, H, W)
            if mask_array.shape[0] == 1:
                mask_tensor = torch.from_numpy(mask_array[0]).float()  # convert to float for NaN
                # Convert -9999 values to NaN in mask tensor
                mask_tensor[mask_tensor == -9999] = float('nan')
            else:
                mask_tensor = torch.from_numpy(mask_array).float()      # convert to float for NaN
                # Convert -9999 values to NaN in mask tensor
                mask_tensor[mask_tensor == -9999] = float('nan')

        unique_id_tensor = torch.full(mask_tensor.shape[-2:], fill_value=unique_id, dtype=torch.long)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "unique_id": unique_id_tensor
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



# #---testing code---
# #--- 1. Setup paths and sample list ---
# image_dir = "multi-temporal-crop-classification-subset/test_unique_id"  
# label_dir = "multi-temporal-crop-classification-subset/test_unique_id"

# # --- 2. Instantiate dataset ---
# dataset = MultiTemporalCropDataset(
#     image_dir=image_dir,
#     label_dir=label_dir,
#     label_bands=list(range(1, 9)),  # or [1], [2], etc.
# )

# #--- 3. Fetch one sample and print shapes ---
# sample = dataset[0]
# image, mask, unique_id = sample["image"], sample["mask"], sample["unique_id"]

# print("Image tensor shape:", image.shape)  # Should be (14, 37, H, W)
# print("Mask tensor shape:", mask.shape)    # (8, H, W) if label_bands=[1,2,3,4,5,6,7,8], else (H, W)
# print("Unique_id Tensor:",  unique_id.shape)
# print(unique_id)

# print("Image stats: min =", image.min().item(), "max =", image.max().item())
# print("Mask unique values:", torch.unique(mask))



# # --- Visualize all 8 mask bands ---
# MultiTemporalCropDataset.plot_mask_tensor(mask)   # Will plot all bands by default
# MultiTemporalCropDataset.plot_all_bands_at_time(image)
