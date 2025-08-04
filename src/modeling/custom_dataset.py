import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, BoundaryNorm
import rasterio

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

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, idx):
        sample_name = self.sample_file_list[idx]
        tif_path = os.path.join(self.data_dir, f"{sample_name}.tif")
        json_path = os.path.join(self.data_dir, f"{sample_name}.json")

        # Load metadata
        with open(json_path, 'r') as f:
            metadata = json.load(f)

        # Load Sentinel-2 time series stack
        with rasterio.open(tif_path) as src:
            # Read the full stack: shape (T*B, H, W) = (518, 100, 100)
            full_array = src.read()  # shape: (518, H, W)
            
            # Reshape to (B, T, H, W) = (14, 37, 100, 100)
            # The data is stored as (T*B, H, W), so we need to reshape it
            image_tensor = torch.from_numpy(full_array).float()
            H, W = image_tensor.shape[1:]
            image_tensor = image_tensor.reshape(self.num_timesteps, self.num_bands, H, W).permute(1, 0, 2, 3)  # (14, 37, H, W)

        # Load irrigation labels
        # For now, we'll create a dummy mask since the labels are not in the same format
        # In your actual implementation, you'll need to load the corresponding label files
        # This is a placeholder - you'll need to implement label loading based on your label format
        mask_tensor = torch.zeros((H, W), dtype=torch.long)  # Placeholder mask
        
        # TODO: Implement proper label loading
        # You'll need to load your irrigation labels here
        # The labels should be 8-band .tif files with the following structure:
        # - Band 1: Per-pixel irrigation type classification (0-5)
        # - Band 2: Per-pixel irrigation presence (0-1) 
        # - Bands 3-7: Binary uncertainty explanation masks
        # - Band 8: Irrigation certainty score (0-4)
        #
        # Example implementation:
        # label_path = os.path.join(label_dir, f"{sample_name}_labels.tif")
        # with rasterio.open(label_path) as label_src:
        #     mask_array = label_src.read(self.label_bands)
        #     if mask_array.shape[0] == 1:
        #         mask_tensor = torch.from_numpy(mask_array[0]).long()  # (H, W)
        #     else:
        #         mask_tensor = torch.from_numpy(mask_array).long()      # (B, H, W)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "metadata": metadata
        }

    @staticmethod
    def plot_image_band_timeseries(image_tensor, band_idx=7, num_steps=5):
        """
        Plot a time-series for a specific spectral band, masking -9999 values (clouds).

        Args:
            image_tensor (Tensor): shape (14, 37, H, W)
            band_idx (int): Spectral band index (0–13) out of 14 available
            num_steps (int): Number of time steps to plot
        """
        band_timeseries = image_tensor[band_idx]  # shape: (37, H, W)

        fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))
        for i in range(num_steps):
            img = band_timeseries[i].numpy()
            img = np.where(img == -9999, np.nan, img)  # Mask clouds

            axes[i].imshow(img, cmap='viridis')
            axes[i].set_title(f"Timestep {i}")
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_mask_tensor(mask_tensor, band_indices=None):
        """
        Plot irrigation mask bands.

        Args:
            mask_tensor (Tensor): shape (H, W) or (B, H, W)
            band_indices (list[int], optional): Which bands to plot (for multi-band)
        """
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)

        band_indices = band_indices or list(range(mask_tensor.shape[0]))
        fig, axes = plt.subplots(1, len(band_indices), figsize=(4 * len(band_indices), 4))

        if len(band_indices) == 1:
            axes = [axes]

        for ax, i in zip(axes, band_indices):
            ax.imshow(mask_tensor[i].numpy(), cmap='tab20')
            ax.set_title(f"Mask Band {i}")
            ax.axis('off')

        plt.tight_layout()
        plt.show()