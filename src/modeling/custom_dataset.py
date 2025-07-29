import os
import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
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