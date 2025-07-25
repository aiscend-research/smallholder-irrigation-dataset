import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import rasterio

class MultiTemporalBarebonesDataset(Dataset):
    def __init__(self, data_dir, sample_file_list, mask_band=2):
        """
        Args:
            data_dir (str): Path to directory containing .tif files.
            sample_file_list (list of str): List of sample base filenames (without extension).
            mask_band (int): Band to use as irrigation mask (default 2).
        """
        self.data_dir = data_dir
        self.sample_file_list = sample_file_list
        self.mask_band = mask_band

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, idx):
        sample_name = self.sample_file_list[idx]
        tif_path = os.path.join(self.data_dir, f"{sample_name}.tif")

        with rasterio.open(tif_path) as src:
            full_array = src.read()  # shape: (481+, H, W)
            mask = src.read(self.mask_band)  # shape: (H, W)

        # Construct image tensor: (13, 37, H, W)
        image_tensor = torch.from_numpy(full_array[:481]).float()
        H, W = image_tensor.shape[1:]
        image_tensor = image_tensor.reshape(37, 13, H, W).permute(1, 0, 2, 3)  # (13, 37, H, W)

        mask_tensor = torch.from_numpy(mask).long()  # shape: (H, W)

        return {
            "image": image_tensor,
            "mask": mask_tensor
        }
    
    def plot_band_timeseries(self, idx=0, band_idx=7, num_steps=5):
        """
        Visualize time-series for a specific spectral band at a sample index.

        Args:
            idx (int): Index of the sample to visualize.
            band_idx (int): Spectral band index (0–12) to visualize across time.
            num_steps (int): Number of time steps to plot (from t=0 to t=num_steps-1).
        """
        sample = self[idx]
        image = sample["image"]  # shape: (13, 37, H, W)
        band_timeseries = image[band_idx]  # shape: (37, H, W)

        fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))
        for i in range(num_steps):
            axes[i].imshow(band_timeseries[i], cmap='viridis')
            axes[i].set_title(f"Timestep {i}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
    
    def _build_metadata(self):
        pass