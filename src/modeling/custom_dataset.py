import os
import torch
from torch.utils.data import Dataset
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
            mask = src.read(self.mask_band)      # shape: (H, W)

        # Optional: create a dummy image of zeros (same shape as mask), for compatibility
        dummy_image = torch.zeros_like(torch.from_numpy(mask)).float().unsqueeze(0)  # shape (1, H, W)
        mask = torch.from_numpy(mask).long()
        return dummy_image, mask
    
    def _build_metadata(self):
        pass

# --- TESTING CODE ---
# data_dir = "experiments"
# sample_file_list = ["test4"]  # Assumes 'experiments/test4.tif' exists

# dataset = IrrigationMaskDataset(data_dir, sample_file_list)
# image, mask = dataset[0]

# import matplotlib.pyplot as plt

# plt.subplot(1,2,1)
# plt.imshow(image[0], cmap='gray')  # dummy image (all zeros)
# plt.title('Dummy Input (no satellite image)')
# plt.axis('off')

# plt.subplot(1,2,2)
# plt.imshow(mask, cmap='Greens', vmin=0, vmax=1)
# plt.title('Irrigation Mask (Band 2)')
# plt.axis('off')
# plt.show()