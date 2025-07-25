import rasterio
import numpy as np
import matplotlib.pyplot as plt
import json

# Set path
TIF_PATH = 'data/features/site_-15.04_26.69_2023_1.tif'
JSON_PATH = TIF_PATH.replace('.tif', '.json')

# Load metadata
with open(JSON_PATH) as f:
    meta = json.load(f)

bands = meta['bands']         # e.g. ['B2', 'B3', ..., 'NDWI']
T, B, H, W = meta['shape']    # e.g. [37, 13, 100, 100]

print(f"Metadata: T={T}, B={B}, H={H}, W={W}")
print(f"Bands ({len(bands)}): {bands}")

# Load and reshape image
with rasterio.open(TIF_PATH) as src:
    raw = src.read()  # shape: (T*B, H, W)
print(f"Raw tif shape: {raw.shape}")

# Ensure raw has correct size
assert raw.shape[0] == T * B, "Band count mismatch between tif and metadata"
assert raw.shape[1] == H and raw.shape[2] == W, "Spatial dimension mismatch"

# Reshape to (T, B, H, W)
stack = raw.reshape(T, B, H, W)

# Find RGB band indices
try:
    bidx_r = bands.index('B4')
    bidx_g = bands.index('B3')
    bidx_b = bands.index('B2')
except ValueError:
    raise RuntimeError("Bands B4, B3, B2 not found in metadata")

# Robust mean: exclude invalid/masked pixels (like -9999)
def masked_mean(arr):
    """
    Compute mean across the time dimension, excluding invalid/masked values (e.g. -9999, <0).
    Args:
        arr (np.ndarray): shape (T, H, W)
    Returns:
        mean_img (np.ndarray): shape (H, W)
    """
    arr = np.where(arr < 0, np.nan, arr)  # All negative values (e.g. -9999) as nan
    return np.nanmean(arr, axis=0)

rgb_mean = np.stack([
    masked_mean(stack[:, bidx_r, :, :]),
    masked_mean(stack[:, bidx_g, :, :]),
    masked_mean(stack[:, bidx_b, :, :])
], axis=-1)

# Normalize for visualization
def normalize_img(arr):
    arr = arr - np.nanpercentile(arr, 1)
    arr = arr / (np.nanpercentile(arr, 99) + 1e-5)
    arr = np.clip(arr, 0, 1)
    return arr

rgb_mean_norm = normalize_img(rgb_mean)

plt.figure(figsize=(7, 7))
plt.imshow(rgb_mean_norm)
plt.title('Mean RGB Composite of All Time Steps (masked)')
plt.axis('off')
plt.tight_layout()
plt.show()

# (Optional) Visualize a single time step's RGB composite to check if one frame is OK
# frame_idx = 0
# rgb_single = np.stack([
#     stack[frame_idx, bidx_r, :, :],
#     stack[frame_idx, bidx_g, :, :],
#     stack[frame_idx, bidx_b, :, :]
# ], axis=-1)
# plt.figure(figsize=(7,7))
# plt.imshow(normalize_img(rgb_single))
# plt.title(f'Time {frame_idx} RGB')
# plt.axis('off')
# plt.tight_layout()
# plt.show()
