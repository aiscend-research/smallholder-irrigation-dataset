import rasterio
import numpy as np
import matplotlib.pyplot as plt
import json

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

# Band info
print(f"Bands in metadata: {bands}")
print("Band indices: R(B4)={}, G(B3)={}, B(B2)={}".format(
    bands.index('B4'), bands.index('B3'), bands.index('B2')
))

# Print raw value stats for each band (across all time steps)
for bname in ['B4', 'B3', 'B2']:
    bidx = bands.index(bname)
    arr = stack[:, bidx, :, :]
    print(f"\nStats for band {bname}:")
    print("  Min:", np.nanmin(arr))
    print("  Max:", np.nanmax(arr))
    print("  % Masked (<=0):", np.mean(arr <= 0) * 100)

# Plot all-time mask for each RGB band
def plot_mask(band_idx, name):
    mask = np.all(stack[:, band_idx, :, :] <= 0, axis=0)
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title(f'All-Time Mask for {name} (white = all-masked)')
    plt.axis('off')

for bname in ['B4', 'B3', 'B2']:
    plot_mask(bands.index(bname), bname)
plt.show()

# Find and plot time step(s) with the most valid pixels
rgb_indices = [bands.index('B4'), bands.index('B3'), bands.index('B2')]
valid_counts = []

for t in range(T):
    rgb_frame = np.stack([stack[t, i, :, :] for i in rgb_indices], axis=-1)
    valid = np.all(rgb_frame > 0, axis=-1)
    valid_counts.append(np.sum(valid))

# Find the time(s) with max valid pixels
max_valid = max(valid_counts)
best_indices = [i for i, c in enumerate(valid_counts) if c == max_valid]
print(f"\nBest time steps (most valid pixels = {max_valid}): {best_indices}")

def normalize_img(arr):
    arr = arr - np.nanpercentile(arr, 1)
    arr = arr / (np.nanpercentile(arr, 99) + 1e-5)
    arr = np.clip(arr, 0, 1)
    return arr

# Plot the best frames (with most valid pixels)
for t in best_indices:
    rgb_frame = np.stack([stack[t, i, :, :] for i in rgb_indices], axis=-1)
    valid = np.all(rgb_frame > 0, axis=-1)
    rgb_norm = np.zeros_like(rgb_frame, dtype=float)
    for c in range(3):
        band = rgb_frame[..., c]
        if np.any(valid):
            band = (band - np.percentile(band[valid], 1)) / (np.percentile(band[valid], 99) + 1e-5)
        else:
            band = np.zeros_like(band)
        rgb_norm[..., c] = np.clip(band, 0, 1)
    rgb_norm[~valid] = 1  # Show masked as white

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_norm)
    plt.title(f"Best Frame (Time {t}) - Most Valid Pixels ({np.sum(valid)})")
    plt.axis('off')
plt.show()

# Plot a few default time step RGBs for reference
for t in [0, T//2, T-1]:
    rgb_frame = np.stack([stack[t, i, :, :] for i in rgb_indices], axis=-1)
    valid = np.all(rgb_frame > 0, axis=-1)
    rgb_norm = np.zeros_like(rgb_frame, dtype=float)
    for c in range(3):
        band = rgb_frame[..., c]
        if np.any(valid):
            band = (band - np.percentile(band[valid], 1)) / (np.percentile(band[valid], 99) + 1e-5)
        else:
            band = np.zeros_like(band)
        rgb_norm[..., c] = np.clip(band, 0, 1)
    rgb_norm[~valid] = 1  # Show masked as white

    plt.figure(figsize=(5, 5))
    plt.imshow(rgb_norm)
    plt.title(f"Time {t} RGB (masked)")
    plt.axis('off')
plt.show()

# Compute mean composite, ignoring invalid (<0) values
def masked_mean(arr):
    """
    Compute mean across the time dimension, excluding invalid/masked values (e.g. -9999, <0).
    Args:
        arr (np.ndarray): shape (T, H, W)
    Returns:
        mean_img (np.ndarray): shape (H, W)
    """
    arr = np.where(arr < 0, np.nan, arr)
    return np.nanmean(arr, axis=0)

bidx_r = bands.index('B4')
bidx_g = bands.index('B3')
bidx_b = bands.index('B2')

rgb_mean = np.stack([
    masked_mean(stack[:, bidx_r, :, :]),
    masked_mean(stack[:, bidx_g, :, :]),
    masked_mean(stack[:, bidx_b, :, :])
], axis=-1)

print("\nStats for mean composite (before normalization):")
for i, name in enumerate(['R','G','B']):
    print(f"{name}: min={np.nanmin(rgb_mean[...,i])}, max={np.nanmax(rgb_mean[...,i])}, mean={np.nanmean(rgb_mean[...,i])}")

rgb_mean_norm = normalize_img(rgb_mean)

plt.figure(figsize=(7, 7))
plt.imshow(rgb_mean_norm)
plt.title('Mean RGB Composite of All Time Steps (masked)')
plt.axis('off')
plt.tight_layout()
plt.show()