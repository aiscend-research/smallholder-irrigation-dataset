import rasterio
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# === Set path ===
TIF_PATH = 'data/features/site_-15.04_26.69_2023_1.tif'
JSON_PATH = TIF_PATH.replace('.tif', '.json')

# === Load metadata ===
with open(JSON_PATH) as f:
    meta = json.load(f)

bands = meta['bands']
T, B, H, W = meta['shape']
assert B == len(bands)

# === Load and reshape image ===
with rasterio.open(TIF_PATH) as src:
    raw = src.read()  # shape: (T * B, H, W)

stack = raw.reshape((B, T, H, W)).transpose(1, 0, 2, 3)  # shape: (T, B, H, W)

# === Find RGB band indices ===
try:
    bidx_r = bands.index('B4')
    bidx_g = bands.index('B3')
    bidx_b = bands.index('B2')
except ValueError:
    raise RuntimeError("Bands B4, B3, B2 not found in metadata")

# === Compute mean RGB composite ===
rgb_mean = np.stack([
    stack[:, bidx_r, :, :].mean(axis=0),
    stack[:, bidx_g, :, :].mean(axis=0),
    stack[:, bidx_b, :, :].mean(axis=0)
], axis=-1)

# === Normalize for visualization ===
def normalize_img(arr):
    arr = arr - np.percentile(arr, 1)
    arr = arr / (np.percentile(arr, 99) + 1e-5)
    arr = np.clip(arr, 0, 1)
    return arr

rgb_mean_norm = normalize_img(rgb_mean)

# === Plot ===
plt.figure(figsize=(7, 7))
plt.imshow(rgb_mean_norm)
plt.title('Mean RGB Composite of All Time Steps')
plt.axis('off')
plt.tight_layout()
plt.show()
