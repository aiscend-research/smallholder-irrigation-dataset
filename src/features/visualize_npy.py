import numpy as np
import matplotlib.pyplot as plt
import json

# Set file path here
NPY_PATH = 'data/features/site_-15.04_26.69_2023_1_stack.npy' 
JSON_PATH = NPY_PATH.replace('.npy', '.json')

# Load the stack and metadata
stack = np.load(NPY_PATH)
with open(JSON_PATH) as f:
    meta = json.load(f)

print('Stack shape:', stack.shape)   # (n_time, bands, H, W)
print('Bands:', meta['bands'])

bands = meta['bands']
n_time, n_bands, H, W = stack.shape

# Find band indices for RGB
try:
    bidx_r = bands.index('B4')
    bidx_g = bands.index('B3')
    bidx_b = bands.index('B2')
except ValueError:
    raise RuntimeError("Bands B4, B3, B2 not found in metadata")

# Compute mean RGB composite
rgb_mean = np.stack([
    stack[:, bidx_r, :, :].mean(axis=0),
    stack[:, bidx_g, :, :].mean(axis=0),
    stack[:, bidx_b, :, :].mean(axis=0)
], axis=-1)

# Normalize image for display
def normalize_img(arr):
    arr = arr - np.percentile(arr, 1)
    arr = arr / (np.percentile(arr, 99) + 1e-5)
    arr = np.clip(arr, 0, 1)
    return arr

rgb_mean_norm = normalize_img(rgb_mean)

# Plot the mean RGB composite
plt.figure(figsize=(7, 7))
plt.imshow(rgb_mean_norm)
plt.title('Mean RGB Composite of All Time Steps')
plt.axis('off')
plt.show()
