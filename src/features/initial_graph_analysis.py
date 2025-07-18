import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Set your file path list
file_list = [
    'data/features/site_-15.04_26.69_2023_Gabriel_1_stack.npy'# add more files if needed
]

def normalize_img(arr):
    arr = arr - np.percentile(arr, 1)
    arr = arr / (np.percentile(arr, 99) + 1e-5)
    return np.clip(arr, 0, 1)

def get_band_idx(bands, name):
    try:
        return bands.index(name)
    except ValueError:
        raise RuntimeError(f'Band {name} not found!')

def get_ndvi(stack, bidx_nir, bidx_red):
    return (stack[:, bidx_nir] - stack[:, bidx_red]) / (stack[:, bidx_nir] + stack[:, bidx_red] + 1e-5)

# Main processing loop
for npy_path in file_list:
    print(f"Processing {npy_path}")
    stack = np.load(npy_path)
    with open(npy_path.replace('.npy', '.json')) as f:
        meta = json.load(f)
    bands = meta['bands']
    n_time, n_bands, H, W = stack.shape

    # Find band indices
    bidx_r = get_band_idx(bands, 'B4')
    bidx_g = get_band_idx(bands, 'B3')
    bidx_b = get_band_idx(bands, 'B2')
    bidx_nir = get_band_idx(bands, 'B8')

    # Read the date range of each time window
    windows = meta['windows']
    # Extract the starting month for each window
    months = [int(w['date_range'][0][5:7]) for w in windows]

    # Rainy season: Nov-Dec & Jan-May
    rain_idxs = [i for i, m in enumerate(months) if m in [11, 12, 1, 2, 3, 4, 5]]
    # Dry season: Jun-Oct
    dry_idxs = [i for i, m in enumerate(months) if m in [6, 7, 8, 9, 10]]

    # Seasonal RGB visualization
    def mean_rgb(idx_list):
        rgb = np.stack([
            stack[idx_list, bidx_r].mean(axis=0),
            stack[idx_list, bidx_g].mean(axis=0),
            stack[idx_list, bidx_b].mean(axis=0)
        ], axis=-1)
        return normalize_img(rgb)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(mean_rgb(rain_idxs))
    plt.title('Rainy Season Mean RGB')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(mean_rgb(dry_idxs))
    plt.title('Dry Season Mean RGB')
    plt.axis('off')
    plt.suptitle(f'Site: {os.path.basename(npy_path)} - RGB')
    plt.tight_layout()
    plt.show()

    # Seasonal NDVI visualization
    ndvi = get_ndvi(stack, bidx_nir, bidx_r)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(ndvi[rain_idxs].mean(axis=0), cmap='YlGn')
    plt.title('Rainy Season Mean NDVI')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(ndvi[dry_idxs].mean(axis=0), cmap='YlGn')
    plt.title('Dry Season Mean NDVI')
    plt.colorbar()
    plt.suptitle(f'Site: {os.path.basename(npy_path)} - NDVI')
    plt.tight_layout()
    plt.show()

    # Year-round NDVI mean and variance
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(ndvi.mean(axis=0), cmap='YlGn')
    plt.title('NDVI Mean (All Year)')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(ndvi.var(axis=0), cmap='YlOrRd')
    plt.title('NDVI Variance (All Year)')
    plt.colorbar()
    plt.suptitle(f'Site: {os.path.basename(npy_path)} - NDVI Year-round Statistics')
    plt.tight_layout()
    plt.show()

# Batch comparison of NDVI mean for all sites
plt.figure(figsize=(4*len(file_list), 4))
for i, npy_path in enumerate(file_list):
    stack = np.load(npy_path)
    with open(npy_path.replace('.npy', '.json')) as f:
        bands = json.load(f)['bands']
    bidx_nir = get_band_idx(bands, 'B8')
    bidx_r = get_band_idx(bands, 'B4')
    ndvi = (stack[:, bidx_nir] - stack[:, bidx_r]) / (stack[:, bidx_nir] + stack[:, bidx_r] + 1e-5)
    plt.subplot(1, len(file_list), i+1)
    plt.imshow(ndvi.mean(axis=0), cmap='YlGn')
    plt.title(f'Site {i+1} NDVI Mean')
    plt.axis('off')
plt.tight_layout()
plt.show()
