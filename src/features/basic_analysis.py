import numpy as np
import rasterio
import json
import matplotlib.pyplot as plt
import os

# ==== 设置输入文件列表 ====
file_list = [
    'data/features/site_-15.04_26.69_2023_1.tif',
    'data/features/site_-15.04_26.69_2019_2.tif'
]

# ==== 通用函数 ====
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

def get_evi(stack, bidx_nir, bidx_red, bidx_blue):
    return 2.5 * (stack[:, bidx_nir] - stack[:, bidx_red]) / \
           (stack[:, bidx_nir] + 6*stack[:, bidx_red] - 7.5*stack[:, bidx_blue] + 1e-5)

def get_ndwi(stack, bidx_green, bidx_swir):
    return (stack[:, bidx_green] - stack[:, bidx_swir]) / (stack[:, bidx_green] + stack[:, bidx_swir] + 1e-5)

# ==== 主流程 ====
for tif_path in file_list:
    print(f"Processing {tif_path}")
    json_path = tif_path.replace('.tif', '.json')

    with open(json_path) as f:
        meta = json.load(f)

    bands = meta['bands']
    T, B, H, W = meta['shape']

    with rasterio.open(tif_path) as src:
        raw = src.read().reshape(B, T, H, W).transpose(1, 0, 2, 3)  # (T, B, H, W)

    # 找波段索引
    bidx_r = get_band_idx(bands, 'B4')
    bidx_g = get_band_idx(bands, 'B3')
    bidx_b = get_band_idx(bands, 'B2')
    bidx_nir = get_band_idx(bands, 'B8')
    bidx_swir = get_band_idx(bands, 'B11')

    months = [int(w['date_range'][0][5:7]) for w in meta['windows']]
    rain_idxs = [i for i, m in enumerate(months) if m in [11, 12, 1, 2, 3, 4, 5]]
    dry_idxs = [i for i, m in enumerate(months) if m in [6, 7, 8, 9, 10]]

    def plot_seasonal_rgb():
        def mean_rgb(idx_list):
            rgb = np.stack([
                raw[idx_list, bidx_r].mean(axis=0),
                raw[idx_list, bidx_g].mean(axis=0),
                raw[idx_list, bidx_b].mean(axis=0)
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
        plt.suptitle(f'Site: {os.path.basename(tif_path)} - RGB')
        plt.tight_layout()
        plt.show()

    def plot_index(name, index_arr):
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.imshow(index_arr[rain_idxs].mean(axis=0), cmap='YlGn')
        plt.title(f'Rainy Season Mean {name}')
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(index_arr[dry_idxs].mean(axis=0), cmap='YlGn')
        plt.title(f'Dry Season Mean {name}')
        plt.colorbar()
        plt.suptitle(f'Site: {os.path.basename(tif_path)} - {name}')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.imshow(index_arr.mean(axis=0), cmap='YlGn')
        plt.title(f'{name} Mean (All Year)')
        plt.colorbar()
        plt.subplot(1,2,2)
        plt.imshow(index_arr.var(axis=0), cmap='YlOrRd')
        plt.title(f'{name} Variance (All Year)')
        plt.colorbar()
        plt.suptitle(f'Site: {os.path.basename(tif_path)} - {name} Year-round Statistics')
        plt.tight_layout()
        plt.show()

    plot_seasonal_rgb()

    ndvi = get_ndvi(raw, bidx_nir, bidx_r)
    evi = get_evi(raw, bidx_nir, bidx_r, bidx_b)
    ndwi = get_ndwi(raw, bidx_g, bidx_swir)

    plot_index("NDVI", ndvi)
    plot_index("EVI", evi)
    plot_index("NDWI", ndwi)
