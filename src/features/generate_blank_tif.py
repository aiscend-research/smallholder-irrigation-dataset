import os
import numpy as np
import rasterio

# Constants matching your main script
WIDTH, HEIGHT = 1098, 1098  # 1km x 1km at 10m resolution
BANDS = 11                   
BLANK_PATH = 'data/features/blank.tif' 

# Ensure output directory exists
os.makedirs(os.path.dirname(BLANK_PATH), exist_ok=True)

# Create an all-zero image
blank = np.zeros((BANDS, HEIGHT, WIDTH), dtype=np.uint16)

profile = {
    'driver': 'GTiff',
    'height': HEIGHT,
    'width': WIDTH,
    'count': BANDS,
    'dtype': 'uint16',
    'crs': "EPSG:32633",
    'transform': rasterio.transform.from_origin(0, 0, 10, 10),
}

with rasterio.open(BLANK_PATH, 'w', **profile) as dst:
    dst.write(blank)

print(f"Blank tif saved to {BLANK_PATH}")
