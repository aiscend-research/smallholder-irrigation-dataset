# %% [markdown]
# # Sentinel-2 Image Exporter - Visual Tests
# 
# This notebook tests the `s2_image_exporter` function by downloading sample images
# and visualizing them to verify quality.

# %% Setup
import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import ee

# Add project root to path if needed
# sys.path.append('/path/to/your/project')

from your_module import s2_image_exporter, BANDS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Initialize Earth Engine
ee.Initialize()

print("✓ Setup complete")

# %% Define test cases
test_cases = [
    {
        'name': 'California Agriculture (Summer)',
        'lat': 37.5,
        'lon': -120.5,
        'start': '2023-06-01',
        'end': '2023-08-31',
        'collection': 'L2A'
    },
    {
        'name': 'California Agriculture (Summer) - L1C',
        'lat': 37.5,
        'lon': -120.5,
        'start': '2023-06-01',
        'end': '2023-08-31',
        'collection': 'L1C'
    },
    {
        'name': 'California Agriculture (Winter) - 2017 - L2A',
        'lat': 37.5,
        'lon': -120.5,
        'start': '2017-01-01',
        'end': '2017-03-31',
        'collection': 'L2A'
    },
    {
        'name': 'California Agriculture (Winter) - 2017 - L1C',
        'lat': 37.5,
        'lon': -120.5,
        'start': '2017-01-01',
        'end': '2017-03-31',
        'collection': 'L1C'
    },
]

out_dir = './test_s2_images'
os.makedirs(out_dir, exist_ok=True)

print(f"✓ Defined {len(test_cases)} test cases")
print(f"✓ Output directory: {out_dir}")

# %% Download images
print("\n" + "="*70)
print("DOWNLOADING TEST IMAGES")
print("="*70 + "\n")

results = []

for i, test in enumerate(test_cases):
    print(f"\n[{i+1}/{len(test_cases)}] {test['name']}")
    print(f"  Location: ({test['lat']}, {test['lon']})")
    print(f"  Date range: {test['start']} to {test['end']}")
    print(f"  Collection: {test['collection']}")
    
    file_name = f"test_{i+1:02d}_{test['collection']}.tif"
    file_path = os.path.join(out_dir, file_name)
    
    # Skip if already downloaded
    if os.path.exists(file_path):
        print(f"  ⊙ Already exists, skipping download")
        results.append({'test': test, 'file_path': file_path, 'success': True})
        continue
    
    result = s2_image_exporter(
        lat=test['lat'],
        lon=test['lon'],
        start_date=test['start'],
        end_date=test['end'],
        file_name=file_name,
        out_dir=out_dir,
        collection=test['collection']
    )
    
    success = os.path.exists(file_path)
    results.append({'test': test, 'file_path': file_path, 'success': success})
    
    if success:
        print(f"  ✓ Success: {file_name}")
    else:
        print(f"  ✗ Failed")

print("\n" + "="*70)
print(f"DOWNLOAD COMPLETE: {sum(r['success'] for r in results)}/{len(results)} successful")
print("="*70)

# %% Helper function for visualization
def visualize_s2_image(file_path, test_info, figsize=(10, 10)):
    """Visualize Sentinel-2 image with RGB and stats"""
    
    if not os.path.exists(file_path):
        print(f"✗ File not found: {file_path}")
        return
    
    with rasterio.open(file_path) as src:
        # Read bands
        # BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
        # RGB: B4(Red)=band3, B3(Green)=band2, B2(Blue)=band1
        
        blue = src.read(1).astype(float)   # B2
        green = src.read(2).astype(float)  # B3
        red = src.read(3).astype(float)    # B4
        nir = src.read(7).astype(float)    # B8
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # --- RGB Composite ---
        rgb = np.dstack([red, green, blue])
        
        # Mask nodata
        nodata_mask = (red == -9999) | (green == -9999) | (blue == -9999)
        rgb_masked = np.ma.masked_where(nodata_mask[:, :, None], rgb)
        
        # Normalize (S2 values are 0-10000)
        rgb_norm = rgb_masked / 10000.0
        rgb_norm = np.clip(rgb_norm, 0, 0.3)  # Clip for contrast
        rgb_norm = rgb_norm / 0.3  # Stretch to 0-1
        
        axes[0].imshow(rgb_norm)
        axes[0].set_title('RGB Composite (B4-B3-B2)', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # --- False Color (NIR-R-G) ---
        false_color = np.dstack([nir, red, green])
        fc_masked = np.ma.masked_where(nodata_mask[:, :, None], false_color)
        fc_norm = fc_masked / 10000.0
        fc_norm = np.clip(fc_norm, 0, 0.5)
        fc_norm = fc_norm / 0.5
        
        axes[1].imshow(fc_norm)
        axes[1].set_title('False Color (B8-B4-B3)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        
        # Overall title
        fig.suptitle(
            f"{test_info['name']}\n"
            f"{test_info['collection']} | {test_info['start']} to {test_info['end']}",
            fontsize=14,
            fontweight='bold',
            y=0.98
        )
        
        # Stats
        nodata_pct = nodata_mask.sum() / nodata_mask.size * 100
        
        print(f"\n{test_info['name']}:")
        print(f"  Shape: {src.shape} pixels")
        print(f"  Bands: {src.count}")
        print(f"  CRS: {src.crs}")
        print(f"  Nodata: {nodata_pct:.1f}%")
        print(f"  Valid pixels: {100-nodata_pct:.1f}%")
        
        # Value ranges
        valid_red = red[~nodata_mask]
        if len(valid_red) > 0:
            print(f"  Red (B4) range: [{valid_red.min():.0f}, {valid_red.max():.0f}]")
        
        plt.tight_layout()
        plt.show()

# %% Visualize all successful downloads
print("\n" + "="*70)
print("VISUALIZING DOWNLOADED IMAGES")
print("="*70)

for result in results:
    if result['success']:
        print("\n" + "-"*70)
        visualize_s2_image(result['file_path'], result['test'])
    else:
        print(f"\n✗ Skipping {result['test']['name']} (download failed)")

# %% Compare L1C vs L2A side-by-side
print("\n" + "="*70)
print("L1C vs L2A COMPARISON")
print("="*70)

# Find matching L1C/L2A pairs
l2a_tests = [r for r in results if r['success'] and r['test']['collection'] == 'L2A']
l1c_tests = [r for r in results if r['success'] and r['test']['collection'] == 'L1C']

for l2a in l2a_tests:
    # Find matching L1C (same location/dates)
    matching_l1c = None
    for l1c in l1c_tests:
        if (l1c['test']['lat'] == l2a['test']['lat'] and
            l1c['test']['lon'] == l2a['test']['lon'] and
            l1c['test']['start'] == l2a['test']['start']):
            matching_l1c = l1c
            break
    
    if matching_l1c:
        print(f"\nComparing: {l2a['test']['name']}")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for idx, (result, title) in enumerate([(l2a, 'L2A'), (matching_l1c, 'L1C')]):
            with rasterio.open(result['file_path']) as src:
                blue = src.read(1).astype(float)
                green = src.read(2).astype(float)
                red = src.read(3).astype(float)
                
                rgb = np.dstack([red, green, blue])
                nodata_mask = (red == -9999) | (green == -9999) | (blue == -9999)
                rgb_masked = np.ma.masked_where(nodata_mask[:, :, None], rgb)
                
                rgb_norm = np.clip(rgb_masked / 10000.0, 0, 0.3) / 0.3
                
                axes[idx].imshow(rgb_norm)
                axes[idx].set_title(f"{title}\nNodata: {nodata_mask.sum()/nodata_mask.size*100:.1f}%", 
                                   fontsize=12, fontweight='bold')
                axes[idx].axis('off')
        
        fig.suptitle(f"L2A vs L1C Comparison\n{l2a['test']['name']}", 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

# %% Summary statistics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nTotal tests: {len(test_cases)}")
print(f"Successful: {sum(r['success'] for r in results)}")
print(f"Failed: {sum(not r['success'] for r in results)}")

print("\nSuccess by collection:")
for collection in ['L1C', 'L2A']:
    col_results = [r for r in results if r['test']['collection'] == collection]
    col_success = sum(r['success'] for r in col_results)
    print(f"  {collection}: {col_success}/{len(col_results)}")

print("\n" + "="*70)
print("✓ Testing complete!")
print("="*70)

# %%