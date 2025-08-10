import rasterio
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob

"""
ENHANCED VISUALIZATION SCRIPT FOR SENTINEL-2 DATA

This script provides visualization functions for Sentinel-2 data with enhanced scaling
and color enhancement for better representation of vegetation, water, and land features.
"""

# Configuration
FEATURES_DIR = "data/features"
OUTPUT_DIR = "src/features/readme_figures"
TIF_FILES = glob.glob(os.path.join(FEATURES_DIR, "site_*.tif"))

# Quality Assessment Criteria
# - EXCELLENT: ≥15 time steps with >50% coverage
# - GOOD: ≥10 time steps with >50% coverage 
# - MODERATE: ≥5 time steps with >50% coverage
# - POOR: <5 time steps with >50% coverage

def find_site_file(unique_id):
    """Find TIF and JSON files for a given site ID"""
    for tif_path in TIF_FILES:
        filename = os.path.basename(tif_path)
        file_unique_id = filename.split('_')[-1].replace('.tif', '')
        if file_unique_id == str(unique_id):
            return tif_path, tif_path.replace('.tif', '.json')
    return None, None

def load_site_data(unique_id):
    """Load and reshape data for a given site"""
    tif_path, json_path = find_site_file(unique_id)
    if tif_path is None:
        print(f"Site {unique_id} not found")
        return None, None, None
    
    with open(json_path) as f:
        meta = json.load(f)
    
    with rasterio.open(tif_path) as src:
        raw = src.read()
    
    T, B, H, W = meta['shape']
    stack = raw.reshape(T, B, H, W)
    
    return stack, meta, tif_path



def stretch_band_enhanced(band_data, lower_percentile=2, upper_percentile=98):
    """Enhanced histogram stretching with adaptive scaling for Sentinel-2 data"""
    valid_data = band_data[~np.isnan(band_data)]
    if len(valid_data) == 0:
        return np.zeros_like(band_data)
    
    min_val = np.percentile(valid_data, lower_percentile)
    max_val = np.percentile(valid_data, upper_percentile)
    
    if max_val > min_val and (max_val - min_val) > 100:
        gamma = 0.7
        stretched = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
        return np.power(stretched, gamma)
    else:
        global_min = np.percentile(valid_data, 1)
        global_max = np.percentile(valid_data, 99)
        if global_max > global_min:
            gamma = 0.7
            stretched = np.clip((band_data - global_min) / (global_max - global_min), 0, 1)
            return np.power(stretched, gamma)
        return np.zeros_like(band_data)

def create_time_series_plot(data, title, output_filename, cmap=None, vmin=None, vmax=None):
    """Create a time series plot with consistent layout"""
    n_cols, n_rows = 7, 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, 18))
    axes = axes.flatten()
    
    for i in range(37):
        img = data[i]
        ax = axes[i]
        
        if cmap:
            # NDVI data (single channel)
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='bilinear')
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('NDVI', fontsize=8, fontweight='bold')
            cbar.ax.tick_params(labelsize=7)
        else:
            # RGB data (3 channels) - need to transpose from (3, H, W) to (H, W, 3)
            if img.shape[0] == 3:  # RGB data
                img = img.transpose(1, 2, 0)
            ax.imshow(img, interpolation='bilinear')
        
        ax.set_title(f"Time {i}", fontsize=10, fontweight='bold')
        ax.axis('off')
    
    for ax in axes[37:]:
        ax.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def create_natural_rgb(red_data, green_data, blue_data, apply_cloud_mask=True):
    """Create more natural-looking RGB by using different scaling approach"""
    if apply_cloud_mask:
        # For cloud-masked data, use nan values
        red_clean = np.where(red_data == -9999, np.nan, red_data)
        green_clean = np.where(green_data == -9999, np.nan, green_data)
        blue_clean = np.where(blue_data == -9999, np.nan, blue_data)
    else:
        # For raw data, set -9999 to 0 for visualization
        red_clean = np.where(red_data == -9999, 0, red_data)
        green_clean = np.where(green_data == -9999, 0, green_data)
        blue_clean = np.where(blue_data == -9999, 0, blue_data)
    
    # CRITICAL: Divide by 10,000 to normalize Sentinel-2 TOA values (0-1 range)
    # The data stored in .tif files are the original 0-10000 range values
    red_clean = red_clean / 10000.0
    green_clean = green_clean / 10000.0
    blue_clean = blue_clean / 10000.0
    
    # Get valid data for scaling (now in 0-1 range)
    red_valid = red_clean[red_clean != 0] if not apply_cloud_mask else red_clean[~np.isnan(red_clean)]
    green_valid = green_clean[green_clean != 0] if not apply_cloud_mask else green_clean[~np.isnan(green_clean)]
    blue_valid = blue_clean[blue_clean != 0] if not apply_cloud_mask else blue_clean[~np.isnan(blue_clean)]
    
    if len(red_valid) > 0 and len(green_valid) > 0 and len(blue_valid) > 0:
        # Use 2nd and 98th percentiles for robust scaling (avoid outliers)
        red_min, red_max = np.percentile(red_valid, [2, 98])
        green_min, green_max = np.percentile(green_valid, [2, 98])
        blue_min, blue_max = np.percentile(blue_valid, [2, 98])
        
        # Ensure minimum range for visualization
        min_range = 0.05  # Minimum 5% range to avoid flat images
        if red_max - red_min < min_range:
            mid = (red_max + red_min) / 2
            red_min = max(0, mid - min_range/2)
            red_max = min(1, mid + min_range/2)
        if green_max - green_min < min_range:
            mid = (green_max + green_min) / 2
            green_min = max(0, mid - min_range/2)
            green_max = min(1, mid + min_range/2)
        if blue_max - blue_min < min_range:
            mid = (blue_max + blue_min) / 2
            blue_min = max(0, mid - min_range/2)
            blue_max = min(1, mid + min_range/2)
        
        # Scale each band independently to 0-1 range
        red_scaled = np.clip((red_clean - red_min) / (red_max - red_min), 0, 1)
        green_scaled = np.clip((green_clean - green_min) / (green_max - green_min), 0, 1)
        blue_scaled = np.clip((blue_clean - blue_min) / (blue_max - blue_min), 0, 1)
        
        return red_scaled, green_scaled, blue_scaled
    else:
        return red_clean, green_clean, blue_clean

def visualize_rgb(unique_id, apply_cloud_mask=True):
    """Visualize RGB images with or without cloud masking"""
    stack, meta, _ = load_site_data(unique_id)
    if stack is None:
        return
    
    bands = meta['bands']
    try:
        red_idx = bands.index('B4')    # Red band (B4)
        green_idx = bands.index('B3')  # Green band (B3) 
        blue_idx = bands.index('B2')   # Blue band (B2)
    except ValueError:
        print("RGB bands not found")
        return
    
    red_data = stack[:, red_idx, :, :]
    green_data = stack[:, green_idx, :, :]
    blue_data = stack[:, blue_idx, :, :]
    
    # Use the natural RGB approach
    red_scaled, green_scaled, blue_scaled = create_natural_rgb(red_data, green_data, blue_data, apply_cloud_mask)
    
    # Stack RGB bands and transpose for proper plotting
    rgb_data = np.stack([red_scaled, green_scaled, blue_scaled], axis=1)
    
    mask_status = "after" if apply_cloud_mask else "before"
    title = f"RGB Images {mask_status.title()} Cloud Masking - Site {unique_id}"
    filename = f"rgb_{mask_status}_cloud_mask_site_{unique_id}.png"
    
    create_time_series_plot(rgb_data, title, filename)

def visualize_ndvi(unique_id, apply_cloud_mask=True):
    """Visualize NDVI with or without cloud masking"""
    stack, meta, _ = load_site_data(unique_id)
    if stack is None:
        return
    
    bands = meta['bands']
    try:
        ndvi_idx = bands.index('NDVI')
    except ValueError:
        print("NDVI band not found")
        return
    
    ndvi_data = stack[:, ndvi_idx, :, :]
    
    if apply_cloud_mask:
        ndvi_data = np.where(ndvi_data == -9999, np.nan, ndvi_data / 10000.0)
        mask_status = "after"
    else:
        ndvi_data = ndvi_data / 10000.0
        mask_status = "before"
    
    valid_data = ndvi_data[~np.isnan(ndvi_data)] if apply_cloud_mask else ndvi_data[ndvi_data != -0.9999]
    
    if len(valid_data) > 0:
        global_min = max(np.percentile(valid_data, 2), -0.15)
        global_max = min(np.percentile(valid_data, 98), 0.85)
        
        if global_max - global_min < 0.15:
            mid_point = (global_max + global_min) / 2
            global_min = max(mid_point - 0.15, -0.15)
            global_max = min(mid_point + 0.15, 0.85)
    else:
        global_min, global_max = -0.15, 0.85
    
    title = f"NDVI {mask_status.title()} Cloud Masking - Site {unique_id}"
    filename = f"ndvi_{mask_status}_cloud_mask_site_{unique_id}.png"
    
    create_time_series_plot(ndvi_data, title, filename, cmap='RdYlGn_r', vmin=global_min, vmax=global_max)

def analyze_data_scaling(unique_id):
    """Analyze data scaling characteristics"""
    stack, meta, _ = load_site_data(unique_id)
    if stack is None:
        return
    
    bands = meta['bands']
    T, B, H, W = meta['shape']
    
    print(f"=== Data Scaling Analysis for Site {unique_id} ===")
    print(f"Data: {stack.shape}, {stack.dtype}")
    
    # Analyze RGB bands
    try:
        red_idx = bands.index('B4')
        green_idx = bands.index('B3')
        blue_idx = bands.index('B2')
        
        print(f"\n--- RGB Band Analysis ---")
        for t in range(min(3, T)):
            red_data = stack[t, red_idx, :, :]
            green_data = stack[t, green_idx, :, :]
            blue_data = stack[t, blue_idx, :, :]
            
            red_valid = red_data[red_data != -9999]
            green_valid = green_data[green_data != -9999]
            blue_valid = blue_data[blue_data != -9999]
            
            if len(red_valid) > 0:
                print(f"Time Step {t}: R({np.min(red_valid):.0f}-{np.max(red_valid):.0f}), "
                      f"G({np.min(green_valid):.0f}-{np.max(green_valid):.0f}), "
                      f"B({np.min(blue_valid):.0f}-{np.max(blue_valid):.0f})")
            else:
                print(f"Time Step {t}: All pixels cloudy")
        
        # Overall statistics
        all_red = stack[:, red_idx, :, :].flatten()
        all_red_valid = all_red[all_red != -9999]
        if len(all_red_valid) > 0:
            print(f"Overall RGB ranges: R({np.min(all_red_valid):.0f}-{np.max(all_red_valid):.0f})")
    
    except ValueError:
        print("RGB bands not found")
    
    # Analyze NDVI band
    try:
        ndvi_idx = bands.index('NDVI')
        print(f"\n--- NDVI Band Analysis ---")
        
        ndvi_data = stack[:, ndvi_idx, :, :]
        all_ndvi = ndvi_data.flatten()
        all_ndvi_valid = all_ndvi[all_ndvi != -9999]
        
        if len(all_ndvi_valid) > 0:
            all_ndvi_scaled = all_ndvi_valid / 10000.0
            print(f"Global NDVI range: {np.min(all_ndvi_scaled):.3f} to {np.max(all_ndvi_scaled):.3f}")
            print(f"Global percentiles: 1%={np.percentile(all_ndvi_scaled, 1):.3f}, "
                  f"50%={np.percentile(all_ndvi_scaled, 50):.3f}, "
                  f"99%={np.percentile(all_ndvi_scaled, 99):.3f}")
    
    except ValueError:
        print("NDVI band not found")
    
def assess_data_quality(unique_id):
    """Assess data quality based on coverage"""
    stack, meta, _ = load_site_data(unique_id)
    if stack is None:
        return
    
    bands = meta['bands']
    try:
        ndvi_idx = bands.index('NDVI')
        ndvi_data = stack[:, ndvi_idx, :, :]
        ndvi_data = np.where(ndvi_data == -9999, np.nan, ndvi_data / 10000.0)
        
        good_coverage_steps = sum(1 for i in range(37) 
                                 if np.sum(~np.isnan(ndvi_data[i])) > 5000)
        
        valid_data = ndvi_data[~np.isnan(ndvi_data)]
        coverage_percent = (len(valid_data)/(37*100*100)*100)
        
        quality = ('✅ Excellent' if good_coverage_steps >= 15 else 
                  '✅ Good' if good_coverage_steps >= 10 else 
                  '⚠️ Moderate' if good_coverage_steps >= 5 else '❌ Poor')
        
        print(f"\n=== Data Quality Assessment ===")
        print(f"Valid data: {len(valid_data):,} pixels ({coverage_percent:.1f}%)")
        print(f"Good coverage: {good_coverage_steps}/37 time steps")
        print(f"Quality: {quality}")
        
    except ValueError:
        print("NDVI band not found for quality assessment")

def create_ndvi_time_series(unique_id):
    """Create enhanced NDVI time series visualization"""
    stack, meta, _ = load_site_data(unique_id)
    if stack is None:
        return
    
    bands = meta['bands']
    try:
        ndvi_idx = bands.index('NDVI')
    except ValueError:
        print("NDVI band not found")
        return
    
    ndvi_data = stack[:, ndvi_idx, :, :]
    ndvi_data = np.where(ndvi_data == -9999, np.nan, ndvi_data / 10000.0)
    
    valid_data = ndvi_data[~np.isnan(ndvi_data)]
    if len(valid_data) > 0:
        global_min = max(np.percentile(valid_data, 2), -0.15)
        global_max = min(np.percentile(valid_data, 98), 0.85)
        
        if global_max - global_min < 0.15:
            mid_point = (global_max + global_min) / 2
            global_min = max(mid_point - 0.15, -0.15)
            global_max = min(mid_point + 0.15, 0.85)
    else:
        global_min, global_max = -0.15, 0.85
    
    title = f"Enhanced NDVI Time Series - Site {unique_id}"
    filename = f"ndvi_time_series_site_{unique_id}.png"
    
    create_time_series_plot(ndvi_data, title, filename, cmap='RdYlGn_r', vmin=global_min, vmax=global_max)

def main():
    """Main function to run all visualizations"""
    unique_id = "1"  # Default site ID
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Run analysis and visualizations
    analyze_data_scaling(unique_id)
    visualize_rgb(unique_id, apply_cloud_mask=False)
    visualize_rgb(unique_id, apply_cloud_mask=True)
    visualize_ndvi(unique_id, apply_cloud_mask=False)
    visualize_ndvi(unique_id, apply_cloud_mask=True)
    create_ndvi_time_series(unique_id)
    assess_data_quality(unique_id)

if __name__ == "__main__":
    main()