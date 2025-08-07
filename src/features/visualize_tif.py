import rasterio
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import glob
from datetime import datetime, timedelta

# Quality Assessment Criteria (Consistent across all functions):
# - EXCELLENT: ≥15 time steps with >50% coverage (≥15/37)
# - GOOD: ≥10 time steps with >50% coverage (≥10/37) 
# - MODERATE: ≥5 time steps with >50% coverage (≥5/37)
# - POOR: <5 time steps with >50% coverage (<5/37)

# Coverage threshold: >5000 valid pixels per time step (>50% of 100x100 pixels)

features_dir = "data/features/"
tif_files = glob.glob(os.path.join(features_dir, "site_*.tif"))
print(f"Found {len(tif_files)} sites")

# Choose which site to visualize by unique ID
unique_id = "1" 
TIF_PATH = None
JSON_PATH = None

for tif_path in tif_files:
    filename = os.path.basename(tif_path)
    file_unique_id = filename.split('_')[-1].replace('.tif', '')
    if file_unique_id == unique_id:
        TIF_PATH = tif_path
        JSON_PATH = tif_path.replace('.tif', '.json')
        break

if TIF_PATH is None:
    print(f"Site with unique ID {unique_id} not found!")
    exit()

print(f"Visualizing: {os.path.basename(TIF_PATH)} (Unique ID: {unique_id})")

with open(JSON_PATH) as f:
    meta = json.load(f)

bands = meta['bands']
T, B, H, W = meta['shape']
print(f"Metadata: T={T}, B={B}, H={H}, W={W}")
print(f"Bands ({len(bands)}): {bands}")

# Load and reshape image
with rasterio.open(TIF_PATH) as src:
    raw = src.read()

# Reshape to (T, B, H, W)
stack = raw.reshape(T, B, H, W)

# Get NDVI band
ndvi_idx = bands.index('NDVI')
ndvi_data = stack[:, ndvi_idx, :, :]  # Shape: (37, 100, 100)

# Convert scaled values back to proper NDVI range
ndvi_data = np.where(ndvi_data == -9999, np.nan, ndvi_data / 10000.0)

# Calculate global min/max for consistent colorbar - optimize color range for better farmland visualization
valid_data = ndvi_data[~np.isnan(ndvi_data)]
if len(valid_data) > 0:
    # Use more conservative percentiles to avoid extreme values affecting color range
    global_min = np.percentile(valid_data, 2)  # Changed from 1% to 2%
    global_max = np.percentile(valid_data, 98)  # Changed from 99% to 98%
    
    # Ensure color range is suitable for farmland visualization
    if global_min < -0.1:
        global_min = -0.1
    if global_max > 0.8:
        global_max = 0.8
else:
    global_min, global_max = -0.1, 0.8

print(f"NDVI range: {global_min:.3f} to {global_max:.3f}")

n_cols = 7
n_rows = 6
fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, 18))
axes = axes.flatten()

# Use standard matplotlib colormap for NDVI visualization
# 'RdYlGn' (Red-Yellow-Green) is commonly used for vegetation indices
# Red = low NDVI (no vegetation), Green = high NDVI (dense vegetation)
cmap = 'RdYlGn'

for i in range(37):
    img = ndvi_data[i]  # Shape: (100, 100)
    
    ax = axes[i]
    im = ax.imshow(img, cmap=cmap, vmin=global_min, vmax=global_max)
    ax.set_title(f"Time {i}", fontsize=10, fontweight='bold')
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

for ax in axes[37:]:
    ax.axis('off')

plt.suptitle(f"NDVI (Normalized Difference Vegetation Index) Over Time - {os.path.basename(TIF_PATH)}", fontsize=16)
plt.tight_layout()

# Save the NDVI time series plot
ndvi_plot_path = os.path.join("src/features/readme_figures", f"ndvi_time_series_site_{unique_id}.png")
plt.savefig(ndvi_plot_path, dpi=300, bbox_inches='tight')
print(f"Saved NDVI time series plot to: {ndvi_plot_path}")

plt.show()

print(f"\nNDVI Statistics:")
print(f"Global min: {global_min:.3f}")
print(f"Global max: {global_max:.3f}")
print(f"Mean: {np.nanmean(ndvi_data):.3f}")
print(f"Std: {np.nanstd(ndvi_data):.3f}")

# Count masked pixels per time step
print(f"\nMasked pixels per time step:")
for i in range(37):
    masked_count = np.sum(np.isnan(ndvi_data[i]))
    total_pixels = ndvi_data[i].size
    masked_percent = (masked_count / total_pixels) * 100
    print(f"Time {i:2d}: {masked_count:4d} pixels masked ({masked_percent:5.1f}%)")

# Data Quality Assessment
print(f"\nData Quality Assessment:")
print(f"Total valid pixels: {len(valid_data):,}")
print(f"Percentage of valid data: {(len(valid_data)/(37*100*100)*100):.1f}%")

# Count time steps with good coverage (>50% of pixels are valid)
good_coverage_steps = sum(1 for i in range(37) 
                         if np.sum(~np.isnan(ndvi_data[i])) > 5000)  # >50% coverage
print(f"Time steps with >50% coverage: {good_coverage_steps}/37")

# Quality assessment with consistent criteria
if good_coverage_steps >= 15:
    print("✅ Excellent temporal coverage for irrigation detection")
elif good_coverage_steps >= 10:
    print("✅ Good temporal coverage for irrigation detection")
elif good_coverage_steps >= 5:
    print("⚠️ Moderate temporal coverage - may need more sites")
else:
    print("❌ Poor temporal coverage - may need more sites or relaxed cloud filtering")

"""
# Optional:Function to visualize different sites
def visualize_ndvi_time_series(unique_id):
    # Visualize NDVI time series for a specific site by unique ID
    
    # Find the file with this unique ID
    tif_path = None
    for path in tif_files:
        filename = os.path.basename(path)
        file_unique_id = filename.split('_')[-1].replace('.tif', '')
        if file_unique_id == str(unique_id):
            tif_path = path
            break
    
    if tif_path is None:
        print(f"Site with unique ID {unique_id} not found!")
        return
    
    json_path = tif_path.replace('.tif', '.json')
    
    print(f"Visualizing site with unique ID {unique_id}: {os.path.basename(tif_path)}")
    
    # Load data
    with open(json_path) as f:
        meta = json.load(f)
    
    with rasterio.open(tif_path) as src:
        raw = src.read()
    
    # Reshape
    T, B, H, W = meta['shape']
    stack = raw.reshape(T, B, H, W)
    
    # Get NDVI and convert scaling
    ndvi_idx = meta['bands'].index('NDVI')
    ndvi_data = stack[:, ndvi_idx, :, :]
    ndvi_data = np.where(ndvi_data == -9999, np.nan, ndvi_data / 10000.0)
    
    # Calculate global range - optimize color range for better farmland visualization
    valid_data = ndvi_data[~np.isnan(ndvi_data)]
    if len(valid_data) > 0:
        # Use more conservative percentiles to avoid extreme values affecting color range
        global_min = np.percentile(valid_data, 2)  # Changed from 1% to 2%
        global_max = np.percentile(valid_data, 98)  # Changed from 99% to 98%
        
        # Ensure color range is suitable for farmland visualization
        if global_min < -0.1:
            global_min = -0.1
        if global_max > 0.8:
            global_max = 0.8
    else:
        global_min, global_max = -0.1, 0.8
    
    # Create plot
    n_cols = 7
    n_rows = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(21, 18))
    axes = axes.flatten()
    
    # Use color scheme suitable for farmland visualization
    cmap = 'RdYlGn'  # Red-Yellow-Green, Red=low NDVI, Green=high NDVI
    
    for i in range(37):
        img = ndvi_data[i]
        
        ax = axes[i]
        im = ax.imshow(img, cmap=cmap, vmin=global_min, vmax=global_max)
        ax.set_title(f"Time {i}", fontsize=10, fontweight='bold')
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused axes
    for ax in axes[37:]:
        ax.axis('off')
    
    plt.suptitle(f"NDVI (Normalized Difference Vegetation Index) Over Time - {os.path.basename(tif_path)}", fontsize=16)
    plt.tight_layout()
    plt.show()

def compare_sites_quality():
    # Compare data quality across multiple sites
    results = []
    
    print("Analyzing data quality across sites...")
    
    # Analyze all sites, not just first 10
    for i in range(len(tif_files)):
        tif_path = tif_files[i]
        json_path = tif_path.replace('.tif', '.json')
        
        try:
            with open(json_path) as f:
                meta = json.load(f)
            
            with rasterio.open(tif_path) as src:
                raw = src.read()
            
            T, B, H, W = meta['shape']
            stack = raw.reshape(T, B, H, W)
            
            ndvi_idx = meta['bands'].index('NDVI')
            ndvi_data = stack[:, ndvi_idx, :, :]
            ndvi_data = np.where(ndvi_data == -9999, np.nan, ndvi_data / 10000.0)
            
            valid_pixels = np.sum(~np.isnan(ndvi_data))
            good_steps = np.sum([np.sum(~np.isnan(ndvi_data[i])) > 5000 for i in range(37)])
            
            results.append({
                'site': os.path.basename(tif_path),
                'valid_pixels': valid_pixels,
                'good_steps': good_steps,
                'coverage_percent': (valid_pixels/(37*100*100)*100)
            })
        except Exception as e:
            print(f"Error processing {os.path.basename(tif_path)}: {e}")
    
    print(f"\nData Quality Comparison (All {len(results)} sites):")
    print("Unique ID | Site Name                    | Good Steps | Coverage % | Status")
    print("-" * 70)
    
    for r in results:
        # Extract unique ID from filename
        site_name = r['site']
        unique_id = site_name.split('_')[-1].replace('.tif', '')  # Get the last part before .tif
        
        # Consistent quality assessment criteria
        if r['good_steps'] >= 15:
            status = "✅ Excellent"
        elif r['good_steps'] >= 10:
            status = "✅ Good"
        elif r['good_steps'] >= 5:
            status = "⚠️ Moderate"
        else:
            status = "❌ Poor"
        print(f"{unique_id:>8} | {site_name:<30} | {r['good_steps']:>10} | {r['coverage_percent']:>9.1f}% | {status}")
    
    # Summary statistics
    avg_good_steps = np.mean([r['good_steps'] for r in results])
    avg_coverage = np.mean([r['coverage_percent'] for r in results])
    
    print(f"\nSummary:")
    print(f"Average good time steps: {avg_good_steps:.1f}/37")
    print(f"Average coverage: {avg_coverage:.1f}%")
    
    # Consistent overall assessment
    if avg_good_steps >= 15:
        print("🎉 Overall data quality is EXCELLENT for irrigation detection!")
    elif avg_good_steps >= 10:
        print("✅ Overall data quality is GOOD for irrigation detection!")
    elif avg_good_steps >= 5:
        print("⚠️ Overall data quality is MODERATE - consider processing more sites")
    else:
        print("❌ Overall data quality is POOR - consider adjusting cloud filtering parameters")
"""

def find_site_by_name(site_name):
    """Find a specific site by name and return its unique ID"""
    for tif_path in tif_files:
        if site_name in os.path.basename(tif_path):
            filename = os.path.basename(tif_path)
            return filename.split('_')[-1].replace('.tif', '')
    return None

def find_site_by_unique_id(unique_id):
    """Check if a site with this unique ID exists"""
    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        file_unique_id = filename.split('_')[-1].replace('.tif', '')
        if file_unique_id == str(unique_id):
            return True
    return False

# Uncomment to run quality comparison
# compare_sites_quality()

def analyze_temporal_patterns(unique_id):
    """Analyze temporal patterns for a specific site"""
    
    # Find the file with this unique ID
    tif_path = None
    for path in tif_files:
        filename = os.path.basename(path)
        file_unique_id = filename.split('_')[-1].replace('.tif', '')
        if file_unique_id == str(unique_id):
            tif_path = path
            break
    
    if tif_path is None:
        print(f"Site with unique ID {unique_id} not found!")
        return None, None, None
    
    json_path = tif_path.replace('.tif', '.json')
    
    # Load data
    with open(json_path) as f:
        meta = json.load(f)
    
    with rasterio.open(tif_path) as src:
        raw = src.read()
    
    T, B, H, W = meta['shape']
    stack = raw.reshape(T, B, H, W)
    
    # Get NDVI and convert scaling
    ndvi_idx = meta['bands'].index('NDVI')
    ndvi_data = stack[:, ndvi_idx, :, :]
    ndvi_data = np.where(ndvi_data == -9999, np.nan, ndvi_data / 10000.0)
    
    # Calculate coverage per time step
    coverage_per_step = []
    for i in range(37):
        valid_pixels = np.sum(~np.isnan(ndvi_data[i]))
        coverage_percent = (valid_pixels / (100*100)) * 100
        coverage_per_step.append(coverage_percent)
    
    # Find good time steps (>50% coverage)
    good_steps = [i for i, coverage in enumerate(coverage_per_step) if coverage > 50]
    poor_steps = [i for i, coverage in enumerate(coverage_per_step) if coverage <= 50]
    
    print(f"\n=== Temporal Pattern Analysis for Site {unique_id} ===")
    print(f"Site: {os.path.basename(tif_path)}")
    print(f"Year: {meta.get('year', 'Unknown')}")
    print(f"Location: {meta.get('lat', 'Unknown')}, {meta.get('lon', 'Unknown')}")
    
    print(f"\n Coverage Summary:")
    print(f"Good time steps (>50% coverage): {len(good_steps)}/37")
    print(f"Poor time steps (≤50% coverage): {len(poor_steps)}/37")
    print(f"Average coverage: {np.mean(coverage_per_step):.1f}%")
    
    print(f"\n✅ Good Time Steps: {good_steps}")
    print(f"❌ Poor Time Steps: {poor_steps}")
    
    return coverage_per_step, good_steps, poor_steps

def analyze_seasonal_patterns(unique_id):
    """Analyze seasonal patterns for a specific site"""
    
    # Get temporal patterns first
    coverage_per_step, good_steps, poor_steps = analyze_temporal_patterns(unique_id)
    if coverage_per_step is None:
        return
    
    # Find the file to get metadata
    tif_path = None
    for path in tif_files:
        filename = os.path.basename(path)
        file_unique_id = filename.split('_')[-1].replace('.tif', '')
        if file_unique_id == str(unique_id):
            tif_path = path
            break
    
    json_path = tif_path.replace('.tif', '.json')
    
    # Load metadata to get the survey date
    with open(json_path) as f:
        meta = json.load(f)
    
    # Get the survey date (approximate)
    survey_year = meta.get('year', 2020)
    survey_month = meta.get('month', 6)  # Default to June if not available
    survey_day = meta.get('day', 15)     # Default to middle of month
    
    survey_date = datetime(survey_year, survey_month, survey_day)
    
    # Calculate time step dates
    time_step_dates = []
    for i in range(37):
        # Each time step is 10 days apart
        # Time step 18 is the center (survey date)
        days_offset = (i - 18) * 10
        step_date = survey_date + timedelta(days=days_offset)
        time_step_dates.append(step_date)
    
    print(f"\n Seasonal Analysis:")
    print(f"Survey date: {survey_date.strftime('%Y-%m-%d')}")
    
    # Group by month
    monthly_coverage = {}
    for i, (date, coverage) in enumerate(zip(time_step_dates, coverage_per_step)):
        month = date.month
        if month not in monthly_coverage:
            monthly_coverage[month] = []
        monthly_coverage[month].append((i, coverage, date))
    
    print(f"\n Monthly Coverage Patterns:")
    for month in sorted(monthly_coverage.keys()):
        avg_coverage = np.mean([data[1] for data in monthly_coverage[month]])
        month_name = datetime(2020, month, 1).strftime('%B')
        time_steps = [data[0] for data in monthly_coverage[month]]
        print(f"{month_name:>9}: {avg_coverage:5.1f}% coverage (time steps: {time_steps})")
    
    # Determine dry vs wet season
    print(f"\n Seasonal Classification:")
    dry_months = [5, 6, 7, 8, 9, 10]  # May-October
    wet_months = [11, 12, 1, 2, 3, 4]  # November-April
    
    dry_coverage = []
    wet_coverage = []
    dry_steps = []
    wet_steps = []
    
    for month in dry_months:
        if month in monthly_coverage:
            for data in monthly_coverage[month]:
                dry_coverage.append(data[1])
                dry_steps.append(data[0])
    
    for month in wet_months:
        if month in monthly_coverage:
            for data in monthly_coverage[month]:
                wet_coverage.append(data[1])
                wet_steps.append(data[0])
    
    if dry_coverage:
        print(f"Dry season (May-Oct): {np.mean(dry_coverage):.1f}% average coverage (time steps: {dry_steps})")
    if wet_coverage:
        print(f"Wet season (Nov-Apr): {np.mean(wet_coverage):.1f}% average coverage (time steps: {wet_steps})")
    
    # Create seasonal pattern plot with improved colors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    # Define soft, high-contrast colors
    dry_color = '#FF8C42'  # Soft orange
    wet_color = '#4A90E2'  # Soft blue
    line_color = '#2C3E50'  # Dark blue-gray
    threshold_color = '#E74C3C'  # Soft red
    
    # Plot 1: Coverage over time with seasonal coloring
    ax1.plot(range(37), coverage_per_step, color=line_color, linewidth=2.5, label='Coverage %', alpha=0.8)
    ax1.axhline(y=50, color=threshold_color, linestyle='--', alpha=0.8, linewidth=2, label='50% threshold')
    
    # Color by season with improved styling
    for i, (date, coverage) in enumerate(zip(time_step_dates, coverage_per_step)):
        if date.month in dry_months:
            color = dry_color  # Soft orange for dry season
        else:
            color = wet_color  # Soft blue for wet season
        
        ax1.scatter(i, coverage, color=color, s=40, alpha=0.8, edgecolors='white', linewidth=1)
    
    ax1.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Coverage (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Seasonal Coverage Pattern - Site {unique_id}', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.set_facecolor('#F8F9FA')  # Light gray background
    
    # Plot 2: Monthly average coverage with improved styling
    months = sorted(monthly_coverage.keys())
    monthly_avg = [np.mean([data[1] for data in monthly_coverage[month]]) for month in months]
    month_names = [datetime(2020, month, 1).strftime('%b') for month in months]
    
    # Use improved colors for bars
    bar_colors = [dry_color if month in dry_months else wet_color for month in months]
    bars = ax2.bar(range(len(months)), monthly_avg, color=bar_colors, alpha=0.8, 
                   edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, monthly_avg)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Coverage (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Monthly Coverage Pattern - Site {unique_id}', fontsize=14, fontweight='bold', pad=20)
    ax2.set_xticks(range(len(months)))
    ax2.set_xticklabels(month_names, fontsize=11)
    ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax2.set_facecolor('#F8F9FA')  # Light gray background
    
    # Add improved legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=dry_color, alpha=0.8, label='Dry Season (May-Oct)'),
                      Patch(facecolor=wet_color, alpha=0.8, label='Wet Season (Nov-Apr)')]
    ax2.legend(handles=legend_elements, fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save the seasonal patterns plot
    seasonal_plot_path = os.path.join("src/features/readme_figures", f"seasonal_patterns_site_{unique_id}.png")
    plt.savefig(seasonal_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved seasonal patterns plot to: {seasonal_plot_path}")
    
    plt.show()
    
    return time_step_dates, coverage_per_step

def analyze_multi_site_seasonal_patterns():
    """Analyze seasonal patterns across all sites"""
    
    print("=== Multi-Site Seasonal Pattern Analysis ===")
    
    # Collect data from all sites
    all_coverage_data = []
    site_info = []
    
    for tif_path in tif_files:
        try:
            json_path = tif_path.replace('.tif', '.json')
            with open(json_path) as f:
                meta = json.load(f)
            
            with rasterio.open(tif_path) as src:
                raw = src.read()
            
            T, B, H, W = meta['shape']
            stack = raw.reshape(T, B, H, W)
            
            ndvi_idx = meta['bands'].index('NDVI')
            ndvi_data = stack[:, ndvi_idx, :, :]
            ndvi_data = np.where(ndvi_data == -9999, np.nan, ndvi_data / 10000.0)
            
            # Calculate coverage per time step
            coverage_per_step = []
            for i in range(37):
                valid_pixels = np.sum(~np.isnan(ndvi_data[i]))
                coverage_percent = (valid_pixels / (100*100)) * 100
                coverage_per_step.append(coverage_percent)
            
            # Extract site info
            filename = os.path.basename(tif_path)
            unique_id = filename.split('_')[-1].replace('.tif', '')
            lat = meta.get('lat', 0)
            lon = meta.get('lon', 0)
            year = meta.get('year', 0)
            
            all_coverage_data.append(coverage_per_step)
            site_info.append({
                'unique_id': unique_id,
                'lat': lat,
                'lon': lon,
                'year': year,
                'filename': filename
            })
            
        except Exception as e:
            print(f"Error processing {os.path.basename(tif_path)}: {e}")
    
    # Convert to numpy array
    coverage_array = np.array(all_coverage_data)
    
    # Calculate average coverage per time step across all sites
    avg_coverage_per_step = np.mean(coverage_array, axis=0)
    std_coverage_per_step = np.std(coverage_array, axis=0)
    
    print(f"\n Overall Seasonal Patterns:")
    print(f"Analyzed {len(all_coverage_data)} sites")
    
    # Find best and worst time steps
    best_steps = np.argsort(avg_coverage_per_step)[-5:]  # Top 5
    worst_steps = np.argsort(avg_coverage_per_step)[:5]  # Bottom 5
    
    print(f"Best time steps: {best_steps} (coverage: {avg_coverage_per_step[best_steps]:.1f}%)")
    print(f"Worst time steps: {worst_steps} (coverage: {avg_coverage_per_step[worst_steps]:.1f}%)")
    
    # Create comprehensive plot with improved styling
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Define improved colors
    main_color = '#2C3E50'  # Dark blue-gray
    fill_color = '#3498DB'  # Soft blue
    threshold_color = '#E74C3C'  # Soft red
    heatmap_cmap = 'RdYlBu_r'  # Red-Yellow-Blue reversed for better contrast
    
    # Plot 1: Average coverage over time
    ax1.plot(range(37), avg_coverage_per_step, color=main_color, linewidth=2.5, label='Average Coverage', alpha=0.9)
    ax1.fill_between(range(37), avg_coverage_per_step - std_coverage_per_step, 
                     avg_coverage_per_step + std_coverage_per_step, alpha=0.3, color=fill_color, label='±1 Std Dev')
    ax1.axhline(y=50, color=threshold_color, linestyle='--', alpha=0.8, linewidth=2, label='50% threshold')
    ax1.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Coverage (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Overall Temporal Coverage Pattern', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.set_facecolor('#F8F9FA')
    
    # Plot 2: Coverage heatmap with improved colormap
    im = ax2.imshow(coverage_array, cmap=heatmap_cmap, aspect='auto', 
                    extent=[0, 36, 0, len(all_coverage_data)])
    ax2.set_xlabel('Time Step', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Site Index', fontsize=12, fontweight='bold')
    ax2.set_title('Coverage Heatmap (All Sites)', fontsize=14, fontweight='bold', pad=20)
    cbar = plt.colorbar(im, ax=ax2, label='Coverage (%)')
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.set_ylabel('Coverage (%)', fontsize=11, fontweight='bold')
    
    # Plot 3: Geographic pattern with improved styling
    lats = [info['lat'] for info in site_info]
    lons = [info['lon'] for info in site_info]
    avg_coverage = [np.mean(coverage_array[i]) for i in range(len(all_coverage_data))]
    
    scatter = ax3.scatter(lons, lats, c=avg_coverage, cmap='RdYlBu_r', s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax3.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax3.set_title('Geographic Coverage Pattern', fontsize=14, fontweight='bold', pad=20)
    cbar2 = plt.colorbar(scatter, ax=ax3, label='Average Coverage (%)')
    cbar2.ax.tick_params(labelsize=10)
    cbar2.ax.set_ylabel('Average Coverage (%)', fontsize=11, fontweight='bold')
    ax3.set_facecolor('#F8F9FA')
    
    # Plot 4: Year-based pattern with improved styling
    years = [info['year'] for info in site_info]
    year_coverage = {}
    for i, year in enumerate(years):
        if year not in year_coverage:
            year_coverage[year] = []
        year_coverage[year].append(avg_coverage[i])
    
    year_avg = {year: np.mean(coverage) for year, coverage in year_coverage.items()}
    year_std = {year: np.std(coverage) for year, coverage in year_coverage.items()}
    
    years_sorted = sorted(year_avg.keys())
    coverage_values = [year_avg[year] for year in years_sorted]
    coverage_stds = [year_std[year] for year in years_sorted]
    
    bars = ax4.bar(years_sorted, coverage_values, yerr=coverage_stds, alpha=0.8, 
                   color=fill_color, edgecolor='white', linewidth=1.5, capsize=5)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, coverage_values)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average Coverage (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Coverage by Year', fontsize=14, fontweight='bold', pad=20)
    ax4.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax4.set_facecolor('#F8F9FA')
    
    plt.tight_layout()
    plt.show()
    
    return coverage_array, site_info

# Uncomment to run seasonal analysis
analyze_seasonal_patterns(unique_id)