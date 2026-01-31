"""
EVI time series visualization for irrigation analysis.

This module provides functions to:
1. Extract EVI time series from satellite stacks (Sentinel-2 or PlanetScope)
2. Group pixels by irrigation status
3. Create smoothed time series plots showing irrigation patterns

Supports both Sentinel-2 (10 bands) and PlanetScope (4 bands) via sensor parameter.
"""

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from glob import glob
from scipy.ndimage import uniform_filter1d

from ...utils.utils import find_project_root, get_data_root
from .satellite_visualization import (
    get_labeled_timestep, SENSOR_CONFIG, get_features_dir, get_irrigation_table_path
)


def _get_project_root():
    """Get the project root directory."""
    return find_project_root(os.path.dirname(__file__))


def get_survey_info(site):
    """
    Look up survey and internal_id for a given site.

    Parameters:
        site (int or str): The site from the stack filename

    Returns:
        tuple: (survey, internal_id) or (None, None) if not found
    """
    import re

    irrigation_df = pd.read_csv(get_irrigation_table_path())
    site = int(site)

    matches = irrigation_df[irrigation_df['site_id'] == site]
    if len(matches) == 0:
        return None, None

    row = matches.iloc[0]

    # Extract survey from source_file (e.g., "AB_101-125.zip" -> "101-125")
    source_file = row.get('source_file', '')
    survey = None
    if pd.notna(source_file):
        match = re.search(r'(\d+-\d+)', str(source_file))
        if match:
            survey = match.group(1)

    internal_id = row.get('internal_id', None)
    if pd.notna(internal_id):
        internal_id = int(internal_id)
    else:
        internal_id = None

    return survey, internal_id


def compute_evi(nir, red, blue, nodata=0, scale=10000.0):
    """
    Compute Enhanced Vegetation Index (EVI).

    EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)

    Parameters:
        nir, red, blue: Arrays of band values (in DN units, 0-10000 scale)
        nodata: Value to use for invalid data
        scale: Scale factor to convert DN to reflectance (default 10000)

    Returns:
        Array of EVI values, with NaN where inputs are invalid
    """
    # Handle nodata
    valid = (nir != nodata) & (red != nodata) & (blue != nodata)

    # Initialize output with NaN
    evi = np.full_like(nir, np.nan, dtype=np.float32)

    # Compute EVI where valid
    if valid.any():
        # Convert to reflectance [0-1]
        nir_v = nir[valid].astype(np.float32) / scale
        red_v = red[valid].astype(np.float32) / scale
        blue_v = blue[valid].astype(np.float32) / scale

        # Standard EVI formula with L=1, C1=6, C2=7.5, G=2.5
        denom = nir_v + 6.0 * red_v - 7.5 * blue_v + 1.0
        denom = np.where(np.abs(denom) < 1e-6, 1e-6, denom)

        evi[valid] = 2.5 * (nir_v - red_v) / denom

        # Clip to reasonable range
        evi[valid] = np.clip(evi[valid], -1.0, 1.0)

    return evi


def extract_evi_timeseries(stack_path, sensor='sentinel2', nodata=0):
    """
    Extract EVI time series for all pixels from a satellite stack.

    The stack is stored with shape (num_bands * T, H, W) where bands are grouped:
    - For Sentinel-2: Bands 1 to T: B2 (Blue), T+1 to 2T: B3 (Green), etc.
    - For PlanetScope: Bands 1 to T: Blue, T+1 to 2T: Green, 2T+1 to 3T: Red, 3T+1 to 4T: NIR

    Parameters:
        stack_path: Path to the stack file
        sensor (str): Sensor type ('sentinel2' or 'planetscope')
        nodata: Nodata value in the stack

    Returns:
        tuple: (evi_array, valid_mask)
            - evi_array: shape (n_timesteps, height, width)
            - valid_mask: shape (n_timesteps, height, width) - True where data is valid
    """
    config = SENSOR_CONFIG[sensor]
    n_bands = config['n_bands']
    band_indices = config['band_indices']

    with rasterio.open(stack_path) as src:
        total_bands = src.count
        n_timesteps = total_bands // n_bands

        height, width = src.height, src.width
        evi_array = np.full((n_timesteps, height, width), np.nan, dtype=np.float32)
        valid_mask = np.zeros((n_timesteps, height, width), dtype=bool)

        for t in range(n_timesteps):
            # Stack layout: all timesteps for band 0, then all for band 1, etc.
            # Band index in file = band_type * n_timesteps + timestep
            nir_idx = band_indices['nir'] * n_timesteps + t
            red_idx = band_indices['red'] * n_timesteps + t
            blue_idx = band_indices['blue'] * n_timesteps + t

            # Read bands (1-indexed in rasterio)
            nir = src.read(nir_idx + 1).astype(np.float32)
            red = src.read(red_idx + 1).astype(np.float32)
            blue = src.read(blue_idx + 1).astype(np.float32)

            # Check for valid data
            valid = (nir != nodata) & (red != nodata) & (blue != nodata)
            valid_mask[t] = valid

            # Compute EVI
            evi = compute_evi(nir, red, blue, nodata)
            evi_array[t] = np.where(valid, evi, np.nan)

    return evi_array, valid_mask


def load_irrigation_mask(label_path, min_certainty=3):
    """
    Load binary irrigation mask from label file.

    Parameters:
        label_path: Path to the label file
        min_certainty: Minimum certainty to consider irrigated

    Returns:
        numpy.ndarray: Boolean mask of irrigated pixels
    """
    with rasterio.open(label_path) as src:
        binary = src.read(2)  # Band 2 is binary mask
        certainty = src.read(8)  # Band 8 is certainty

    # Irrigated = binary > 0 AND certainty >= min_certainty
    # Note: certainty is stored in the label file, already filtered at creation time
    return binary > 0


def sample_pixel_timeseries(evi_array, mask, n_samples=50, min_valid_frac=0.5):
    """
    Sample pixel time series from EVI array based on mask.

    Parameters:
        evi_array: shape (n_timesteps, height, width)
        mask: Boolean mask indicating which pixels to sample
        n_samples: Maximum number of pixels to sample
        min_valid_frac: Minimum fraction of valid timesteps for a pixel

    Returns:
        numpy.ndarray: shape (n_sampled, n_timesteps) - sampled time series
    """
    n_timesteps = evi_array.shape[0]

    # Get coordinates of masked pixels
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.array([])

    # Extract time series for all masked pixels
    pixel_ts = evi_array[:, rows, cols].T  # shape (n_pixels, n_timesteps)

    # Filter by minimum valid fraction
    valid_count = np.sum(~np.isnan(pixel_ts), axis=1)
    valid_frac = valid_count / n_timesteps
    valid_pixels = valid_frac >= min_valid_frac

    pixel_ts = pixel_ts[valid_pixels]

    if len(pixel_ts) == 0:
        return np.array([])

    # Sample if too many
    if len(pixel_ts) > n_samples:
        indices = np.random.choice(len(pixel_ts), n_samples, replace=False)
        pixel_ts = pixel_ts[indices]

    return pixel_ts


def smooth_timeseries(ts, window=3):
    """
    Smooth time series using a moving average.

    Parameters:
        ts: Time series array (can be 1D or 2D)
        window: Window size for smoothing

    Returns:
        Smoothed time series (same shape as input)
    """
    if ts.ndim == 1:
        # Handle NaN values - interpolate first
        valid = ~np.isnan(ts)
        if not valid.any():
            return ts

        ts_interp = np.interp(
            np.arange(len(ts)),
            np.where(valid)[0],
            ts[valid]
        )
        return uniform_filter1d(ts_interp, size=window, mode='nearest')
    else:
        # Apply to each row
        return np.array([smooth_timeseries(row, window) for row in ts])


def _get_timestep_dates(stack_path, n_timesteps, sensor='sentinel2'):
    """
    Get actual dates for each timestep from metadata or compute from filename.

    Parameters:
        stack_path: Path to the stack file
        n_timesteps: Number of timesteps in the stack
        sensor: Sensor type

    Returns:
        tuple: (list of datetime objects for each timestep center, labeled_date, labeled_timestep)
    """
    import json
    from datetime import datetime, timedelta

    stack_name = os.path.basename(stack_path).replace('_stack.tif', '')
    parts = stack_name.split('_')
    date_str = parts[1]  # e.g., '2021.09.16'
    labeled_date = datetime.strptime(date_str, '%Y.%m.%d')

    # Try to load metadata
    features_dir = os.path.dirname(stack_path)
    meta_path = os.path.join(features_dir, f'{stack_name}_metadata.json')

    timestep_dates = []
    labeled_timestep = 21  # Default fallback

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        windows = meta.get('windows', [])
        for i, window in enumerate(windows):
            start = datetime.strptime(window['date_range'][0], '%Y-%m-%d')
            end = datetime.strptime(window['date_range'][1], '%Y-%m-%d')
            # Use center of window as the date
            center = start + (end - start) / 2
            timestep_dates.append(center)

            # Check if this window contains the labeled date
            if start <= labeled_date < end:
                labeled_timestep = i
    else:
        # Fallback: compute dates assuming 10-day windows centered on labeled date
        timestep_days = 10
        labeled_timestep = 21
        for t in range(n_timesteps):
            days_offset = (t - labeled_timestep) * timestep_days
            timestep_dates.append(labeled_date + timedelta(days=days_offset))

    return timestep_dates, labeled_date, labeled_timestep


def plot_evi_timeseries(stack_path, label_path=None, ax=None, figsize=(12, 6),
                        n_samples=30, smooth_window=3, show_individual=True,
                        title=None, start_day_of_year=None, sensor='sentinel2'):
    """
    Plot EVI time series for irrigated vs non-irrigated pixels.

    Parameters:
        stack_path: Path to the satellite stack
        label_path: Path to the label file. If None, shows all pixels.
        ax: Matplotlib axes to plot on
        figsize: Figure size if creating new figure
        n_samples: Maximum number of pixels to sample per class. If fewer pixels
            are available for a class, all available pixels are used.
        smooth_window: Window size for time series smoothing (moving average).
            Uses scipy.ndimage.uniform_filter1d. Any positive integer works.
            A value of 1 means no smoothing. Missing values (NaN) are linearly
            interpolated before smoothing is applied.
        show_individual: Whether to show individual pixel traces
        title: Plot title
        start_day_of_year: (Deprecated) No longer used - dates are from metadata
        sensor (str): Sensor type ('sentinel2' or 'planetscope')

    Returns:
        matplotlib.axes.Axes: The axes with the plot
    """
    import matplotlib.dates as mdates
    from datetime import datetime

    config = SENSOR_CONFIG[sensor]

    # Extract EVI time series
    evi_array, valid_mask = extract_evi_timeseries(stack_path, sensor)
    n_timesteps = evi_array.shape[0]

    # Get actual dates for timesteps
    timestep_dates, labeled_date, labeled_timestep = _get_timestep_dates(
        stack_path, n_timesteps, sensor
    )

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Colors
    irr_color = "#2c70a0"  # Blue for irrigated
    non_irr_color = '#d62728'  # Red for non-irrigated

    if label_path is not None:
        # Load irrigation mask
        irr_mask = load_irrigation_mask(label_path)
        non_irr_mask = ~irr_mask

        # Sample pixels from each class
        irr_ts = sample_pixel_timeseries(evi_array, irr_mask, n_samples)
        non_irr_ts = sample_pixel_timeseries(evi_array, non_irr_mask, n_samples)

        # Plot individual traces (faded)
        if show_individual:
            if len(irr_ts) > 0:
                for ts in smooth_timeseries(irr_ts, smooth_window):
                    ax.plot(timestep_dates, ts, color=irr_color, alpha=0.15, linewidth=0.5)
            if len(non_irr_ts) > 0:
                for ts in smooth_timeseries(non_irr_ts, smooth_window):
                    ax.plot(timestep_dates, ts, color=non_irr_color, alpha=0.15, linewidth=0.5)

        # Plot mean time series (bold)
        if len(irr_ts) > 0:
            irr_mean = np.nanmean(smooth_timeseries(irr_ts, smooth_window), axis=0)
            ax.plot(timestep_dates, irr_mean, color=irr_color, linewidth=2.5, label='Irrigated')

        if len(non_irr_ts) > 0:
            non_irr_mean = np.nanmean(smooth_timeseries(non_irr_ts, smooth_window), axis=0)
            ax.plot(timestep_dates, non_irr_mean, color=non_irr_color, linewidth=2.5, label='Non-irrigated')
    else:
        # No labels - show all pixels
        all_ts = sample_pixel_timeseries(evi_array, np.ones_like(evi_array[0], dtype=bool), n_samples)

        if show_individual and len(all_ts) > 0:
            for ts in smooth_timeseries(all_ts, smooth_window):
                ax.plot(timestep_dates, ts, color='gray', alpha=0.2, linewidth=0.5)

        if len(all_ts) > 0:
            all_mean = np.nanmean(smooth_timeseries(all_ts, smooth_window), axis=0)
            ax.plot(timestep_dates, all_mean, color='black', linewidth=2.5, label='All pixels')

    # Add vertical line at labeled date (add to legend)
    ax.axvline(x=labeled_date, color='gray', linestyle='--', alpha=0.7,
               linewidth=1.5, label='Label date')

    # Legend
    ax.legend(loc='upper right')

    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Labels
    ax.set_xlabel('Date')
    ax.set_ylabel('EVI')
    ax.set_ylim(-0.1, 0.7)

    # Title
    if title is None:
        stack_name = os.path.basename(stack_path).replace('_stack.tif', '')
        parts = stack_name.split('_')
        sensor_name = 'PlanetScope' if sensor == 'planetscope' else 'Sentinel-2'
        if len(parts) >= 3:
            site = parts[0]
            date = parts[1]
            # Look up survey and internal_id
            survey, internal_id = get_survey_info(site)
            if survey and internal_id:
                title = f'{sensor_name} EVI - Survey {survey}, ID {internal_id} (Site {site}, {date})'
            else:
                title = f'{sensor_name} EVI - Site {site}, {date}'

    ax.set_title(title)

    return ax


def plot_clustered_timeseries(stack_path, label_path=None, ax=None, figsize=(12, 6),
                               n_clusters=4, smooth_window=3, title=None, sensor='sentinel2'):
    """
    Plot EVI time series with automatic clustering of temporal patterns.

    This is useful for exploring the data without relying on labels.

    Parameters:
        stack_path: Path to the satellite stack
        label_path: Path to the label file (for coloring clusters by irrigation status)
        ax: Matplotlib axes to plot on
        figsize: Figure size if creating new figure
        n_clusters: Number of clusters to identify
        smooth_window: Window size for time series smoothing
        title: Plot title
        sensor (str): Sensor type ('sentinel2' or 'planetscope')

    Returns:
        matplotlib.axes.Axes: The axes with the plot
    """
    import matplotlib.dates as mdates
    from sklearn.cluster import KMeans

    # Extract EVI time series
    evi_array, valid_mask = extract_evi_timeseries(stack_path, sensor)
    n_timesteps = evi_array.shape[0]
    height, width = evi_array.shape[1], evi_array.shape[2]

    # Get actual dates for timesteps
    timestep_dates, labeled_date, labeled_timestep = _get_timestep_dates(
        stack_path, n_timesteps, sensor
    )

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Flatten pixels and filter valid ones
    pixel_ts = evi_array.reshape(n_timesteps, -1).T  # (n_pixels, n_timesteps)
    valid_frac = np.sum(~np.isnan(pixel_ts), axis=1) / n_timesteps
    valid_pixels = valid_frac >= 0.5

    pixel_ts_valid = pixel_ts[valid_pixels]

    if len(pixel_ts_valid) < n_clusters:
        print(f"Warning: Only {len(pixel_ts_valid)} valid pixels, need at least {n_clusters}")
        return ax

    # Interpolate NaN values for clustering
    pixel_ts_filled = np.zeros_like(pixel_ts_valid)
    for i, ts in enumerate(pixel_ts_valid):
        valid = ~np.isnan(ts)
        if valid.any():
            pixel_ts_filled[i] = np.interp(
                np.arange(len(ts)),
                np.where(valid)[0],
                ts[valid]
            )

    # Smooth before clustering
    pixel_ts_smooth = smooth_timeseries(pixel_ts_filled, smooth_window)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixel_ts_smooth)

    # Get cluster colors
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # If we have irrigation labels, compute fraction irrigated per cluster
    cluster_info = []
    if label_path is not None:
        irr_mask = load_irrigation_mask(label_path)
        irr_flat = irr_mask.flatten()[valid_pixels]

        for c in range(n_clusters):
            cluster_mask = labels == c
            n_in_cluster = cluster_mask.sum()
            n_irrigated = (cluster_mask & irr_flat).sum()
            frac_irrigated = n_irrigated / n_in_cluster if n_in_cluster > 0 else 0
            cluster_info.append({
                'cluster': c,
                'n_pixels': n_in_cluster,
                'frac_irrigated': frac_irrigated
            })

    # Plot cluster means
    for c in range(n_clusters):
        cluster_mask = labels == c
        cluster_ts = pixel_ts_smooth[cluster_mask]
        cluster_mean = np.mean(cluster_ts, axis=0)

        label_str = f'Cluster {c+1} (n={cluster_mask.sum()})'
        if cluster_info:
            frac_irr = cluster_info[c]['frac_irrigated']
            label_str += f' [{frac_irr:.0%} irrigated]'

        ax.plot(timestep_dates, cluster_mean, color=colors[c], linewidth=2.5, label=label_str)

    # Add vertical line at labeled date
    ax.axvline(x=labeled_date, color='gray', linestyle='--', alpha=0.7,
               linewidth=1.5, label='Label date')

    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Labels
    ax.set_xlabel('Date')
    ax.set_ylabel('EVI')
    ax.set_ylim(-0.1, 0.7)
    ax.legend(loc='upper right', fontsize=8)

    # Title
    if title is None:
        stack_name = os.path.basename(stack_path).replace('_stack.tif', '')
        sensor_name = 'PlanetScope' if sensor == 'planetscope' else 'Sentinel-2'
        title = f'{sensor_name} EVI Clusters - {stack_name}'

    ax.set_title(title)

    return ax


if __name__ == "__main__":
    # Test the visualization
    from .satellite_visualization import find_labels_for_stack, load_label_mask

    # Test Sentinel-2
    print("Testing Sentinel-2 EVI visualization...")
    features_dir = get_features_dir(sensor='sentinel2')
    stack_files = glob(os.path.join(features_dir, '*_stack.tif'))

    # Find a stack with irrigation labels
    for stack_path in stack_files[:50]:
        labels = find_labels_for_stack(stack_path)
        if labels:
            label_path = labels[0]
            mask = load_label_mask(label_path, 'binary')
            if mask.sum() > 10:  # Need enough irrigated pixels
                print(f"Testing with: {os.path.basename(stack_path)}")

                fig, axes = plt.subplots(1, 2, figsize=(18, 6))
                plot_evi_timeseries(stack_path, label_path, ax=axes[0], sensor='sentinel2')
                plot_clustered_timeseries(stack_path, label_path, ax=axes[1], sensor='sentinel2')
                plt.tight_layout()
                plt.show()
                break

    # Test PlanetScope if available
    print("\nTesting PlanetScope EVI visualization...")
    planet_dir = get_features_dir(sensor='planetscope')
    if os.path.exists(planet_dir):
        planet_files = glob(os.path.join(planet_dir, '*_stack.tif'))
        for stack_path in planet_files[:50]:
            labels = find_labels_for_stack(stack_path)
            if labels:
                label_path = labels[0]
                mask = load_label_mask(label_path, 'binary')
                if mask.sum() > 10:
                    print(f"Testing with: {os.path.basename(stack_path)}")

                    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
                    plot_evi_timeseries(stack_path, label_path, ax=axes[0], sensor='planetscope')
                    plot_clustered_timeseries(stack_path, label_path, ax=axes[1], sensor='planetscope')
                    plt.tight_layout()
                    plt.show()
                    break
