"""
EVI time series visualization for irrigation analysis.

This module provides functions to:
1. Extract EVI time series from Sentinel-2 stacks
2. Group pixels by irrigation status
3. Create smoothed time series plots showing irrigation patterns
"""

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from glob import glob
from scipy.ndimage import uniform_filter1d

from ..utils.utils import find_project_root, get_data_root
from .sentinel2_visualization import get_labeled_timestep


def _get_project_root():
    """Get the project root directory."""
    return find_project_root(os.path.dirname(__file__))


def get_features_dir(version='20260107_180813'):
    """Get the features directory for a specific version."""
    return os.path.join(get_data_root(), 'features', version)


# Band indices within each 10-band timestep
BAND_INDICES = {
    'B2': 0,   # Blue
    'B3': 1,   # Green
    'B4': 2,   # Red
    'B5': 3,   # Red Edge 1
    'B6': 4,   # Red Edge 2
    'B7': 5,   # Red Edge 3
    'B8': 6,   # NIR
    'B8A': 7,  # NIR Narrow
    'B11': 8,  # SWIR 1
    'B12': 9,  # SWIR 2
}

# Time series structure
N_BANDS = 10
N_TIMESTEPS = 42  # 36 windows + 3 buffer on each side
TIMESTEP_DAYS = 10


def compute_evi(nir, red, blue, nodata=-9999, scale=10000.0):
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


def extract_evi_timeseries(stack_path, nodata=-9999):
    """
    Extract EVI time series for all pixels from a Sentinel-2 stack.

    The stack is stored with shape (num_bands * T, H, W) where bands are grouped:
    - Bands 1 to T: B2 (Blue) for all timesteps
    - Bands T+1 to 2T: B3 (Green) for all timesteps
    - etc.

    Parameters:
        stack_path: Path to the stack file
        nodata: Nodata value in the stack

    Returns:
        tuple: (evi_array, valid_mask)
            - evi_array: shape (n_timesteps, height, width)
            - valid_mask: shape (n_timesteps, height, width) - True where data is valid
    """
    with rasterio.open(stack_path) as src:
        total_bands = src.count
        n_timesteps = total_bands // N_BANDS

        height, width = src.height, src.width
        evi_array = np.full((n_timesteps, height, width), np.nan, dtype=np.float32)
        valid_mask = np.zeros((n_timesteps, height, width), dtype=bool)

        for t in range(n_timesteps):
            # Stack layout: all timesteps for band 0, then all for band 1, etc.
            # Band index in file = band_type * n_timesteps + timestep
            nir_idx = BAND_INDICES['B8'] * n_timesteps + t   # NIR
            red_idx = BAND_INDICES['B4'] * n_timesteps + t   # Red
            blue_idx = BAND_INDICES['B2'] * n_timesteps + t  # Blue

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


def plot_evi_timeseries(stack_path, label_path=None, ax=None, figsize=(12, 6),
                        n_samples=30, smooth_window=3, show_individual=True,
                        title=None, start_day_of_year=None):
    """
    Plot EVI time series for irrigated vs non-irrigated pixels.

    Parameters:
        stack_path: Path to the Sentinel-2 stack
        label_path: Path to the label file. If None, shows all pixels.
        ax: Matplotlib axes to plot on
        figsize: Figure size if creating new figure
        n_samples: Number of pixels to sample per class
        smooth_window: Window size for time series smoothing
        show_individual: Whether to show individual pixel traces
        title: Plot title
        start_day_of_year: Starting day of year for x-axis (inferred from filename if None)

    Returns:
        matplotlib.axes.Axes: The axes with the plot
    """
    # Extract EVI time series
    evi_array, valid_mask = extract_evi_timeseries(stack_path)
    n_timesteps = evi_array.shape[0]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create x-axis (days relative to labeled date)
    # The labeled date corresponds to a specific timestep, not necessarily the middle
    labeled_timestep = get_labeled_timestep(stack_path)
    days = (np.arange(n_timesteps) - labeled_timestep) * TIMESTEP_DAYS

    # Colors
    irr_color = '#2ca02c'  # Green for irrigated
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
                    ax.plot(days, ts, color=irr_color, alpha=0.15, linewidth=0.5)
            if len(non_irr_ts) > 0:
                for ts in smooth_timeseries(non_irr_ts, smooth_window):
                    ax.plot(days, ts, color=non_irr_color, alpha=0.15, linewidth=0.5)

        # Plot mean time series (bold)
        if len(irr_ts) > 0:
            irr_mean = np.nanmean(smooth_timeseries(irr_ts, smooth_window), axis=0)
            ax.plot(days, irr_mean, color=irr_color, linewidth=2.5, label='Irrigated')

        if len(non_irr_ts) > 0:
            non_irr_mean = np.nanmean(smooth_timeseries(non_irr_ts, smooth_window), axis=0)
            ax.plot(days, non_irr_mean, color=non_irr_color, linewidth=2.5, label='Non-irrigated')

        ax.legend(loc='upper right')
    else:
        # No labels - show all pixels
        all_ts = sample_pixel_timeseries(evi_array, np.ones_like(evi_array[0], dtype=bool), n_samples)

        if show_individual and len(all_ts) > 0:
            for ts in smooth_timeseries(all_ts, smooth_window):
                ax.plot(days, ts, color='gray', alpha=0.2, linewidth=0.5)

        if len(all_ts) > 0:
            all_mean = np.nanmean(smooth_timeseries(all_ts, smooth_window), axis=0)
            ax.plot(days, all_mean, color='black', linewidth=2.5, label='All pixels')

        ax.legend(loc='upper right')

    # Add vertical line at center (labeled date)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Labeled date')

    # Labels
    ax.set_xlabel('Days from labeled date')
    ax.set_ylabel('EVI')
    ax.set_ylim(-0.1, 0.7)

    # Title
    if title is None:
        stack_name = os.path.basename(stack_path).replace('_stack.tif', '')
        parts = stack_name.split('_')
        if len(parts) >= 3:
            site = parts[1]
            date = parts[2]
            title = f'EVI Time Series - Site {site}, {date}'

    ax.set_title(title)

    return ax


def plot_clustered_timeseries(stack_path, label_path=None, ax=None, figsize=(12, 6),
                               n_clusters=4, smooth_window=3, title=None):
    """
    Plot EVI time series with automatic clustering of temporal patterns.

    This is useful for exploring the data without relying on labels.

    Parameters:
        stack_path: Path to the Sentinel-2 stack
        label_path: Path to the label file (for coloring clusters by irrigation status)
        ax: Matplotlib axes to plot on
        figsize: Figure size if creating new figure
        n_clusters: Number of clusters to identify
        smooth_window: Window size for time series smoothing
        title: Plot title

    Returns:
        matplotlib.axes.Axes: The axes with the plot
    """
    from sklearn.cluster import KMeans

    # Extract EVI time series
    evi_array, valid_mask = extract_evi_timeseries(stack_path)
    n_timesteps = evi_array.shape[0]
    height, width = evi_array.shape[1], evi_array.shape[2]

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Create x-axis (days relative to labeled date)
    labeled_timestep = get_labeled_timestep(stack_path)
    days = (np.arange(n_timesteps) - labeled_timestep) * TIMESTEP_DAYS

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

        ax.plot(days, cluster_mean, color=colors[c], linewidth=2.5, label=label_str)

    # Add vertical line at center
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

    # Labels
    ax.set_xlabel('Days from labeled date')
    ax.set_ylabel('EVI')
    ax.set_ylim(-0.1, 0.7)
    ax.legend(loc='upper right', fontsize=8)

    # Title
    if title is None:
        stack_name = os.path.basename(stack_path).replace('_stack.tif', '')
        title = f'EVI Clusters - {stack_name}'

    ax.set_title(title)

    return ax


if __name__ == "__main__":
    # Test the visualization
    from sentinel2_visualization import find_labels_for_stack, load_label_mask

    features_dir = get_features_dir()
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
                plot_evi_timeseries(stack_path, label_path, ax=axes[0])
                plot_clustered_timeseries(stack_path, label_path, ax=axes[1])
                plt.tight_layout()
                plt.show()
                break
