"""
Visualization functions for satellite data with pixel-level irrigation masks.

This module provides functions to:
1. Load satellite stack files (Sentinel-2 or PlanetScope) and extract RGB imagery
2. Overlay pixel-level irrigation masks from label files
3. Create publication-quality visualizations

Supports both Sentinel-2 (10 bands, 42 timesteps) and PlanetScope (4 bands, 42 timesteps).
Use the 'sensor' parameter to switch between sensors (default: 'sentinel2').
"""

import os
import json
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from glob import glob
from datetime import datetime

from ...utils.utils import find_project_root, get_data_root


# Sensor configurations (single source of truth)
SENSOR_CONFIG = {
    'sentinel2': {
        'n_bands': 10,
        'n_timesteps': 42,
        'band_names': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'],
        'band_indices': {
            'blue': 0,   # B2
            'green': 1,  # B3
            'red': 2,    # B4
            'nir': 6,    # B8
            'B2': 0, 'B3': 1, 'B4': 2, 'B5': 3, 'B6': 4,
            'B7': 5, 'B8': 6, 'B8A': 7, 'B11': 8, 'B12': 9,
        },
        'default_version': '20260107_180813',
        'data_dir': 'features/sentinel2',
        'normalization': 3000.0,
    },
    'planetscope': {
        'n_bands': 4,
        'n_timesteps': 42,
        'band_names': ['Blue', 'Green', 'Red', 'NIR'],
        'band_indices': {
            'blue': 0,
            'green': 1,
            'red': 2,
            'nir': 3,
        },
        'default_version': '20260127_161535_SR',
        'data_dir': 'features/planetscope',
        'normalization': 3000.0,
    }
}

# Irrigation type colors
IRRIGATION_COLORS = {
    0: (0.9, 0.9, 0.9, 0),      # No irrigation - transparent
    1: (0.2, 0.6, 0.2, 0.6),    # Small-scale - green
    2: (0.4, 0.8, 0.4, 0.6),    # Tree crop - light green
    3: (0.8, 0.2, 0.2, 0.6),    # Industrial - red
    4: (0.2, 0.2, 0.8, 0.6),    # Lawn - blue
    5: (0.6, 0.4, 0.8, 0.6),    # Covered - purple
}

IRRIGATION_LABELS = {
    0: 'No irrigation',
    1: 'Small-scale',
    2: 'Tree crop',
    3: 'Industrial',
    4: 'Lawn',
    5: 'Covered',
}

# Labeler colors for multi-labeler comparison (RGB tuples for matplotlib)
LABELER_COLORS = {
    'AB': (0.8, 0.2, 0.2),    # Red
    'DSB': (0.2, 0.6, 0.8),   # Cyan
    'JL': (0.2, 0.8, 0.2),    # Green
    'KL': (0.8, 0.6, 0.2),    # Orange
    'MV': (0.6, 0.2, 0.8),    # Purple
    'PS': (0.8, 0.8, 0.2),    # Yellow
}

# Hex color versions (for APIs that prefer hex strings)
LABELER_COLORS_HEX = {
    'AB': '#cc3333',   # Red
    'DSB': '#33a0cc',  # Cyan
    'JL': '#33cc33',   # Green
    'KL': '#cc9933',   # Orange
    'MV': '#9933cc',   # Purple
    'PS': '#cccc33',   # Yellow
}


def _get_project_root():
    """Get the project root directory."""
    return find_project_root(os.path.dirname(__file__))


def get_features_dir(version=None, sensor='sentinel2'):
    """
    Get the features directory for a specific version and sensor.

    Parameters:
        version (str, optional): Version folder name. If None, uses sensor default.
        sensor (str): Sensor type ('sentinel2' or 'planetscope')

    Returns:
        str: Path to the features directory
    """
    config = SENSOR_CONFIG[sensor]
    if version is None:
        version = config['default_version']
    return os.path.join(get_data_root(), config['data_dir'], version)


def get_irrigation_table_path():
    """Get path to the irrigation table."""
    return os.path.join(get_data_root(), 'labels/labeled_surveys/random_sample/latest_irrigation_table.csv')


def find_stack_for_site(site_id, year, month, day, version=None, sensor='sentinel2'):
    """
    Find the stack file for a given site and date.

    Parameters:
        site_id (str): Site ID (e.g., 'id_5119273' or just '5119273')
        year, month, day (int): Date of the image
        version (str, optional): Features version folder name. If None, uses sensor default.
        sensor (str): Sensor type ('sentinel2' or 'planetscope')

    Returns:
        str: Path to the stack file, or None if not found
    """
    features_dir = get_features_dir(version, sensor)

    # Extract numeric part of site_id
    if site_id.startswith('id_'):
        site_numeric = site_id[3:]
    else:
        site_numeric = site_id

    # Build the expected filename pattern
    date_str = f"{year}.{month:02d}.{day:02d}"
    pattern = f"{site_numeric}_{date_str}_stack.tif"

    matches = glob(os.path.join(features_dir, pattern))

    if len(matches) == 0:
        return None
    return matches[0]


def get_labeled_timestep(stack_path, sensor='sentinel2'):
    """
    Find the timestep index that corresponds to the labeled date.

    The labeled date is encoded in the stack filename (e.g., '5119273_2021.09.16_stack.tif').
    This function reads the metadata JSON to find which timestep contains that date.

    Parameters:
        stack_path (str): Path to the stack file
        sensor (str): Sensor type ('sentinel2' or 'planetscope')

    Returns:
        int: Timestep index (0-41) corresponding to the labeled date
    """
    # Parse labeled date from filename
    stack_name = os.path.basename(stack_path).replace('_stack.tif', '')
    parts = stack_name.split('_')
    date_str = parts[1]  # e.g., '2021.09.16'
    labeled_date = datetime.strptime(date_str, '%Y.%m.%d')

    # Try to load metadata
    features_dir = os.path.dirname(stack_path)
    meta_path = os.path.join(features_dir, f'{stack_name}_metadata.json')

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

        # Find timestep containing the labeled date
        for i, window in enumerate(meta.get('windows', [])):
            start = datetime.strptime(window['date_range'][0], '%Y-%m-%d')
            end = datetime.strptime(window['date_range'][1], '%Y-%m-%d')
            if start <= labeled_date < end:
                return i

    # Fallback: assume middle timestep (legacy behavior)
    return 21


def trace_pixel_boundaries(mask):
    """
    Trace the boundaries of labeled pixels as line segments at pixel edges.

    Unlike skimage.measure.find_contours which cuts diagonally across corners,
    this function produces proper rectangular outlines with 90-degree corners.

    Parameters:
        mask (numpy.ndarray): Binary mask (2D array)

    Returns:
        list: List of ((x1,y1), (x2,y2)) line segments defining boundaries
    """
    # Pad mask to handle edge cases
    padded = np.pad(mask, 1, mode='constant', constant_values=0)

    # Find all boundary edges
    # Horizontal edges: where vertically adjacent pixels differ
    h_edges = padded[:-1, :] != padded[1:, :]
    # Vertical edges: where horizontally adjacent pixels differ
    v_edges = padded[:, :-1] != padded[:, 1:]

    segments = []

    # Horizontal edges (top/bottom of pixels)
    # h_edges[r, c] in padded coords -> boundary between original rows r-1 and r
    h_rows, h_cols = np.where(h_edges)
    for r, c in zip(h_rows, h_cols):
        # y = r - 0.5 - 0.5 = r - 1 + 0.5 (boundary at r-0.5 in original coords)
        # x spans original column c-1 (padded col c maps to original col c-1)
        y = r - 0.5
        x1, x2 = c - 1.5, c - 0.5
        segments.append(((x1, y), (x2, y)))

    # Vertical edges (left/right of pixels)
    # v_edges[r, c] in padded coords -> boundary between original cols c-1 and c
    v_rows, v_cols = np.where(v_edges)
    for r, c in zip(v_rows, v_cols):
        # x = c - 0.5 - 0.5 = c - 1 + 0.5 (boundary at c-0.5 in original coords)
        # y spans original row r-1 (padded row r maps to original row r-1)
        x = c - 0.5
        y1, y2 = r - 1.5, r - 0.5
        segments.append(((x, y1), (x, y2)))

    return segments


def find_labels_for_stack(stack_path, operator=None):
    """
    Find label files for a given stack file.

    Parameters:
        stack_path (str): Path to the stack file
        operator (str, optional): Specific operator initials. If None, returns all.

    Returns:
        list: Paths to matching label files
    """
    stack_dir = os.path.dirname(stack_path)
    stack_name = os.path.basename(stack_path).replace('_stack.tif', '')

    if operator:
        pattern = f"{stack_name}_{operator}_labels.tif"
        label_path = os.path.join(stack_dir, pattern)
        return [label_path] if os.path.exists(label_path) else []
    else:
        pattern = f"{stack_name}_*_labels.tif"
        return glob(os.path.join(stack_dir, pattern))


def load_rgb_from_stack(stack_path, timestep=None, sensor='sentinel2'):
    """
    Load RGB image from a satellite stack file.

    The stack is stored with shape (num_bands * T, H, W) where bands are grouped:
    - For Sentinel-2: Bands 1 to T: B2 (Blue), T+1 to 2T: B3 (Green), etc.
    - For PlanetScope: Bands 1 to T: Blue, T+1 to 2T: Green, 2T+1 to 3T: Red, 3T+1 to 4T: NIR

    Parameters:
        stack_path (str): Path to the stack file
        timestep (int, optional): Which timestep to use (0-41).
            If None, uses the timestep corresponding to the labeled date.
        sensor (str): Sensor type ('sentinel2' or 'planetscope')

    Returns:
        tuple: (rgb_array, transform, crs) where rgb_array is (H, W, 3) float [0-1]
    """
    config = SENSOR_CONFIG[sensor]
    band_indices = config['band_indices']
    normalization = config['normalization']

    with rasterio.open(stack_path) as src:
        n_bands = config['n_bands']
        n_timesteps = src.count // n_bands

        if timestep is None:
            # Find the timestep corresponding to the labeled date
            timestep = get_labeled_timestep(stack_path, sensor)

        # Stack layout: all timesteps for band 0, then all for band 1, etc.
        # Band index in file = band_type * n_timesteps + timestep
        red_idx = band_indices['red'] * n_timesteps + timestep
        green_idx = band_indices['green'] * n_timesteps + timestep
        blue_idx = band_indices['blue'] * n_timesteps + timestep

        # Read bands (1-indexed in rasterio)
        red = src.read(red_idx + 1).astype(np.float32)
        green = src.read(green_idx + 1).astype(np.float32)
        blue = src.read(blue_idx + 1).astype(np.float32)

        transform = src.transform
        crs = src.crs

    # Handle nodata (both -9999 and 0 are used depending on the data source)
    nodata_mask = (red == -9999) | (green == -9999) | (blue == -9999)
    nodata_mask |= (red == 0) & (green == 0) & (blue == 0)

    # Stack into RGB and normalize
    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.clip(rgb / normalization, 0, 1)
    rgb[nodata_mask] = 0  # Set nodata to black

    return rgb, transform, crs


def load_label_mask(label_path, band='binary'):
    """
    Load a specific band from a label file.

    Parameters:
        label_path (str): Path to the label file
        band (str): Which band to load:
            'categorical' (0), 'binary' (1), 'certainty' (7), 'coverage' (8)

    Returns:
        numpy.ndarray: The requested band
    """
    band_map = {
        'categorical': 0,
        'binary': 1,
        'unclear_agriculture': 2,
        'slightly_green': 3,
        'uneven': 4,
        'may_be_natural': 5,
        'may_be_fishpond': 6,
        'certainty': 7,
        'coverage': 8,
    }

    band_idx = band_map.get(band, band) if isinstance(band, str) else band

    with rasterio.open(label_path) as src:
        data = src.read(band_idx + 1)  # 1-indexed

    return data


def plot_satellite_with_mask(stack_path, timestep=None, ax=None, figsize=(10, 10),
                              title=None, show_legend=True, linewidth=1.5,
                              sensor='sentinel2'):
    """
    Plot satellite RGB with ALL labelers' irrigation masks overlaid.

    Each labeler's mask is shown in a distinct color, allowing comparison
    of how different labelers annotated the same image.

    Parameters:
        stack_path (str): Path to the stack file
        timestep (int, optional): Which timestep to use for RGB.
            If None, uses the timestep corresponding to the labeled date.
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        figsize (tuple): Figure size if creating new figure
        title (str, optional): Plot title
        show_legend (bool): Whether to show labeler legend
        linewidth (float): Width of outline lines
        sensor (str): Sensor type ('sentinel2' or 'planetscope')

    Returns:
        matplotlib.axes.Axes: The axes with the plot
    """
    # Load RGB at the correct timestep (labeled date)
    rgb, _, _ = load_rgb_from_stack(stack_path, timestep, sensor)

    # Find all label files for this stack
    label_paths = find_labels_for_stack(stack_path, operator=None)

    if len(label_paths) == 0:
        print(f"Warning: No label files found for {stack_path}")

    # Create figure if needed
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Plot RGB
    ax.imshow(rgb)

    # Draw outlines for each labeler
    legend_elements = []
    for label_path in sorted(label_paths):
        # Extract operator from filename: {site}_{date}_{operator}_labels.tif
        operator = os.path.basename(label_path).split('_')[-2]

        # Get color for this labeler
        color = LABELER_COLORS.get(operator, (0.5, 0.5, 0.5))  # Gray default

        # Load binary mask (any irrigation)
        binary_mask = load_label_mask(label_path, 'binary')

        if not binary_mask.any():
            continue  # No irrigation marked by this labeler

        # Get pixel boundary segments
        segments = trace_pixel_boundaries(binary_mask)

        if segments:
            lc = LineCollection(segments, colors=[color], linewidths=linewidth)
            ax.add_collection(lc)

            # Add to legend
            from matplotlib.lines import Line2D
            legend_elements.append(
                Line2D([0], [0], color=color, linewidth=linewidth, label=operator)
            )

    if show_legend and legend_elements:
        ax.legend(handles=legend_elements, loc='upper right', title='Labeler')

    # Set title
    if title is None:
        stack_name = os.path.basename(stack_path).replace('_stack.tif', '')
        parts = stack_name.split('_')
        sensor_name = 'PlanetScope' if sensor == 'planetscope' else 'Sentinel-2'
        if len(parts) >= 2:
            site = parts[0]
            date = parts[1]
            title = f"{sensor_name} - Site {site}, {date} (All Labelers)"

    ax.set_title(title)
    ax.axis('off')

    return ax


def find_matching_stack_for_screenshot(survey, internal_id, month, day, year,
                                        version=None, sensor='sentinel2'):
    """
    Find the satellite stack file that matches a GEP screenshot.

    Parameters:
        survey (str): Survey ID (e.g., '201-225')
        internal_id (int): Internal ID within the survey
        month, day, year (int): Date of the screenshot
        version (str, optional): Features version. If None, uses sensor default.
        sensor (str): Sensor type ('sentinel2' or 'planetscope')

    Returns:
        str or None: Path to matching stack file
    """
    # Load irrigation table to find site_id
    irrigation_df = pd.read_csv(get_irrigation_table_path())

    # Filter by survey
    irrigation_df = irrigation_df[irrigation_df['source_file'].str.contains(survey, na=False)]

    # Filter by internal_id and date
    matches = irrigation_df[
        (irrigation_df['internal_id'] == internal_id) &
        (irrigation_df['year'] == year) &
        (irrigation_df['month'] == month) &
        (irrigation_df['day'] == day)
    ]

    if len(matches) == 0:
        # Try without exact date match
        matches = irrigation_df[irrigation_df['internal_id'] == internal_id]
        if len(matches) == 0:
            return None

    site_id = matches.iloc[0]['site_id']

    return find_stack_for_site(site_id, year, month, day, version, sensor)


if __name__ == "__main__":
    # Test the visualization
    import matplotlib.pyplot as plt

    # Test Sentinel-2
    print("Testing Sentinel-2...")
    features_dir = get_features_dir(sensor='sentinel2')
    stack_files = glob(os.path.join(features_dir, '*_stack.tif'))

    if stack_files:
        # Find a stack with labels
        for stack_path in stack_files[:20]:
            labels = find_labels_for_stack(stack_path)
            if labels:
                print(f"Testing with: {os.path.basename(stack_path)}")
                print(f"Labels: {[os.path.basename(l) for l in labels]}")

                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                plot_satellite_with_mask(stack_path, ax=axes[0], sensor='sentinel2')
                axes[0].set_title('Sentinel-2')
                plt.tight_layout()
                plt.show()
                break

    # Test PlanetScope if available
    print("\nTesting PlanetScope...")
    planet_dir = get_features_dir(sensor='planetscope')
    if os.path.exists(planet_dir):
        planet_files = glob(os.path.join(planet_dir, '*_stack.tif'))
        if planet_files:
            stack_path = planet_files[0]
            print(f"Testing with: {os.path.basename(stack_path)}")
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_satellite_with_mask(stack_path, ax=ax, sensor='planetscope')
            plt.show()
