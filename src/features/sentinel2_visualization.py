"""
Visualization functions for Sentinel-2 data with pixel-level irrigation masks.

This module provides functions to:
1. Load Sentinel-2 stack files and extract RGB imagery
2. Overlay pixel-level irrigation masks from label files
3. Create publication-quality visualizations
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

from ..utils.utils import find_project_root, get_data_root


def _get_project_root():
    """Get the project root directory."""
    return find_project_root(os.path.dirname(__file__))


def get_features_dir(version='20260107_180813'):
    """Get the features directory for a specific version."""
    return os.path.join(get_data_root(), 'features', version)


def get_irrigation_table_path():
    """Get path to the irrigation table."""
    return os.path.join(get_data_root(), 'labels/labeled_surveys/random_sample/latest_irrigation_table.csv')


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


def find_stack_for_site(site_id, year, month, day, version='20260107_180813'):
    """
    Find the stack file for a given site and date.

    Parameters:
        site_id (str): Site ID (e.g., 'id_5119273' or just '5119273')
        year, month, day (int): Date of the image
        version (str): Features version folder name

    Returns:
        str: Path to the stack file, or None if not found
    """
    features_dir = get_features_dir(version)

    # Extract numeric part of site_id
    if site_id.startswith('id_'):
        site_numeric = site_id[3:]
    else:
        site_numeric = site_id

    # Build the expected filename pattern
    date_str = f"{year}.{month:02d}.{day:02d}"
    pattern = f"*_{site_numeric}_{date_str}_stack.tif"

    matches = glob(os.path.join(features_dir, pattern))

    if len(matches) == 0:
        return None
    return matches[0]


def get_labeled_timestep(stack_path):
    """
    Find the timestep index that corresponds to the labeled date.

    The labeled date is encoded in the stack filename (e.g., '1245_5119273_2021.09.16_stack.tif').
    This function reads the metadata JSON to find which timestep contains that date.

    Parameters:
        stack_path (str): Path to the stack file

    Returns:
        int: Timestep index (0-41) corresponding to the labeled date
    """
    # Parse labeled date from filename
    stack_name = os.path.basename(stack_path).replace('_stack.tif', '')
    parts = stack_name.split('_')
    date_str = parts[2]  # e.g., '2021.09.16'
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
    # h_edges[r, c] in padded coords → boundary between original rows r-1 and r
    h_rows, h_cols = np.where(h_edges)
    for r, c in zip(h_rows, h_cols):
        # y = r - 0.5 - 0.5 = r - 1 + 0.5 (boundary at r-0.5 in original coords)
        # x spans original column c-1 (padded col c maps to original col c-1)
        y = r - 0.5
        x1, x2 = c - 1.5, c - 0.5
        segments.append(((x1, y), (x2, y)))

    # Vertical edges (left/right of pixels)
    # v_edges[r, c] in padded coords → boundary between original cols c-1 and c
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


def load_rgb_from_stack(stack_path, timestep=None):
    """
    Load RGB image from a Sentinel-2 stack file.

    The stack is stored with shape (num_bands * T, H, W) where bands are grouped:
    - Bands 1 to T: B2 (Blue) for all timesteps
    - Bands T+1 to 2T: B3 (Green) for all timesteps
    - Bands 2T+1 to 3T: B4 (Red) for all timesteps
    - etc.

    Parameters:
        stack_path (str): Path to the stack file
        timestep (int, optional): Which timestep to use (0-41).
            If None, uses the timestep corresponding to the labeled date.

    Returns:
        tuple: (rgb_array, transform, crs) where rgb_array is (H, W, 3) float [0-1]
    """
    with rasterio.open(stack_path) as src:
        n_bands = 10
        n_timesteps = src.count // n_bands

        if timestep is None:
            # Find the timestep corresponding to the labeled date
            timestep = get_labeled_timestep(stack_path)

        # Stack layout: all timesteps for band 0, then all for band 1, etc.
        # Band index in file = band_type * n_timesteps + timestep
        # RGB: B4 (Red, index 2), B3 (Green, index 1), B2 (Blue, index 0)
        red_idx = BAND_INDICES['B4'] * n_timesteps + timestep    # 2 * 42 + t
        green_idx = BAND_INDICES['B3'] * n_timesteps + timestep  # 1 * 42 + t
        blue_idx = BAND_INDICES['B2'] * n_timesteps + timestep   # 0 * 42 + t

        # Read bands (1-indexed in rasterio)
        red = src.read(red_idx + 1).astype(np.float32)
        green = src.read(green_idx + 1).astype(np.float32)
        blue = src.read(blue_idx + 1).astype(np.float32)

        transform = src.transform
        crs = src.crs

    # Handle nodata
    nodata_mask = (red == -9999) | (green == -9999) | (blue == -9999)

    # Stack into RGB and normalize (same method as download_sentinel2.py)
    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.clip(rgb / 3000.0, 0, 1)
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


def plot_sentinel2_with_mask(stack_path, label_path=None, operator=None,
                              timestep=None, ax=None, figsize=(10, 10),
                              title=None, show_legend=True, linewidth=1.5):
    """
    Plot Sentinel-2 RGB with irrigation mask outlines (pixel-edge boundaries).

    Parameters:
        stack_path (str): Path to the stack file
        label_path (str, optional): Path to the label file. If None, will search.
        operator (str, optional): Operator initials for label file
        timestep (int, optional): Which timestep to use for RGB.
            If None, uses the timestep corresponding to the labeled date.
        ax (matplotlib.axes.Axes, optional): Axes to plot on
        figsize (tuple): Figure size if creating new figure
        title (str, optional): Plot title
        show_legend (bool): Whether to show irrigation type legend
        linewidth (float): Width of outline lines

    Returns:
        matplotlib.axes.Axes: The axes with the plot
    """
    # Load RGB at the correct timestep (labeled date)
    rgb, _, _ = load_rgb_from_stack(stack_path, timestep)

    # Find label file if not provided
    if label_path is None:
        label_paths = find_labels_for_stack(stack_path, operator)
        if len(label_paths) == 0:
            print(f"Warning: No label file found for {stack_path}")
            label_path = None
        else:
            label_path = label_paths[0]
            if operator is None:
                # Extract operator from filename
                operator = os.path.basename(label_path).split('_')[-2]

    # Create figure if needed
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Plot RGB
    ax.imshow(rgb)

    # Draw outlines if label available
    if label_path is not None and os.path.exists(label_path):
        categorical = load_label_mask(label_path, 'categorical')

        # Draw pixel boundaries for each irrigation type
        legend_elements = []
        for irr_type in sorted(IRRIGATION_COLORS.keys()):
            if irr_type == 0:
                continue  # Skip "no irrigation"

            mask = (categorical == irr_type).astype(np.uint8)
            if not mask.any():
                continue

            color = IRRIGATION_COLORS[irr_type][:3]

            # Get pixel boundary segments
            segments = trace_pixel_boundaries(mask)

            if segments:
                # Use LineCollection for efficient rendering
                lc = LineCollection(segments, colors=[color], linewidths=linewidth)
                ax.add_collection(lc)

            # Add to legend
            if show_legend:
                from matplotlib.lines import Line2D
                legend_elements.append(
                    Line2D([0], [0], color=color, linewidth=linewidth,
                           label=IRRIGATION_LABELS[irr_type])
                )

        if show_legend and legend_elements:
            ax.legend(handles=legend_elements, loc='upper right',
                     title='Irrigation Type')

    # Set title
    if title is None:
        stack_name = os.path.basename(stack_path).replace('_stack.tif', '')
        parts = stack_name.split('_')
        if len(parts) >= 3:
            site = parts[1]
            date = parts[2]
            title = f"Site {site}, {date}"
            if operator:
                title += f" (Labeler: {operator})"

    ax.set_title(title)
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')

    return ax


def find_matching_stack_for_screenshot(survey, internal_id, month, day, year,
                                        version='20260107_180813'):
    """
    Find the Sentinel-2 stack file that matches a GEE screenshot.

    Parameters:
        survey (str): Survey ID (e.g., '201-225')
        internal_id (int): Internal ID within the survey
        month, day, year (int): Date of the screenshot
        version (str): Features version

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

    return find_stack_for_site(site_id, year, month, day, version)


if __name__ == "__main__":
    # Test the visualization
    import matplotlib.pyplot as plt

    features_dir = get_features_dir()
    stack_files = glob(os.path.join(features_dir, '*_stack.tif'))

    if stack_files:
        # Find a stack with labels
        for stack_path in stack_files[:20]:
            labels = find_labels_for_stack(stack_path)
            if labels:
                print(f"Testing with: {os.path.basename(stack_path)}")
                print(f"Labels: {[os.path.basename(l) for l in labels]}")

                fig, axes = plt.subplots(1, 2, figsize=(16, 8))
                plot_sentinel2_with_mask(stack_path, ax=axes[0])
                plt.tight_layout()
                plt.show()
                break
