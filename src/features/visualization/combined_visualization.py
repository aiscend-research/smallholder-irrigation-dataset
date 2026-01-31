"""
Combined visualization module for multi-source satellite comparisons.

This module orchestrates comparisons between:
- GEE screenshots (with polygon overlays)
- Sentinel-2 RGB imagery (with pixel-level masks)
- PlanetScope RGB imagery (with pixel-level masks)
- EVI time series from both sensors

Uses existing visualization functions - no duplication of core visualization logic.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from ...utils.utils import get_data_root
from .satellite_visualization import (
    SENSOR_CONFIG, get_features_dir, find_stack_for_site,
    find_labels_for_stack, plot_satellite_with_all_masks, get_irrigation_table_path
)
from .evi_timeseries_visualization import plot_evi_timeseries
from .gee_screenshot_visualization import (
    plot_screenshot_with_polygons, list_available_screenshots, get_screenshot_dir
)


def find_all_available_sources(site_id, year, month, day,
                                sentinel_version=None, planet_version=None):
    """
    Find all available data sources for a given site and date.

    Parameters:
        site_id (str): Site ID (e.g., 'id_5119273' or '5119273')
        year, month, day (int): Date of the image
        sentinel_version (str, optional): Sentinel-2 version folder
        planet_version (str, optional): PlanetScope version folder

    Returns:
        dict: Dictionary with keys 'gee_screenshot', 'sentinel_stack', 'planet_stack',
              each containing the path or None if not available
    """
    sources = {
        'gee_screenshot': None,
        'sentinel_stack': None,
        'planet_stack': None,
        'sentinel_labels': [],
        'planet_labels': [],
    }

    # Look for Sentinel-2 stack
    sentinel_stack = find_stack_for_site(site_id, year, month, day, sentinel_version, 'sentinel2')
    if sentinel_stack and os.path.exists(sentinel_stack):
        sources['sentinel_stack'] = sentinel_stack
        sources['sentinel_labels'] = find_labels_for_stack(sentinel_stack)

    # Look for PlanetScope stack
    planet_stack = find_stack_for_site(site_id, year, month, day, planet_version, 'planetscope')
    if planet_stack and os.path.exists(planet_stack):
        sources['planet_stack'] = planet_stack
        sources['planet_labels'] = find_labels_for_stack(planet_stack)

    # Look for GEE screenshot - need to match via irrigation table
    try:
        irrigation_df = pd.read_csv(get_irrigation_table_path())

        # Find matching row
        if site_id.startswith('id_'):
            site_match = site_id
        else:
            site_match = f'id_{site_id}'

        matches = irrigation_df[
            (irrigation_df['site_id'] == site_match) &
            (irrigation_df['year'] == year) &
            (irrigation_df['month'] == month) &
            (irrigation_df['day'] == day)
        ]

        if len(matches) > 0:
            row = matches.iloc[0]
            # Try to find screenshot by survey and internal_id
            import re
            source_file = row.get('source_file', '')
            if pd.notna(source_file):
                survey_match = re.search(r'(\d+-\d+)', str(source_file))
                if survey_match:
                    survey = survey_match.group(1)
                    internal_id = row.get('internal_id')
                    if pd.notna(internal_id):
                        # Look for matching screenshot
                        screenshots = list_available_screenshots()
                        for ss in screenshots:
                            if (ss['survey'] == survey and
                                ss['internal_id'] == int(internal_id) and
                                ss['year'] == year and
                                ss['month'] == month and
                                ss['day'] == day):
                                sources['gee_screenshot'] = ss['path']
                                sources['gee_info'] = ss
                                break
    except Exception as e:
        print(f"Warning: Could not search for GEE screenshot: {e}")

    return sources


def plot_combined_comparison(site_id, year, month, day,
                              sentinel_version=None, planet_version=None,
                              operator=None, figsize=(16, 12),
                              show_evi=True, title=None):
    """
    Create a combined comparison figure showing all available data sources.

    Layout adapts based on available data:
    - Row 1: [GEE Screenshot] [Sentinel-2 RGB] [PlanetScope RGB]
    - Row 2: [Sentinel-2 EVI time series - full width] (if show_evi)
    - Row 3: [PlanetScope EVI time series - full width] (if show_evi)

    Parameters:
        site_id (str): Site ID (e.g., 'id_5119273' or '5119273')
        year, month, day (int): Date of the image
        sentinel_version (str, optional): Sentinel-2 version folder
        planet_version (str, optional): PlanetScope version folder
        operator (str, optional): Specific labeler initials to show
        figsize (tuple): Figure size
        show_evi (bool): Whether to include EVI time series panels
        title (str, optional): Overall figure title

    Returns:
        matplotlib.figure.Figure: The figure with all panels
    """
    # Find all available sources
    sources = find_all_available_sources(site_id, year, month, day,
                                          sentinel_version, planet_version)

    # Count how many image sources we have
    n_images = sum([
        sources['gee_screenshot'] is not None,
        sources['sentinel_stack'] is not None,
        sources['planet_stack'] is not None,
    ])

    if n_images == 0:
        raise ValueError(f"No data sources found for site {site_id} on {year}-{month:02d}-{day:02d}")

    # Count EVI panels needed
    n_evi = 0
    if show_evi:
        if sources['sentinel_stack'] and sources['sentinel_labels']:
            n_evi += 1
        if sources['planet_stack'] and sources['planet_labels']:
            n_evi += 1

    # Create figure with GridSpec
    n_rows = 1 + n_evi
    fig = plt.figure(figsize=figsize)

    if n_evi > 0:
        # Use GridSpec for flexible layout
        height_ratios = [2] + [1] * n_evi  # Image row is taller
        gs = GridSpec(n_rows, 3, figure=fig, height_ratios=height_ratios,
                      hspace=0.3, wspace=0.2)
    else:
        gs = GridSpec(1, 3, figure=fig, wspace=0.2)

    # Track which column to use for each image type
    col_idx = 0

    # Row 1: Image panels
    # GEE Screenshot
    if sources['gee_screenshot'] is not None:
        ax_gee = fig.add_subplot(gs[0, col_idx])
        try:
            info = sources.get('gee_info', {})
            plot_screenshot_with_polygons(
                screenshot_path=sources['gee_screenshot'],
                survey=info.get('survey'),
                internal_id=info.get('internal_id'),
                month=month, day=day, year=year,
                ax=ax_gee, show_legend=True
            )
            ax_gee.set_title('GEE Screenshot')
        except Exception as e:
            ax_gee.text(0.5, 0.5, f'Error loading GEE screenshot:\n{e}',
                       ha='center', va='center', transform=ax_gee.transAxes)
            ax_gee.set_title('GEE Screenshot (Error)')
        col_idx += 1

    # Sentinel-2 RGB
    if sources['sentinel_stack'] is not None:
        ax_s2 = fig.add_subplot(gs[0, col_idx])
        label_path = sources['sentinel_labels'][0] if sources['sentinel_labels'] else None
        try:
            plot_satellite_with_all_masks(
                sources['sentinel_stack'], 
                operator=operator, ax=ax_s2, sensor='sentinel2',
                show_legend=True
            )
        except Exception as e:
            ax_s2.text(0.5, 0.5, f'Error loading Sentinel-2:\n{e}',
                      ha='center', va='center', transform=ax_s2.transAxes)
            ax_s2.set_title('Sentinel-2 (Error)')
        col_idx += 1

    # PlanetScope RGB
    if sources['planet_stack'] is not None:
        ax_ps = fig.add_subplot(gs[0, col_idx])
        label_path = sources['planet_labels'][0] if sources['planet_labels'] else None
        try:
            plot_satellite_with_all_masks(
                sources['planet_stack'],
                operator=operator, ax=ax_ps, sensor='planetscope',
                show_legend=True
            )
        except Exception as e:
            ax_ps.text(0.5, 0.5, f'Error loading PlanetScope:\n{e}',
                      ha='center', va='center', transform=ax_ps.transAxes)
            ax_ps.set_title('PlanetScope (Error)')

    # EVI time series panels (full width)
    evi_row = 1
    if show_evi and sources['sentinel_stack'] and sources['sentinel_labels']:
        ax_evi_s2 = fig.add_subplot(gs[evi_row, :])
        try:
            plot_evi_timeseries(
                sources['sentinel_stack'],
                sources['sentinel_labels'][0],
                ax=ax_evi_s2, sensor='sentinel2'
            )
        except Exception as e:
            ax_evi_s2.text(0.5, 0.5, f'Error loading Sentinel-2 EVI:\n{e}',
                          ha='center', va='center', transform=ax_evi_s2.transAxes)
        evi_row += 1

    if show_evi and sources['planet_stack'] and sources['planet_labels']:
        ax_evi_ps = fig.add_subplot(gs[evi_row, :])
        try:
            plot_evi_timeseries(
                sources['planet_stack'],
                sources['planet_labels'][0],
                ax=ax_evi_ps, sensor='planetscope'
            )
        except Exception as e:
            ax_evi_ps.text(0.5, 0.5, f'Error loading PlanetScope EVI:\n{e}',
                          ha='center', va='center', transform=ax_evi_ps.transAxes)

    # Set overall title
    if title is None:
        site_str = site_id if site_id.startswith('id_') else f'id_{site_id}'
        title = f'Site {site_str} | {year}-{month:02d}-{day:02d}'
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


def plot_sensor_comparison(site_id, year, month, day,
                            sentinel_version=None, planet_version=None,
                            operator=None, figsize=(14, 6)):
    """
    Create a side-by-side comparison of Sentinel-2 and PlanetScope RGB.

    Simpler than plot_combined_comparison - just shows the two satellite sources.

    Parameters:
        site_id (str): Site ID
        year, month, day (int): Date of the image
        sentinel_version, planet_version (str, optional): Version folders
        operator (str, optional): Labeler initials
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The comparison figure
    """
    sources = find_all_available_sources(site_id, year, month, day,
                                          sentinel_version, planet_version)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Sentinel-2
    if sources['sentinel_stack']:
        label_path = sources['sentinel_labels'][0] if sources['sentinel_labels'] else None
        plot_satellite_with_all_masks(
            sources['sentinel_stack'], 
            operator=operator, ax=axes[0], sensor='sentinel2'
        )
    else:
        axes[0].text(0.5, 0.5, 'Sentinel-2 not available',
                    ha='center', va='center', transform=axes[0].transAxes)
        axes[0].set_title('Sentinel-2')

    # PlanetScope
    if sources['planet_stack']:
        label_path = sources['planet_labels'][0] if sources['planet_labels'] else None
        plot_satellite_with_all_masks(
            sources['planet_stack'], 
            operator=operator, ax=axes[1], sensor='planetscope'
        )
    else:
        axes[1].text(0.5, 0.5, 'PlanetScope not available',
                    ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('PlanetScope')

    site_str = site_id if site_id.startswith('id_') else f'id_{site_id}'
    fig.suptitle(f'Sensor Comparison: {site_str} | {year}-{month:02d}-{day:02d}',
                fontsize=12, fontweight='bold')
    plt.tight_layout()

    return fig


def find_sites_with_both_sensors(sentinel_version=None, planet_version=None, limit=100):
    """
    Find sites that have both Sentinel-2 and PlanetScope data available.

    Parameters:
        sentinel_version (str, optional): Sentinel-2 version folder
        planet_version (str, optional): PlanetScope version folder
        limit (int): Maximum number of sites to return

    Returns:
        list: List of dicts with site_id, year, month, day for each matching site
    """
    from glob import glob

    # Get file lists from both directories
    s2_dir = get_features_dir(sentinel_version, 'sentinel2')
    ps_dir = get_features_dir(planet_version, 'planetscope')

    if not os.path.exists(s2_dir) or not os.path.exists(ps_dir):

        return []

    # Parse filenames to get site-date combinations
    def parse_stack_files(directory):
        files = glob(os.path.join(directory, '*_stack.tif'))
        parsed = set()
        for f in files:
            name = os.path.basename(f).replace('_stack.tif', '')
            parts = name.split('_')
            if len(parts) >= 2:
                site = parts[0]
                date = parts[1]
                parsed.add((site, date))
        return parsed

    s2_sites = parse_stack_files(s2_dir)
    ps_sites = parse_stack_files(ps_dir)

    # Find intersection
    common = s2_sites & ps_sites

    # Convert to list of dicts
    results = []
    for site, date_str in list(common)[:limit]:
        try:
            date_parts = date_str.split('.')
            results.append({
                'site_id': f'id_{site}',
                'year': int(date_parts[0]),
                'month': int(date_parts[1]),
                'day': int(date_parts[2]),
            })
        except (ValueError, IndexError):
            continue

    return results


if __name__ == "__main__":
    # Test the combined visualization
    print("Finding sites with both Sentinel-2 and PlanetScope data...")
    common_sites = find_sites_with_both_sensors(limit=5)

    if common_sites:
        print(f"Found {len(common_sites)} sites with both sensors")
        site = common_sites[0]
        print(f"Testing with: {site}")

        fig = plot_combined_comparison(
            site['site_id'], site['year'], site['month'], site['day'],
            show_evi=True
        )
        plt.show()
    else:
        print("No sites found with both sensors. Testing with single sensor...")
        # Test with just Sentinel-2
        from glob import glob
        s2_dir = get_features_dir(sensor='sentinel2')
        if os.path.exists(s2_dir):
            stacks = glob(os.path.join(s2_dir, '*_stack.tif'))[:1]
            if stacks:
                name = os.path.basename(stacks[0]).replace('_stack.tif', '')
                parts = name.split('_')
                if len(parts) >= 2:
                    site_id = f'id_{parts[0]}'
                    date_parts = parts[1].split('.')
                    fig = plot_combined_comparison(
                        site_id, int(date_parts[0]), int(date_parts[1]), int(date_parts[2]),
                        show_evi=True
                    )
                    plt.show()
