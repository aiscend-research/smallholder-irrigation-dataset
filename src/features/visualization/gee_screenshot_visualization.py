"""
Visualization functions for Google Earth Engine screenshots with polygon overlays.

This module provides functions to:
1. Load GEE screenshots from the data/labels/GEE_screenshots directory
2. Overlay irrigation polygons from all labelers with different colors
3. Query specific images or select random ones for visualization
"""

import os
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from PIL import Image
from glob import glob
from shapely.geometry import box
from pyproj import Transformer

from ...utils.utils import find_project_root
from .satellite_visualization import LABELER_COLORS_HEX


def _get_project_root():
    """Get the project root directory."""
    return find_project_root(os.path.dirname(__file__))


# Directory containing GEE screenshots
def get_screenshot_dir():
    return os.path.join(_get_project_root(), 'data/labels/GEE_screenshots')


def get_polygons_path():
    return os.path.join(_get_project_root(), 'data/labels/labeled_surveys/random_sample/latest_polygons.geojson')


def get_irrigation_table_path():
    return os.path.join(_get_project_root(), 'data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv')


def parse_screenshot_filename(filename):
    """
    Parse a GEE screenshot filename to extract survey, internal_id, and date.

    Expected formats:
    - 201-225_13_07-31-23.png -> survey='201-225', internal_id=13, month=7, day=31, year=2023
    - 201-225_16-06-26-19.png -> survey='201-225', internal_id=16, month=6, day=26, year=2019

    Parameters:
        filename (str): The filename (with or without path)

    Returns:
        dict with keys: survey, internal_id, month, day, year
    """
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]

    # Try pattern: survey_id_MM-DD-YY
    match = re.match(r'(\d+-\d+)_(\d+)_(\d+)-(\d+)-(\d+)', name)
    if match:
        survey, internal_id, month, day, year = match.groups()
        year = int(year)
        # Convert 2-digit year to 4-digit
        if year < 50:
            year = 2000 + year
        else:
            year = 1900 + year
        return {
            'survey': survey,
            'internal_id': int(internal_id),
            'month': int(month),
            'day': int(day),
            'year': year
        }

    # Try pattern: survey_id-MM-DD-YY (underscore missing before date)
    match = re.match(r'(\d+-\d+)_(\d+)-(\d+)-(\d+)-(\d+)', name)
    if match:
        survey, internal_id, month, day, year = match.groups()
        year = int(year)
        if year < 50:
            year = 2000 + year
        else:
            year = 1900 + year
        return {
            'survey': survey,
            'internal_id': int(internal_id),
            'month': int(month),
            'day': int(day),
            'year': year
        }

    raise ValueError(f"Could not parse filename: {filename}")


def list_available_screenshots():
    """
    List all available GEE screenshots.

    Returns:
        list of dicts with screenshot info (path, survey, internal_id, date)
    """
    screenshot_dir = get_screenshot_dir()
    screenshots = []

    for f in glob(os.path.join(screenshot_dir, '*.png')):
        try:
            info = parse_screenshot_filename(f)
            info['path'] = f
            screenshots.append(info)
        except ValueError as e:
            print(f"Warning: {e}")

    return screenshots


def get_image_bounds(x, y, size_m=1000):
    """
    Get the bounding box for an image centered at (x, y) in WGS84.

    Parameters:
        x (float): Longitude (WGS84)
        y (float): Latitude (WGS84)
        size_m (float): Size of the image in meters (default 1000m = 1km)

    Returns:
        tuple: (minx, miny, maxx, maxy) in WGS84
    """
    # Convert to UTM to get accurate meter-based bounds
    # Determine UTM zone from longitude
    utm_zone = int((x + 180) / 6) + 1
    hemisphere = 'north' if y >= 0 else 'south'
    epsg_utm = 32600 + utm_zone if hemisphere == 'north' else 32700 + utm_zone

    # Transform center point to UTM
    transformer_to_utm = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg_utm}', always_xy=True)
    transformer_to_wgs = Transformer.from_crs(f'EPSG:{epsg_utm}', 'EPSG:4326', always_xy=True)

    x_utm, y_utm = transformer_to_utm.transform(x, y)

    # Create bounds in UTM
    half_size = size_m / 2
    minx_utm = x_utm - half_size
    maxx_utm = x_utm + half_size
    miny_utm = y_utm - half_size
    maxy_utm = y_utm + half_size

    # Transform corners back to WGS84
    minx, miny = transformer_to_wgs.transform(minx_utm, miny_utm)
    maxx, maxy = transformer_to_wgs.transform(maxx_utm, maxy_utm)

    return (minx, miny, maxx, maxy)


def get_polygons_for_image(survey, internal_id, month, day, year, date_tolerance_days=1):
    """
    Get all polygons for a specific image from all labelers.

    Parameters:
        survey (str): Survey ID (e.g., '201-225')
        internal_id (int): Internal ID within the survey
        month, day, year (int): Date of the image
        date_tolerance_days (int): Allow date matching within this tolerance

    Returns:
        GeoDataFrame with polygons from all labelers
    """
    # Load polygons
    polygons_gdf = gpd.read_file(get_polygons_path())

    # Load irrigation table to get site_id
    irrigation_df = pd.read_csv(get_irrigation_table_path())

    # Filter to matching survey
    irrigation_df = irrigation_df[irrigation_df['source_file'].str.contains(survey, na=False)]

    # Filter to matching internal_id
    irrigation_df = irrigation_df[irrigation_df['internal_id'] == internal_id]

    if len(irrigation_df) == 0:
        raise ValueError(f"No matching images found for survey={survey}, internal_id={internal_id}")

    # Get site_id (should be same for all dates at this location)
    site_id = irrigation_df['site_id'].iloc[0]

    # Get coordinates
    x = irrigation_df['x'].iloc[0]
    y = irrigation_df['y'].iloc[0]

    # Filter polygons by site_id and date (with tolerance)
    from datetime import datetime, timedelta
    target_date = datetime(year, month, day)

    def date_matches(row):
        try:
            poly_date = datetime(int(row['year']), int(row['month']), int(row['day']))
            return abs((poly_date - target_date).days) <= date_tolerance_days
        except:
            return False

    matching_polygons = polygons_gdf[
        (polygons_gdf['site_id'] == site_id) &
        polygons_gdf.apply(date_matches, axis=1)
    ].copy()

    return matching_polygons, x, y, site_id


def plot_screenshot_with_polygons(screenshot_path=None, survey=None, internal_id=None,
                                   month=None, day=None, year=None,
                                   ax=None, figsize=(10, 10), title=None,
                                   polygon_linewidth=2, show_legend=True):
    """
    Plot a GEE screenshot with polygon overlays from all labelers.

    Parameters:
        screenshot_path (str, optional): Path to the screenshot. If None, will try to find
            based on survey, internal_id, and date.
        survey, internal_id, month, day, year: Image identifiers (used if screenshot_path is None
            or to find matching polygons)
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
        figsize (tuple): Figure size if creating new figure
        title (str, optional): Plot title. If None, auto-generates from image info.
        polygon_linewidth (float): Line width for polygon outlines
        show_legend (bool): Whether to show the labeler legend

    Returns:
        matplotlib.axes.Axes: The axes with the plot
    """
    # If no screenshot path provided, find it
    if screenshot_path is None:
        if survey is None or internal_id is None:
            raise ValueError("Must provide either screenshot_path or (survey, internal_id, date)")

        screenshots = list_available_screenshots()
        matching = [s for s in screenshots
                   if s['survey'] == survey and s['internal_id'] == internal_id]

        if month is not None and day is not None and year is not None:
            matching = [s for s in matching
                       if s['month'] == month and s['day'] == day and s['year'] == year]

        if len(matching) == 0:
            raise ValueError(f"No screenshot found for survey={survey}, internal_id={internal_id}")

        screenshot_path = matching[0]['path']
        if month is None:
            month, day, year = matching[0]['month'], matching[0]['day'], matching[0]['year']

    # Parse screenshot filename if needed
    if survey is None or internal_id is None or month is None:
        info = parse_screenshot_filename(screenshot_path)
        survey = info['survey']
        internal_id = info['internal_id']
        month = info['month']
        day = info['day']
        year = info['year']

    # Load the screenshot
    img = Image.open(screenshot_path)
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]

    # Get polygons and image bounds
    polygons_gdf, x, y, site_id = get_polygons_for_image(
        survey, internal_id, month, day, year
    )

    # Get image bounds in WGS84
    minx, miny, maxx, maxy = get_image_bounds(x, y)

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot the screenshot
    ax.imshow(img_array, extent=[minx, maxx, miny, maxy])

    # Plot polygons by labeler
    labelers_plotted = set()

    for labeler in polygons_gdf['operator_initials'].unique():
        labeler_polys = polygons_gdf[polygons_gdf['operator_initials'] == labeler]
        color = LABELER_COLORS_HEX.get(labeler, '#333333')

        for _, row in labeler_polys.iterrows():
            geom = row.geometry
            if geom.geom_type == 'Polygon':
                coords = np.array(geom.exterior.coords)
                ax.plot(coords[:, 0], coords[:, 1], color=color,
                       linewidth=polygon_linewidth, label=labeler if labeler not in labelers_plotted else '')
                labelers_plotted.add(labeler)
            elif geom.geom_type == 'MultiPolygon':
                for poly in geom.geoms:
                    coords = np.array(poly.exterior.coords)
                    ax.plot(coords[:, 0], coords[:, 1], color=color,
                           linewidth=polygon_linewidth, label=labeler if labeler not in labelers_plotted else '')
                    labelers_plotted.add(labeler)

    # Set axis limits to image bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Add title
    if title is None:
        title = f"Survey {survey}, ID {internal_id}\n{year}-{month:02d}-{day:02d} | Site: {site_id}"
    ax.set_title(title)

    # Add legend
    if show_legend and len(labelers_plotted) > 0:
        legend_elements = [Line2D([0], [0], color=LABELER_COLORS_HEX.get(lab, '#333333'),
                                  linewidth=polygon_linewidth, label=lab)
                         for lab in sorted(labelers_plotted)]
        ax.legend(handles=legend_elements, loc='upper right', title='Labeler')

    # Remove axes for cleaner visualization
    ax.axis('off')

    return ax


def plot_random_screenshot(ax=None, figsize=(10, 10), **kwargs):
    """
    Plot a random available GEE screenshot with polygon overlays.

    Parameters:
        ax: Axes to plot on (optional)
        figsize: Figure size if creating new figure
        **kwargs: Additional arguments passed to plot_screenshot_with_polygons

    Returns:
        matplotlib.axes.Axes: The axes with the plot
    """
    screenshots = list_available_screenshots()
    if len(screenshots) == 0:
        raise ValueError("No screenshots available in GEE_screenshots directory")

    # Pick a random one
    import random
    selected = random.choice(screenshots)

    print(f"Selected: Survey {selected['survey']}, ID {selected['internal_id']}, "
          f"Date {selected['year']}-{selected['month']:02d}-{selected['day']:02d}")

    return plot_screenshot_with_polygons(
        screenshot_path=selected['path'],
        survey=selected['survey'],
        internal_id=selected['internal_id'],
        month=selected['month'],
        day=selected['day'],
        year=selected['year'],
        ax=ax,
        figsize=figsize,
        **kwargs
    )


if __name__ == "__main__":
    # Test the visualization
    import matplotlib.pyplot as plt

    print("Available screenshots:")
    for s in list_available_screenshots():
        print(f"  {s['survey']} ID {s['internal_id']}: {s['year']}-{s['month']:02d}-{s['day']:02d}")

    print("\nPlotting random screenshot...")
    plot_random_screenshot()
    plt.show()
