'''
create_label_band.py
Functions to apply labels to all .tif images with corresponding labelled polygons.

Output:
    For each stack .tif and each labeler who annotated that location-date:
    A label .tif with 9 bands at the same resolution as the original image.

Supports both Sentinel-2 and PlanetScope via the 'sensor' parameter:
    - sentinel2: Uses data/features/ directory
    - planetscope: Uses data/features_planet/ directory
'''

import os
import sys
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.utils import get_data_root

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sensor configurations (matches satellite_visualization.py)
SENSOR_CONFIG = {
    'sentinel2': {
        'default_version': '20260107_180813',
        'data_dir': 'features',
    },
    'planetscope': {
        'default_version': '20260127_161535_SR',
        'data_dir': 'features_planet',
    }
}


def create_labels(download_dir=None, version_name=None, sensor='sentinel2'):
    """
    Creates labels for all downloaded .tif stacks with perfect spatial alignment.
    Creates SEPARATE label files for EACH labeler who annotated each location-date.

    Works for both Sentinel-2 and PlanetScope data - the label format is identical.

    Parameters:
        - download_dir (str, optional): Root directory containing versioned downloads.
            If None, uses the default for the sensor type.
        - version_name (str, optional): Specific version folder name. If None, uses latest.
        - sensor (str): Sensor type ('sentinel2' or 'planetscope'). Default 'sentinel2'.

    Returns:
        - None. Creates {file_id}_{operator}_labels.tif for each stack file and labeler.
    """
    if sensor not in SENSOR_CONFIG:
        raise ValueError(f"Unknown sensor type: {sensor}. Must be one of {list(SENSOR_CONFIG.keys())}")

    config = SENSOR_CONFIG[sensor]

    # Set default download directory based on sensor
    if download_dir is None:
        download_dir = os.path.join(get_data_root(), config['data_dir'])

    # Get the version directory
    if version_name is None:
        versions = [d for d in os.listdir(download_dir) if os.path.isdir(os.path.join(download_dir, d))]
        versions.sort(reverse=True)
        if not versions:
            raise RuntimeError(f"No version directories found in {download_dir}")
        version_name = versions[0]
        logging.info(f"Using latest version: {version_name}")

    version_dir = os.path.join(download_dir, version_name)
    if not os.path.exists(version_dir):
        raise RuntimeError(f"Version directory not found: {version_dir}")

    logging.info(f"Creating labels for {sensor} data in {version_dir}")

    # Load the combined polygons file (contains all labelers)
    polygons_path = os.path.join(get_data_root(), 'labels/labeled_surveys/random_sample/latest_polygons.geojson')
    if not os.path.exists(polygons_path):
        raise RuntimeError(f"Polygons file not found: {polygons_path}")
    all_polygons = gpd.read_file(polygons_path)
    logging.info(f"Loaded {len(all_polygons)} polygons from {polygons_path}")

    # Load irrigation table to map unique_id -> site_id, date
    irrigation_table = load_irrigation_table()

    # Find all stack files (unmasked versions)
    stack_files = [f for f in os.listdir(version_dir)
                   if f.endswith('_stack.tif') and '_masked' not in f]

    logging.info(f"Found {len(stack_files)} stack files to label")

    for stack_file in stack_files:
        file_id = stack_file.replace('_stack.tif', '')
        stack_path = os.path.join(version_dir, stack_file)

        # Parse from filename: "{unique_id}_{site_numeric}_{YYYY.MM.DD}_stack.tif"
        # NOTE: unique_id in filename may not match irrigation table, so we use site + date
        try:
            parts = file_id.split('_')
            site_numeric = parts[1]  # e.g., "5119273"
            date_str = parts[2]      # e.g., "2021.09.16"
            date_parts = date_str.split('.')
            year, month, day = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
            site_id = f"id_{site_numeric}"
        except (ValueError, IndexError) as e:
            logging.warning(f"Could not parse filename {file_id}: {e}, skipping")
            continue

        # Verify site exists in irrigation table
        table_rows = irrigation_table[irrigation_table['site_id'] == site_id]
        if len(table_rows) == 0:
            logging.warning(f"No irrigation table entry found for site_id {site_id}, skipping")
            continue

        # Read the actual stack to get exact CRS and transform
        with rasterio.open(stack_path) as src:
            image_meta = {
                'crs': src.crs,
                'transform': src.transform,
                'height': src.height,
                'width': src.width,
                'dtype': 'uint8'
            }

        logging.info(f"Processing {file_id}: site={site_id}, date={year}-{month:02d}-{day:02d}")

        # Find all labelers who annotated this site-date
        site_polygons = all_polygons[
            (all_polygons['site_id'] == site_id) &
            (all_polygons['year'] == year) &
            (all_polygons['month'] == month) &
            (all_polygons['day'] == day)
        ]

        if len(site_polygons) == 0:
            # No polygons = "no irrigation" label. Find which labelers assessed this image.
            # Look up in irrigation table by site_id and date
            matching_rows = irrigation_table[
                (irrigation_table['site_id'] == site_id) &
                (irrigation_table['year'] == year) &
                (irrigation_table['month'] == month) &
                (irrigation_table['day'] == day)
            ]

            if len(matching_rows) == 0:
                logging.warning(f"No irrigation table entry for {file_id}, skipping")
                continue

            # Create empty label files for each labeler who assessed this image
            labelers = matching_rows['operator_initials'].unique()
            logging.info(f"No polygons for {file_id}, creating empty labels for {len(labelers)} labelers: {list(labelers)}")

            for operator in labelers:
                # Create empty label array (all zeros)
                empty_gdf = gpd.GeoDataFrame(columns=['geometry', 'certainty', 'category', 'uncertainty_explanation'])
                label_array = rasterize_polygons(empty_gdf, image_meta)

                output_label_path = os.path.join(version_dir, f"{file_id}_{operator}_labels.tif")
                save_label_raster(
                    label_array,
                    image_meta,
                    output_label_path,
                    description=f"Labels for {file_id} by {operator}: site {site_id} at {year}.{month:02d}.{day:02d} (no irrigation)"
                )
                logging.info(f"  Created empty labels for {operator}")
            continue

        # Get unique labelers for this site-date
        labelers = site_polygons['operator_initials'].unique()
        logging.info(f"Found {len(site_polygons)} polygons from {len(labelers)} labelers: {list(labelers)}")

        # Create a label file for EACH labeler
        for operator in labelers:
            operator_polygons = site_polygons[site_polygons['operator_initials'] == operator].copy()

            # Rasterize this labeler's polygons
            label_array = rasterize_polygons(operator_polygons, image_meta)

            # Save with operator in filename
            output_label_path = os.path.join(version_dir, f"{file_id}_{operator}_labels.tif")
            save_label_raster(
                label_array,
                image_meta,
                output_label_path,
                description=f"Labels for {file_id} by {operator}: site {site_id} at {year}.{month:02d}.{day:02d}"
            )

            logging.info(f"  Created labels for {operator}: {len(operator_polygons)} polygons")


def load_irrigation_table():
    """Load and prepare the irrigation table."""
    import pandas as pd
    irrigation_path = os.path.join(get_data_root(), 'labels/labeled_surveys/random_sample/latest_irrigation_table.csv')
    df = pd.read_csv(irrigation_path)
    return df


def rasterize_polygons(gdf, image_meta, certainty_thresh=3, coverage_supersample=10):
    """
    Rasterizes the polygons to match the resolution of the particular image.
    Properly transforms coordinates from EPSG:4326 to image CRS.

    Parameters:
        - gdf (geopandas.geodataframe.GeoDataFrame): Polygons for this image/labeler
        - image_meta (dict): Metadata of the .tif image
        - certainty_thresh (int): Minimum certainty for irrigation classification (default 3)
        - coverage_supersample (int): Factor to supersample for coverage calculation (default 10)

    Output:
        - label_array (numpy.ndarray): Array with shape (9, height, width) containing:
            Band 0: Categorical irrigation (1=small-scale, 2=tree_crop, 3=industrial, 4=lawn, 5=covered)
            Band 1: Binary irrigation mask (1=irrigated, 0=not irrigated)
            Bands 2-6: Uncertainty flags (1=has this uncertainty type, 0=doesn't)
            Band 7: Certainty score (1-5)
            Band 8: % polygon coverage (0-100) for certainty >= threshold
    """
    IRRIGATION_TYPES = {
        "small-scale": 1,
        "tree_crop": 2,
        "industrial": 3,
        "lawn": 4,
        "covered": 5
    }

    UNCERTAINTY_TYPES = [
        "unclear signs of agriculture",
        "only slightly green",
        "uneven",
        "may naturally be green",
        "may be a fishpond"
    ]

    height = image_meta['height']
    width = image_meta['width']
    target_crs = image_meta['crs']
    transform = image_meta['transform']

    # Create a label array with 9 bands (8 original + 1 coverage)
    labels = np.zeros((9, height, width), dtype=np.float32)

    if len(gdf) == 0:
        return labels.astype(np.uint8)

    # Reproject entire GeoDataFrame from EPSG:4326 to image CRS
    # This is more reliable than transforming individual geometries
    gdf_projected = gdf.copy()
    if gdf_projected.crs is None:
        gdf_projected = gdf_projected.set_crs("EPSG:4326")
    gdf_projected = gdf_projected.to_crs(target_crs)

    def get_shapes(gdf_subset, value_column=None, default_value=1):
        """Extract (geometry, value) tuples from projected GeoDataFrame."""
        shapes = []
        for idx, row in gdf_subset.iterrows():
            if value_column and value_column in row:
                value = row[value_column]
            else:
                value = default_value
            shapes.append((mapping(row.geometry), value))
        return shapes

    # Band 7: Certainty score (all polygons) - use projected geometries
    cert_mask = gdf_projected['certainty'] > 0
    if cert_mask.any():
        shapes = [(mapping(geom), int(cert)) for geom, cert in
                  zip(gdf_projected.loc[cert_mask, 'geometry'], gdf_projected.loc[cert_mask, 'certainty'])]
        labels[7] = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='uint8'
        )

    # Bands 2-6: Uncertainty flags - use projected geometries
    for i, uncertainty_type in enumerate(UNCERTAINTY_TYPES):
        shapes = []
        for geom, explanation in zip(gdf_projected.geometry, gdf_projected['uncertainty_explanation']):
            if isinstance(explanation, str) and uncertainty_type in explanation:
                shapes.append((mapping(geom), 1))
        if shapes:
            labels[i + 2] = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype='uint8'
            )

    # Filter to high-certainty polygons for irrigation bands and coverage
    # Use PROJECTED geometries for rasterization
    high_cert_mask = gdf_projected['certainty'] >= certainty_thresh
    high_cert_gdf = gdf_projected[high_cert_mask].copy()

    if len(high_cert_gdf) > 0:
        # First, calculate coverage percentage using supersampling
        # This determines which pixels are considered "irrigated" (>= 50% coverage)
        super_height = height * coverage_supersample
        super_width = width * coverage_supersample

        # Adjust transform for supersampled resolution
        super_transform = rasterio.transform.Affine(
            transform.a / coverage_supersample,
            transform.b,
            transform.c,
            transform.d,
            transform.e / coverage_supersample,
            transform.f
        )

        # Create binary mask at supersampled resolution using projected geometries
        coverage_shapes = [(mapping(geom), 1) for geom in high_cert_gdf.geometry]

        coverage_pct = np.zeros((height, width), dtype=np.float32)
        if coverage_shapes:
            super_mask = rasterize(
                shapes=coverage_shapes,
                out_shape=(super_height, super_width),
                transform=super_transform,
                fill=0,
                dtype='uint8'
            )

            # Reshape and compute mean for each output pixel
            # This gives us the fraction of sub-pixels that are covered
            super_mask = super_mask.reshape(height, coverage_supersample, width, coverage_supersample)
            coverage_pct = super_mask.mean(axis=(1, 3)) * 100  # Convert to percentage

        # Band 8: Store coverage percentage
        labels[8] = coverage_pct

        # Create mask for pixels with >= 50% coverage
        irrigated_mask = coverage_pct >= 50

        # Band 0: Categorical irrigation type (only where coverage >= 50%)
        shapes = []
        for geom, cat in zip(high_cert_gdf.geometry, high_cert_gdf['category']):
            if cat is None or cat == "" or (isinstance(cat, float) and np.isnan(cat)):
                cat = "small-scale"
            cat = str(cat).split(";")[0]
            if cat not in IRRIGATION_TYPES:
                logging.warning(f"Unknown category '{cat}', defaulting to small-scale")
                cat = "small-scale"
            shapes.append((mapping(geom), IRRIGATION_TYPES[cat]))

        if shapes:
            categorical = rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype='uint8'
            )
            # Apply coverage threshold: only mark as irrigated if >= 50% covered
            labels[0] = np.where(irrigated_mask, categorical, 0)

        # Band 1: Binary irrigation mask (1 where coverage >= 50%)
        labels[1] = irrigated_mask.astype(np.uint8)

    # Convert to appropriate dtype (uint8 for all except coverage which stays float)
    # We'll store coverage as uint8 (0-100)
    labels = labels.astype(np.uint8)

    return labels


def save_label_raster(label_array, image_meta, output_label_path, description="Label for irrigation data"):
    """
    Saves the rasterized labels.

    Parameters:
        - label_array (numpy.ndarray): Array with shape (9, height, width) containing irrigation labels.
        - image_meta (dict): Metadata of particular .tif image we are working with.
        - output_label_path (str): The output path of the labelled .tif image.
        - description (str): Description to save in metadata.
    """
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_label_path) if os.path.dirname(output_label_path) else '.', exist_ok=True)

    # Prepare metadata for saving
    label_meta = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': image_meta['width'],
        'height': image_meta['height'],
        'count': label_array.shape[0],
        'crs': image_meta['crs'],
        'transform': image_meta['transform'],
        'compress': 'lzw'
    }

    # Save with rasterio
    with rasterio.open(output_label_path, 'w', **label_meta) as dst:
        dst.write(label_array)
        dst.update_tags(DESCRIPTION=description)
        dst.set_band_description(1, 'Categorical irrigation (1=small-scale, 2=tree_crop, 3=industrial, 4=lawn, 5=covered)')
        dst.set_band_description(2, 'Binary irrigation mask')
        dst.set_band_description(3, 'Uncertainty: unclear agriculture')
        dst.set_band_description(4, 'Uncertainty: only slightly green')
        dst.set_band_description(5, 'Uncertainty: uneven')
        dst.set_band_description(6, 'Uncertainty: may be natural')
        dst.set_band_description(7, 'Uncertainty: may be fishpond')
        dst.set_band_description(8, 'Certainty score (1-5)')
        dst.set_band_description(9, 'Polygon coverage % (certainty >= 3)')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create label bands for downloaded satellite stacks')
    parser.add_argument('--download_dir', type=str, default=None,
                        help='Directory containing versioned downloads (default: auto from sensor)')
    parser.add_argument('--version', type=str, default=None,
                        help='Specific version name (default: latest)')
    parser.add_argument('--sensor', type=str, default='sentinel2',
                        choices=['sentinel2', 'planetscope'],
                        help='Sensor type (default: sentinel2)')

    args = parser.parse_args()

    create_labels(args.download_dir, args.version, args.sensor)
