'''
create_label_band.py
Functions to apply labels to all .tif images with corresponding labelled polygons.

Output: 
    .tif image at time T with Label band at the same resolution of the original image,
    binary mask which represents the prescence or absence of irrigation at each pixel.
'''

import os
import sys
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
from matplotlib import pyplot as plt
from datetime import date, datetime, timedelta
import json
import logging

# Not sure if we need this, but wouldn't load utils without this.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# import utils.utils
from utils.utils import *
from utils.geometries import bounding_box

def create_labels(download_dir, version_name=None):
    """
    Creates labels for all downloaded .tif stacks with perfect spatial alignment.
    
    This function:
    1. Finds all *_stack.tif files in the download directory
    2. Parses unique_id from filename (first part: "{unique_id}_{site_id}_{date}_stack.tif")
    3. Looks up site info (lat, lon, date, source_file) from irrigation table
    4. Reads the actual tif to get exact CRS and transform (critical for alignment!)
    5. Rasterizes polygons using the exact same spatial reference as the image
    
    Parameters:
        - download_dir (str): Root directory containing versioned downloads
        - version_name (str, optional): Specific version folder name. If None, uses latest.
    
    Returns:
        - None. Creates {file_id}_labels.tif for each stack file.
    """
    
    # Get the version directory
    if version_name is None:
        # Find the most recent version
        versions = [d for d in os.listdir(download_dir) if os.path.isdir(os.path.join(download_dir, d))]
        versions.sort(reverse=True)
        if not versions:
            raise RuntimeError(f"No version directories found in {download_dir}")
        version_name = versions[0]
        logging.info(f"Using latest version: {version_name}")
    
    version_dir = os.path.join(download_dir, version_name)
    if not os.path.exists(version_dir):
        raise RuntimeError(f"Version directory not found: {version_dir}")
    
    # Get irrigation table for polygon lookups
    IRRIGATION_TABLE = create_irrigation_table()
    
    # Find all stack files (unmasked versions)
    stack_files = [f for f in os.listdir(version_dir) 
                   if f.endswith('_stack.tif') and not f.endswith('_masked.tif')]
    
    logging.info(f"Found {len(stack_files)} stack files to label")
    
    for stack_file in stack_files:
        file_id = stack_file.replace('_stack.tif', '')
        stack_path = os.path.join(version_dir, stack_file)
        
        # Parse unique_id from filename: "{unique_id}_{site_id}_{date}_stack.tif"
        try:
            unique_id = int(file_id.split('_')[0])
        except (ValueError, IndexError) as e:
            logging.warning(f"Could not parse unique_id from {file_id}, skipping")
            continue
        
        # Find matching row in irrigation table using unique_id
        table_row = IRRIGATION_TABLE[IRRIGATION_TABLE['unique_id'] == unique_id]
        if len(table_row) == 0:
            logging.warning(f"No irrigation table entry found for unique_id {unique_id}, skipping")
            continue
        
        table_row = table_row.iloc[0]
        irrigation_geojson = table_row.source_file
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/" + irrigation_geojson + ".geojson"
        
        if not os.path.isfile(irrigation_geojson):
            logging.warning(f"Unable to find irrigation geojson file: {irrigation_geojson}, skipping")
            continue
        
        # Get site IDs and date from table
        internal_id = table_row.internal_id
        survey_id = int(table_row.site_id)
        timestamp = date(table_row.year, table_row.month, table_row.day)
        lat = table_row.y
        lon = table_row.x
        
        # Read the actual stack to get exact CRS and transform for spatial alignment
        with rasterio.open(stack_path) as src:
            image_meta = {
                'crs': src.crs,
                'transform': src.transform,
                'height': src.height,
                'width': src.width,
                'dtype': 'uint8'
            }
        
        logging.info(f"Processing labels for {file_id} at ({lat:.4f}, {lon:.4f})")
        
        # Retrieve and rasterize polygons using exact image metadata
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)
        
        if len(gdf) == 0:
            logging.warning(f"No polygons found for {file_id}, creating empty labels")
        else:
            logging.info(f"Found {len(gdf)} polygons for {file_id}")
        
        label_array = rasterize_polygons(gdf, image_meta)
        
        # Save labels
        operator_initials = irrigation_geojson.split("/")[-1].split("_")[0]
        output_label_path = os.path.join(version_dir, f"{file_id}_labels.tif")
        
        logging.info(f"Saving labels to: {output_label_path}")
        save_label_raster(label_array, image_meta, output_label_path, 
                          description=f"Labels for {file_id}: site {survey_id} at {timestamp.strftime('%Y.%m.%d')} by {operator_initials}")
        
        # Verify file was created
        if os.path.exists(output_label_path):
            file_size = os.path.getsize(output_label_path)
            logging.info(f"✓ Successfully saved labels: {output_label_path} ({file_size} bytes)")
        else:
            logging.error(f"✗ Failed to save labels: {output_label_path}")
    
    logging.info(f"Completed label creation for {len(stack_files)} stacks")

def create_irrigation_table():
    '''
    Creates irrigation table with location, time, and source.
    '''
    IRRIGATION_TABLE = pd.read_csv(get_data_root() + 
                                "/labels/labeled_surveys/random_sample/latest_irrigation_table.csv")

    IRRIGATION_TABLE['site_id'] = IRRIGATION_TABLE['site_id'].apply(lambda id: id[3:])
    return IRRIGATION_TABLE

def retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp):
    """
    Retrieve polygons corresponding to a particular image.

    Parameters:
        - irrigation_geojson (str): Path of GeoJSON file that corresponds to the particular image 
        we are working with.
        - survey_id (int): Full survey id to retrieve polygons at the correct location
        - internal_id (int): Internal survey id to retrieve polygons at the correct location
        - image_meta (dict): Metadata of particular .tif image we are working with.
        - timestamp (Date): Date

    Output: 
        - gdf (geopandas.geodataframe.GeoDataFrame): DataFrame that corresponds to the polygons 
        for the particular image.
    """

    # Check that irrigation_geojson exists
    if not os.path.isfile(irrigation_geojson):
        raise RuntimeError(f"Unable to find irrigation geojson file: {irrigation_geojson}")
    
    gdf = gpd.read_file(irrigation_geojson)
    gdf = gdf.set_crs(image_meta['crs'], allow_override=True)

    # Retrieve correct location. Note some polygons' internal_id is actually
    # its survey id, so we must check both ids.
    gdf = gdf[ (gdf['internal_id'] == survey_id) | (gdf['internal_id'] == internal_id)]

    # Filter by times
    gdf = gdf[ (gdf['year'] == timestamp.year) & (gdf['month'] == timestamp.month) & (gdf['day'] == timestamp.day)]

    return gdf

def rasterize_polygons(gdf, image_meta, certainty_thresh=3):
    """
    Rasterizes the polygons to match the resolution of the particular image.

    Parameters:
        - gdf (geopandas.geodataframe.GeoDataFrame): DataFrame that corresponds to the polygons for the 
        particular image.
        - image_meta (dict): Metadata of particular .tif image we are working with.
        - certainty_thresh (int): Minimum certainty for a polygon to be considered irrigated.

    Output: 
        - label_array (numpy.ndarray): Array with shape (8, height, width) containing:
            Band 0: Categorical irrigation (1=small-scale, 2=tree_crop, 3=industrial, 4=lawn, 5=covered)
            Band 1: Binary irrigation mask (1=irrigated, 0=not irrigated)
            Bands 2-6: Uncertainty flags (1=has this uncertainty type, 0=doesn't)
            Band 7: Certainty score (1-5)
    """
    IRRIGATION_TYPES = {
        "small-scale": 1,
        "tree_crop": 2,
        "industrial": 3,
        "lawn": 4,
        "covered": 5
    }

    # Create a label array with 8 bands
    labels = np.zeros((8, image_meta['height'], image_meta['width']), dtype=np.uint8)

    # Add certainty score band (band 7)
    shapes = [(geom, certainty) for geom, certainty in zip(gdf.geometry, gdf.certainty)]
    certainty_array = rasterize(
        shapes=shapes,
        out_shape=(image_meta['height'], image_meta['width']),
        transform=image_meta['transform'],
        fill=0,
        dtype='uint8'
    )
    labels[7] = certainty_array

    # Retrieve uncertainty bands 2-6
    UNCERTAINTY_TYPES = [
        "unclear signs of agriculture",
        "only slightly green",
        "uneven",
        "may naturally be green",
        "may be a fishpond"
    ]

    for i in range(5):
        shapes = [(geom, 1) for geom, cat in zip(gdf.geometry, gdf.uncertainty_explanation) 
                  if UNCERTAINTY_TYPES[i] in cat.split(";")]
        mask = rasterize(
            shapes=shapes,
            out_shape=(image_meta['height'], image_meta['width']),
            transform=image_meta['transform'],
            fill=0,
            dtype='uint8'
        ) 
        labels[i + 2] = mask

    # Add the actual irrigation bands, but only if the certainty is high enough
    # Filter out low certainty polygons
    gdf = gdf[gdf['certainty'] >= certainty_thresh]  

    # Retrieve irrigation bands (first and second bands)
    shapes = []
    for geom, cat in zip(gdf.geometry, gdf.category):
        if cat is None or cat == "":
            cat = "small-scale"  # Default category
        cat = cat.split(";")[0]
        if cat not in IRRIGATION_TYPES:
            raise ValueError(f"Unknown category: '{cat}'")
        shapes.append((geom, IRRIGATION_TYPES[cat]))

    label_array = rasterize(
        shapes=shapes,
        out_shape=(image_meta['height'], image_meta['width']),
        transform=image_meta['transform'],
        fill=0,
        dtype='uint8'
    )

    # Second band is a binary mask of first band
    labels[0] = label_array
    labels[1] = np.where(label_array != 0, 1, 0)

    return labels

def save_label_raster(label_array, image_meta, output_label_path, description="Label for irrigation data"):
    """
    Saves the rasterized labels directly with rasterio.

    Parameters:
        - label_array (numpy.ndarray): Array with shape (8, height, width) containing irrigation labels.
        - image_meta (dict): Metadata of particular .tif image we are working with.
        - output_label_path (str): The output path of the labelled .tif image.
        - description (str): Description to save in metadata.
    """
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
    
    # Prepare metadata for saving
    label_meta = {
        'driver': 'GTiff',
        'dtype': 'uint8',
        'width': image_meta['width'],
        'height': image_meta['height'],
        'count': label_array.shape[0],  # Number of bands
        'crs': image_meta['crs'],
        'transform': image_meta['transform'],
        'compress': 'lzw'
    }
    
    # Save directly with rasterio
    with rasterio.open(output_label_path, 'w', **label_meta) as dst:
        dst.write(label_array)
        dst.set_band_description(1, 'Categorical irrigation (1-5)')
        dst.set_band_description(2, 'Binary irrigation mask')
        dst.set_band_description(3, 'Uncertainty: unclear agriculture')
        dst.set_band_description(4, 'Uncertainty: only slightly green')
        dst.set_band_description(5, 'Uncertainty: uneven')
        dst.set_band_description(6, 'Uncertainty: may be natural')
        dst.set_band_description(7, 'Uncertainty: may be fishpond')
        dst.set_band_description(8, 'Certainty score (1-5)')
    
    logging.info(f"Saved label raster: {output_label_path}")

# ============================================================================
# DEPRECATED FUNCTIONS - Kept for backward compatibility with tests
# Only get_survey_data() is used (in test_get_survey_data)
# ============================================================================

def get_survey_data(input_image_path):
    """
    DEPRECATED: This function was designed for old naming convention.
    Use metadata JSON files instead.
    
    Retrieves the survey date for a particular .tif image.
    
    PRECONDITION: 
        Assume that the image path has format s2_{lat}_{lon}_{windowStartDate}_{windowEndDate}_off-{offset}.tif
        Example: s2_-10.4035_29.1319_2023-05-20_2023-05-30_off-15.tif

    Parameters:
        - input_image_path (str): The input path of the .tif image of interest.

    Output:
        - lat (str): Location latitude
        - lon (str): Location longitude
        - survey_date (Date): Date of corresponding survey.
    """
    logging.warning("get_survey_data() is deprecated - use metadata JSON files instead")
    
    # Retrieve tokens
    tokens = input_image_path[:-4].split("_")
    
    # Start, end date
    lat = tokens[1]
    lon = tokens[2]
    start_date = datetime.strptime(tokens[3], "%Y-%m-%d").date()
    end_date = datetime.strptime(tokens[4], "%Y-%m-%d").date()

    # Retrieve survey date
    middle_date = start_date + (end_date - start_date) / 2
    offset = int(tokens[-1][4:])
    survey_date = middle_date + timedelta(days=offset)
    return lat, lon, survey_date

def create_bounding_box(center_lat, center_lon):
    """
    Helper function for TESTS ONLY that creates a bounding box around a center point. 
    Uses method utils.geometries.bounding_box to retrieve lat/lon bounds.
    
    NOTE: This is NOT used in production - production code reads metadata from actual GEE downloads.
    
    Parameters:
        - center_lat (float): Latitude of the center point.
        - center_lon (float): Longitude of the center point.
    
    Returns:
        - image_meta (dict): Dictionary which contains:
            - height (int): Height of the bounding box in pixels.
            - width (int): Width of the bounding box in pixels.
            - crs (str): Coordinate reference system.
            - transform (Affine): Affine transformation for the bounding box.
    """

    # Get lat/lon bounds
    min_lat, min_lon, max_lat, max_lon = bounding_box(center_lat, center_lon)

    # Image dimensions
    width = 100
    height = 100

    pixel_size_lon = (max_lon - min_lon) / width
    pixel_size_lat = (max_lat - min_lat) / height
    top_left_lon = min_lon
    top_left_lat = max_lat

    transform = rasterio.transform.from_origin(top_left_lon, top_left_lat, pixel_size_lon, pixel_size_lat)

    image_meta = {
        'height': height,
        'width': width,
        'crs': 'EPSG:32633',  # UTM zone 33N - used in tests only
        'transform': transform,
        'dtype': 'uint8',
    }

    return image_meta

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get paths
    project_root = find_project_root(current_path=os.getcwd())
    data_root = get_data_root()
    download_dir = os.path.join(data_root, "features")
    
    # Create labels for the latest download
    create_labels(download_dir)