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
from rasterio.warp import transform_geom
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
from matplotlib import pyplot as plt
from datetime import date, datetime, timedelta

# Not sure if we need this, but wouldn't load utils without this.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# import utils.utils
from utils.utils import *
from utils.geometries import bounding_box
IMAGE_CRS = 'EPSG:32735'  # Coordinate reference system for the images

'''
Small class definition to be able to pass in data object
into utils.py's save_data function.
'''
class LabelTif:
    def __init__(self, array, meta):
        self.array = array
        self.meta = meta.copy()

    def read(self):
        return self.array

def create_labels():
    """
    Creates labels the .tif images with corresponding labelled polygons.
    Note: For a particular .tif image, we may have more than one polygon file.
    Therefore, we may create multiple labels for a single .tif image.

    Parameters:
        - path (str): Location of the specific tif image
    """
    
    # Retrieve irrigation table
    IRRIGATION_TABLE = create_irrigation_table()
    
    # Create a label .tif for each location/time/source combination
    for row in IRRIGATION_TABLE.itertuples():
        irrigation_geojson = row.source_file
        irrigation_geojson = get_data_root() + "/labels/labeled_surveys/random_sample/processed/" + irrigation_geojson + ".geojson" 
        
        if not os.path.isfile(irrigation_geojson):
            raise RuntimeError(f"Unable to find irrigation geojson file: {irrigation_geojson}")

        internal_id = row.internal_id
        unique_id = row.unique_id
        survey_id = int(row.site_id)
        path_to_feature_file = get_data_root() + "features_v2/" + f"{unique_id}_{survey_id}_{row.year:04d}.{row.month:02d}.{row.day:02d}_image.tif"
        image_meta = get_image_meta(path_to_feature_file)
        timestamp = date(row.year, row.month, row.day)
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)

        # Retrieve labels
        label_array = rasterize_polygons(gdf, image_meta)

        # Save labels – to data/dataset/labels
        operator_initials = irrigation_geojson.split("/")[-1].split("_")[0]
        output_label_path = f"dataset/labels/{unique_id}_{survey_id}_{row.year:04d}.{row.month:02d}.{row.day:02d}_{operator_initials}.tif"
        save_label_raster(label_array, image_meta, output_label_path, 
                          description=f"Row {unique_id} of irrigation table: Label for site {survey_id} at {timestamp.strftime('%Y.%m.%d')} by {operator_initials}")

def create_irrigation_table():
    '''
    Creates irrigation table with location, time, and source.
    '''
    IRRIGATION_TABLE = pd.read_csv(get_data_root() + 
                                "/labels/labeled_surveys/random_sample/latest_irrigation_table.csv")

    IRRIGATION_TABLE['site_id'] = IRRIGATION_TABLE['site_id'].apply(lambda id: id[3:])
    return IRRIGATION_TABLE

def get_survey_data(input_image_path):
    """
    Retrieves the survey date for a particular .tif image.
    
    PRECONDITON: 
        Assume that the image path has format s2_{lat}_{lon}_{windowStartDate}_{windowEndDate}_off-{offset}.tif
        Example: s2_-10.4035_29.1319_2023-05-20_2023-05-30_off-15.tif

    Parameters:
        - input_image_path (str): The input path of the .tif image of interest.

    Output:
        - lat (str): Location latitute
        - lon (str): Location longitude
        - survey_date (Date): Date of corresponding survey.
    """

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

def get_image_meta(input_image_path):
    """
    Retrieve metadata for a particular .tif image.

    Parameters:
        - input_image_path (str): The input path of the .tif image of interest.

    Output: 
        - image_meta (dict): Stores image metadata
    """
    with rasterio.open(input_image_path) as src:
        image_meta = src.meta.copy()

    return image_meta

def retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp):
    """
    Retrieve polygons corresponding to a particular .tif image.

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
        -image_meta (dict): Metadata of particular .tif image we are working with.
        - certainty_thresh (int): Minimum certainty for a polygon to be considered irrigated.

    Output: 
        - label_array (numpy.ndarray): A binary numpy array of the same shape as the input image, with
        1's representing irrigated pixels, 0's representing unirrigated pixels.
    """
    IRRIGATION_TYPES = {
        "small-scale": 1,
        "tree_crop": 2,
        "industrial": 3,
        "lawn": 4,
        "covered": 5
    }

    # Create a label array with 8 bands, band 1 for each type of irrigation, bands 2-6 for uncertainty explanations
    labels = np.zeros((8, image_meta['height'], image_meta['width']), dtype=np.uint8)

     # Add certainty score band
    shapes = [(geom, certainty) for geom, certainty in zip(gdf.geometry, gdf.certainty)]
    # Transform geoms from ESPG:4326 to image_meta's crs
    shapes = [
        (transform_geom(CRS.from_string("EPSG:4326"), image_meta['crs'], mapping(geom)), value)
        for geom, value in shapes
    ]

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
        shapes = [(geom, 1) for geom, cat in zip(gdf.geometry, gdf.uncertainty_explanation) if UNCERTAINTY_TYPES[i] in cat.split(";")]
        # Transform geoms from ESPG:4326 to image_meta's crs
        shapes = [
            (transform_geom(CRS.from_string("EPSG:4326"), image_meta['crs'], mapping(geom)), value)
            for geom, value in shapes
        ]
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
    gdf = gdf[gdf['certainty'] >=  certainty_thresh]  

    # Retrieve irrigation bands (first and second bands)
    shapes = []
    for geom, cat in zip(gdf.geometry, gdf.category):
        if cat is None or cat == "":
            cat = "small-scale"  # Default category
        cat = cat.split(";")[0]
        if cat not in IRRIGATION_TYPES:
            raise ValueError(f"Unknown category: '{cat}'")
        geom = transform_geom(CRS.from_string("EPSG:4326"), image_meta['crs'], mapping(geom)) # Transform geom from ESPG:4326 to image_meta's crs
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
    Saves the rasterized labels.

    Parameters:
        - label_array (numpy.ndarray): A binary numpy array of the same shape as the input image, with
        1's representing irrigated pixels, 0's representing unirrigated pixels.
        -image_meta (dict): Metadata of particular .tif image we are working with.
        - output_label_path (str): The output path of the labelled .tif image.
    """
    label_meta = image_meta.copy()
    label_meta.update({
        "count": label_array.shape[0],
    })

    data = LabelTif(label_array, label_meta)
    save_data(data, output_label_path, description=description)
    
if __name__ == "__main__":
    create_labels()