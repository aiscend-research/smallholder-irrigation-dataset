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

# Not sure if we need this, but wouldn't load utils without this.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# import utils.utils
from utils.utils import *

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

def create_labels(tif_path):
    """
    Creates labels the .tif images with corresponding labelled polygons.
    Note: For a particular .tif image, we may have more than one polygon file.
    Therefore, we may create multiple labels for a single .tif image.

    Parameters:
        - path (str): Location of the specific tif image
    """
    
    # Retrieve irrigation table
    IRRIGATION_TABLE = create_irrigation_table()
    
    image_meta = get_image_meta(tif_path)

    # Retrieve data labeled image (from file name)
    file = tif_path.split("/")[-1]
    lat, lon, timestamp = get_survey_data(file)

    # Retrieve survey_id, internal_id, and irrigation_geojsons
    survey_id, internal_id, irrigation_geojsons = get_polygon_file(lat, lon, IRRIGATION_TABLE)

    # Create a label .tif for each polygon file
    for irrigation_geojson in irrigation_geojsons:
        gdf = retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp)

        # Retrieve labels
        label_array = rasterize_polygons(gdf, image_meta)

        # Save labels – to data/dataset/labels
        operator_initials = irrigation_geojson.split("/")[-1].split("_")[0]
        output_label_path = f"dataset/labels/{file[:-4]}_{operator_initials}.tif"
        save_label_raster(label_array, image_meta, output_label_path)

def create_irrigation_table():
    '''
    Creates irrigation table with location, time, and source.
    '''
    IRRIGATION_TABLE = pd.read_csv(get_data_root() + 
                                "/labels/labeled_surveys/random_sample/latest_irrigation_table.csv",
                                usecols=['internal_id', 'site_id', 'x', 'y', 'source_file', 'operator_initials'])

    IRRIGATION_TABLE['site_id'] = IRRIGATION_TABLE['site_id'].apply(lambda id: id[3:])

    # Format so we can query for lat/lon without rounding issue
    IRRIGATION_TABLE['x'] = (IRRIGATION_TABLE['x'] * 1000).astype(int) / 1000
    IRRIGATION_TABLE['y'] = (IRRIGATION_TABLE['y'] * 1000).astype(int) / 1000
    IRRIGATION_TABLE['x'] = IRRIGATION_TABLE['x'].astype(str)
    IRRIGATION_TABLE['y'] = IRRIGATION_TABLE['y'].astype(str)
    return IRRIGATION_TABLE

def get_polygon_file(lat, lon, IRRIGATION_TABLE):
    '''
    Retrieves the corresponding polygon file for a particular location
    by querying IRRIGATION_TABLE by location. Also retrieves 
    survey_id and internal_id, such that we can find the correct location
    within the source_file

    Parameters: 
        - lat (str): Location latitude
        - lon (str): Location longitude

    Output:
        - survey_id (int): Full id for the survey
        - internal_id (int): Internal id for the survey
        - source_files (list of str): The .geojson file(s) that contains labelled
         polygons that corresponds with this location.
    '''
    lon = lon[:-1]
    lat = lat[:-1]
    source_file = IRRIGATION_TABLE[ (IRRIGATION_TABLE['x'] == lon) & (IRRIGATION_TABLE['y'] == lat) ]

    if (source_file.empty):
        raise RuntimeError(f"Unable to find polygon file for location ({lat},{lon})")
    
    internal_id = source_file.iloc[0].internal_id
    survey_id = int(source_file.iloc[0].site_id)
    source_files =  [ get_data_root() + "/labels/labeled_surveys/random_sample/processed/" + 
                     source_file.iloc[i].source_file + ".geojson" for i in range(len(source_file)) ]


    return survey_id, internal_id, source_files

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

def retrieve_polygons(irrigation_geojson, survey_id, internal_id, image_meta, timestamp, certainty_thresh=4):
    """
    Retrieve polygons corresponding to a particular .tif image.

    Parameters:
        - irrigation_geojson (str): Path of GeoJSON file that corresponds to the particular image 
        we are working with.
        - survey_id (int): Full survey id to retrieve polygons at the correct location
        - internal_id (int): Internal survey id to retrieve polygons at the correct location
        - image_meta (dict): Metadata of particular .tif image we are working with.
        - timestamp (Date): Date
        - certainty_thresh (int): Minimum certainty for a polygon to be considered irrigated.

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

    # Filter out low certainty polygons
    gdf = gdf[gdf['certainty'] >=  certainty_thresh]  

    return gdf

def rasterize_polygons(gdf, image_meta):
    """
    Rasterizes the polygons to match the resolution of the particular image.

    Parameters:
        - gdf (geopandas.geodataframe.GeoDataFrame): DataFrame that corresponds to the polygons for the 
        particular image.
        -image_meta (dict): Metadata of particular .tif image we are working with.

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

    # Create a label array with 6 bands, band 1 for each type of irrigation, bands 2-6 for uncertainty explanations
    labels = np.zeros((8, image_meta['height'], image_meta['width']), dtype=np.uint8)

    # Retrieve irrigation bands (first and second bands)
    shapes = []
    for geom, cat in zip(gdf.geometry, gdf.category):
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
        mask = rasterize(
            shapes=shapes,
            out_shape=(image_meta['height'], image_meta['width']),
            transform=image_meta['transform'],
            fill=0,
            dtype='uint8'
        ) 
        labels[i + 2] = mask

    # Add certainty score band
    shapes = [(geom, certainty) for geom, certainty in zip(gdf.geometry, gdf.certainty)]
    certainty_array = rasterize(
        shapes=shapes,
        out_shape=(image_meta['height'], image_meta['width']),
        transform=image_meta['transform'],
        fill=0,
        dtype='uint8'
    )
    labels[7] = certainty_array

    return labels

def save_label_raster(label_array, image_meta, output_label_path):
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
    save_data(data, output_label_path)

if __name__ == "__main__":
    path = get_data_root() + "/dataset/images"
    files = [f for f in os.listdir(path) 
                 if os.path.isfile(os.path.join(path, f)) and f.lower().endswith('.tif')]
    
    for file in files:
        file = os.path.join(path, file)
        create_labels(file)