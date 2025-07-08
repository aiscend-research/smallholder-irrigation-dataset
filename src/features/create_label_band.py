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

# CSV to determine source polygon file.
IRRIGATION_TABLE = None

'''
Small class definition to be able to pass in data object
into utils.py's save_data function.
'''
class LabelTif:
    def __init__(self, array, meta):
        self.array = array
        self.meta = meta.copy()

    def read(self):
        return self.array.reshape(1, self.array.shape[0], self.array.shape[1])
    

def create_all_labels(tif_path):
    """
    Creates labels for all .tif images with corresponding labelled polygons.
    Loops through all location-time combos in the hpc, retrieves the corresponding 
    time series and polygons, creates a label band for the location/time .tif, and uploads it to hpc.

    Parameters:
        - path (str): Location of directory containing all location-time combos (images)
    """

    # Retrieve list of all .tif files
    files = [f for f in os.listdir(tif_path) 
                 if os.path.isfile(os.path.join(tif_path, f)) and f.lower().endswith('.tif')]
    
    # Retrieve irrigation table (need location, time, and source)
    global IRRIGATION_TABLE
    IRRIGATION_TABLE = pd.read_csv(get_data_root() + 
                                "/labels/labeled_surveys/random_sample/latest_irrigation_table.csv",
                                usecols=['internal_id', 'x', 'y', 'source_file'])

    # Format so we can query for lat/lon without rounding issue
    IRRIGATION_TABLE['x'] = (IRRIGATION_TABLE['x'] * 1000).astype(int) / 1000
    IRRIGATION_TABLE['y'] = (IRRIGATION_TABLE['y'] * 1000).astype(int) / 1000
    IRRIGATION_TABLE['x'] = IRRIGATION_TABLE['x'].astype(str)
    IRRIGATION_TABLE['y'] = IRRIGATION_TABLE['y'].astype(str)

    for file in files:
        input_image_path = path + "/" + file
        image_meta = get_image_meta(input_image_path)

        # Retrieve data labeled image (from file name)
        lat, lon, timestamp = get_survey_data(file)

        irrigation_geojson = get_polygon_file(lat, lon)
        gdf = retrieve_polygons(irrigation_geojson, image_meta, timestamp)

        # Retrieve labels
        label_array = rasterize_polygons(gdf, image_meta)

        # Save labels – to data/dataset/labels
        output_label_path = "dataset/labels/" + file[:-4] + ".tif"
        save_label_raster(label_array, image_meta, output_label_path)

def get_polygon_file(lat, lon):
    '''
    Retrieves the corresponding polygon file for a particular location
    and time, by querying IRRIGATION_TABLE by location.

    Parameters: 
        - lat (str): Location latitude
        - lon (str): Location longitude

    Output:
        - source_file (str): The .geojson file that contains labelled
         polygons that corresponds with this location.
    '''
    lon = lon[:-1]
    lat = lat[:-1]
    # Todo – Retrieve polygon file that corresponds with this .tif
    source_file = IRRIGATION_TABLE[ (IRRIGATION_TABLE['x'] == lon) & (IRRIGATION_TABLE['y'] == lat) ]

    if (source_file.empty):
        raise RuntimeError(f"Unable to find polygon file for location ({lat},{lon})")

    source_file = get_data_root() + "/labels/labeled_surveys/random_sample/processed/" + source_file.iloc[0].source_file + ".geojson"
    return source_file

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

def retrieve_polygons(irrigation_geojson, image_meta, timestamp, certainty_thresh=4):
    """
    Retrieve polygons corresponding to a particular .tif image.

    Parameters:
        - irrigation_geojson (str): Path of GeoJSON file that corresponds to the particular image 
        we are working with.
        - image_meta (dict): Metadata of particular .tif image we are working with.
        - timestamp (Date): Date
        - certainty_thresh (int): Minimum certainty for a polygon to be considered irrigated.

    Output: 
        - gdf (geopandas.geodataframe.GeoDataFrame): DataFrame that corresponds to the polygons 
        for the particular image.
    """
    gdf = gpd.read_file(irrigation_geojson)
    gdf = gdf.set_crs(image_meta['crs'], allow_override=True)

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
    shapes = [(geom, 1) for geom in gdf.geometry]
    label_array = rasterize(
        shapes=shapes,
        out_shape=(image_meta['height'], image_meta['width']),
        transform=image_meta['transform'],
        fill=0,
        dtype='uint8'
    )
    return label_array

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
        "count": 1
    })

    data = LabelTif(label_array, label_meta)
    save_data(data, output_label_path)

if __name__ == "__main__":
    path = get_data_root() + "/dataset/images"
    create_all_labels(path)