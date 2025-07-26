# Sentinel-2 L1C Mosaic Downloader and Stacker with s2cloudless + Shadow Masking
# Output: stacked GeoTIFF (.tif) + metadata (.json)

import sys
import os
import pandas as pd
import time
import logging
import gcsfs
import json
import shutil
import numpy as np
from datetime import datetime, timedelta
import rasterio
from rasterio.transform import from_origin
from skimage.transform import resize
from s2cloudless.cloud_detector import S2PixelCloudDetector
import ee

# Set up proxy if needed (adjust protocol if necessary)
os.environ["HTTP_PROXY"] = "socks5://127.0.0.1:33210"
os.environ["HTTPS_PROXY"] = "socks5://127.0.0.1:33210"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.utils import load_config, find_project_root
from src.utils.geometries import get_ee_bounding_box 

# Constants
NO_DATA = -9999 # Value for missing/invalid data
BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12'] # All bands downloaded from Sentinel-2
FINAL_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12', 'NDVI', 'EVI', 'NDWI', 'SCL'] # All bands in final .tif
LABEL_CSV = os.path.join(project_root, "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv") 
DOWNLOAD_DIR = os.path.join(project_root, "data/features/") # Path to download
TMP_DIR = os.path.join(DOWNLOAD_DIR, "_tmp_tif")

config = load_config()
bucket = config["earthengine"]["bucket_name"]
ee_key = os.path.join(project_root, config["earthengine"]["service_account_key"])
fs = gcsfs.GCSFileSystem(token=ee_key, project="smallholder-irr")
def_shape = (len(BANDS), 100, 100)
cloud_detector = S2PixelCloudDetector(threshold=0.6, average_over=4, dilation_size=2)

# Pseudo-atmospheric correction (Dark Object Subtraction, DOS) 
def pseudo_atmospheric_correction(image, region):
    """
    Applies a simple Dark Object Subtraction (DOS) pseudo-atmospheric correction to the input image.
    Only bands B2, B3, B4, B8 are corrected, while other bands are left unchanged.
    Args:
        image (ee.Image): Sentinel-2 image in TOA reflectance.
        region (ee.Geometry): The geometry/region for reduction.
    Returns:
        ee.Image: Image with corrected bands (B2, B3, B4, B8) using DOS, other bands untouched.
    """
    dark_percentile = 1
    bands = ['B2', 'B3', 'B4', 'B8']
    # Calculate the 1st percentile (dark object) for each band over the provided region
    stats = image.reduceRegion(
        reducer=ee.Reducer.percentile([dark_percentile]),
        geometry=region,
        scale=20,
        maxPixels=1e8
    )
    # Apply DOS correction for selected bands
    def dos_correction(band):
        offset = ee.Number(stats.get(band))
        return image.select(band).subtract(offset).rename(band)
    corrected_bands = [dos_correction(b) for b in bands]
    # Keep all other bands as they are
    other_bands = [image.select(b) for b in image.bandNames().getInfo() if b not in bands]
    # Merge corrected and original bands
    merged = image.addBands(corrected_bands, overwrite=True)
    return merged

# Custom SCL band simulation
def add_custom_scene_classification(image):
    """
    Adds a custom Scene Classification Layer (SCL) to simulate L2A-like labels for Sentinel-2 L1C imagery.
    The classification schema is as follows:
      0 - No Data
      1 - Saturated/Defective
      2 - Dark Area Pixels
      3 - Cloud Shadow
      4 - Vegetation
      5 - Not Vegetated
      6 - Water
      7 - Unclassified
      8 - Cloud Medium Probability
      9 - Cloud High Probability
     10 - Thin Cirrus
     11 - Snow/Ice

    Args:
        image (ee.Image): Sentinel-2 image (TOA or pseudo-corrected).
    Returns:
        ee.Image: Image with an additional SCL band.
    """
    ndvi = image.normalizedDifference(['B8', 'B4']).rename("NDVI")
    ndwi = image.normalizedDifference(['B3', 'B11']).rename("NDWI")
    ndmi = image.normalizedDifference(['B8', 'B11']).rename("NDMI")
    ndsi = image.normalizedDifference(['B3', 'B11']).rename("NDSI")
    brightness = image.select(['B2', 'B3', 'B4']).reduce(ee.Reducer.sum()).rename("Brightness")

    # Estimate cloud and shadow probability
    cloud_score = image.expression(
        '(B2 + B3 + B4) / 3',
        {'B2': image.select('B2'), 'B3': image.select('B3'), 'B4': image.select('B4')}
    ).rename("CloudScore")

    shadow_score = ndmi.multiply(-1).rename("ShadowScore")

    # Define thresholds (tune if necessary)
    scl = ee.Image(0)  # default No Data
    scl = scl.where(brightness.lt(0.15), 2)  # Dark area
    scl = scl.where(ndwi.gt(0.2), 6)  # Water
    scl = scl.where(ndsi.gt(0.5), 11)  # Snow/Ice
    scl = scl.where(ndvi.gt(0.5), 4)  # Vegetation
    scl = scl.where(ndvi.lt(0.1).And(ndwi.lt(0.2)), 5)  # Not vegetated
    scl = scl.where(cloud_score.gt(0.8), 9)  # High-prob cloud
    scl = scl.where(cloud_score.gt(0.6).And(cloud_score.lte(0.8)), 8)  # Medium-prob cloud
    scl = scl.where(shadow_score.gt(0.25), 3)  # Cloud shadow
    scl = scl.where(ndmi.gt(0.3), 10)  # Cirrus proxy

    scl = scl.rename("SCL").toUint16()
    return image.addBands(scl)

# Initialization
def initialize_earthengine():
    """
    Initializes the Earth Engine Python API using service account credentials from the config file.
    """
    config = load_config()
    ee_key = os.path.join(find_project_root(os.getcwd()), config["earthengine"]["service_account_key"])
    with open(ee_key) as f:
        creds = json.load(f)
        service_email = creds['client_email']
    credentials = ee.ServiceAccountCredentials(service_email, ee_key)
    ee.Initialize(credentials)
    print("Earth Engine initialized.")

# Helpers
def sanitize_description(desc):
    """
    Sanitizes a description string for Earth Engine export tasks.
    Only allows alphanumerics, dots, commas, colons, semicolons, dashes, and underscores.
    Truncates to 95 characters.
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;_-")
    return ''.join([c if c in allowed else '_' for c in desc])[:95]

def download_sentinel2_mosaic(lat, lon, start_date, end_date, output_prefix=None, bands=BANDS):
    '''
    Downloads a mosaic image over the specified location and time range.

    Params
        - lat, lon (float): The latitude and longitude of the location of interest
        - start_date, end_date (Date): The start and end dates of the desired time range
        - output_prefix (str): Prefix for the output file
        - bands (array): List of bands to include in the mosaic. Default as constant BANDS

    Returns:
        - task (Earth Engine Task): Earth Engine task object that downloads the mosaic image
        - output_prefix (str): The prefix of the output file
    '''
    config = load_config()
    bucket = config["earthengine"]["bucket_name"]
    region = get_ee_bounding_box(lat, lon)

    collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
        .select(bands)

    if collection.size().getInfo() == 0:
        return None, output_prefix

    mosaic = collection.mosaic()
    # Pseudo-atmospheric correction and custom SCL
    mosaic = pseudo_atmospheric_correction(mosaic, region)
    mosaic = add_custom_scene_classification(mosaic)
    bands_export = bands + ['SCL']
    mosaic = mosaic.select(bands_export).toUint16()

    if not output_prefix:
        output_prefix = f"s2_mosaic_{lat}_{lon}_{start_date.replace('-', '')}"
    desc = sanitize_description(output_prefix)

    task = ee.batch.Export.image.toCloudStorage(
        image=mosaic,
        description=f"export_{desc}",
        bucket=bucket,
        fileNamePrefix=output_prefix,
        region=region.getInfo()['coordinates'],
        scale=10,
        crs="EPSG:32633",
        maxPixels=1e13
    )
    task.start()
    return task, output_prefix

def get_dense_time_windows(center_date):
    '''
    Retrieves the 18 timesteps (10 day intervals) immediately before the labeled image date,
    and 18 time steps immediately after the labeled image date.

    Param:
        - center_date (datetime.date or datetime.datetime): Date of the labeled image

    Returns:
        - windows (list of tuples, where each tuple contains 2 dates): The list of timesteps
    '''
    window_size = timedelta(days=10)
    total_windows = 37
    half_window = timedelta(days=5)
    half_series_days = (total_windows // 2) * window_size

    # Start of first window is 18 windows before window 19,
    # and 5 days before center_date (so center_date is in middle of window 19)
    series_start = center_date - half_series_days - half_window

    windows = []
    for i in range(total_windows):
        start = series_start + i * window_size
        end = start + window_size
        windows.append((start, end))

    return windows

def wait_for_task(task, interval=30):
    """
    Polls an Earth Engine task until it is complete.
    Args:
        task: Earth Engine batch task object
        interval: seconds between polling
    Returns:
        status: dict, Earth Engine task status
    """
    while task.active():
        time.sleep(interval)
    return task.status()

def resize_img(img, target_shape):
    """
    Resizes a multi-band image array to the given target shape.
    """
    return np.stack([resize(img[b], target_shape[1:], preserve_range=True) for b in range(img.shape[0])], axis=0).astype(img.dtype)

# Retrieve ndvi, evi, and ndwi bands
def calculate_indices(img):
    """
    Calculate NDVI, EVI, NDWI from the first 10 Sentinel-2 bands.
    Returns int16 arrays scaled by 10000, with NO_DATA for missing values.

    Args:
        img: numpy array, shape (>=10, H, W)

    Returns:
        ndvi, evi, ndwi: int16 numpy arrays, each shape (H, W)
    """
    img = img.astype(np.float32) / 10000.0
    # Always use the first 10 bands
    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = img[:10]
    ndvi = np.full(B2.shape, NO_DATA, dtype=np.int16)
    evi = np.full(B2.shape, NO_DATA, dtype=np.int16)
    ndwi = np.full(B2.shape, NO_DATA, dtype=np.int16)

    valid_ndvi = (B8 + B4) != 0
    valid_evi = (B8 + 6*B4 - 7.5*B2 + 1) != 0
    valid_ndwi = (B8 + B11) != 0

    ndvi[valid_ndvi] = ((B8 - B4)[valid_ndvi] / (B8 + B4)[valid_ndvi] * 10000).astype(np.int16)
    evi[valid_evi] = (2.5 * (B8 - B4)[valid_evi] / (B8 + 6*B4 - 7.5*B2 + 1)[valid_evi] * 10000).astype(np.int16)
    ndwi[valid_ndwi] = ((B8 - B11)[valid_ndwi] / (B8 + B11)[valid_ndwi] * 10000).astype(np.int16)
    return ndvi, evi, ndwi

def retrieve_time_series_stack(site_id, lat, lon, date):
    """
    Retrieves the time series of Sentinel-2 mosaics for a given site and computes indices and cloud masking.
    Args:
        site_id: string
        lat, lon: float, coordinates
        date: datetime.date or datetime.datetime, center date

    Returns:
        stack_list: list of numpy arrays, each (14, 100, 100)
        meta_list: list of metadata dictionaries for each time window
        empty_window_count (int): count of windows without images
    """
    windows = get_dense_time_windows(date)
    stack_list, meta_list = [], []
    empty_window_count = 0

    for start, end in windows:
        s, e = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
        prefix = f"{site_id}/s2_{lat:.2f}_{lon:.2f}_{s}_{e}"
        logging.info(f"Fetching image for {prefix}")
        tif_path = os.path.join(TMP_DIR, os.path.basename(prefix) + ".tif")

        if fs.exists(f"{bucket}/{prefix}.tif"):
            fs.get(f"{bucket}/{prefix}.tif", tif_path)
        else:
            task, _ = download_sentinel2_mosaic(lat, lon, s, e, prefix)
            if task is None:
                meta_list.append({
                    "date_range": [s, e],
                    "cloud_fraction": 1.0,
                    "mean_ndvi": NO_DATA,
                    "mean_evi": NO_DATA,
                    "mean_ndwi": NO_DATA 
                })
                img = np.full((len(FINAL_BANDS), 100, 100), NO_DATA, dtype=np.int16)
                stack_list.append(img)
                empty_window_count += 1
                continue

            status = wait_for_task(task)
            if status['state'] != 'COMPLETED':
                logging.info(f"[ERROR] Export task failed or incomplete for {prefix}. Status: {status['state']}")
                if 'error_message' in status:
                    logging.info(f"Error message: {status['error_message']}")
                meta_list.append({
                    "date_range": [s, e],
                    "cloud_fraction": 1.0,
                    "mean_ndvi": NO_DATA,
                    "mean_evi": NO_DATA,
                    "mean_ndwi": NO_DATA
                })
                img = np.full((len(FINAL_BANDS), 100, 100), NO_DATA, dtype=np.int16)
                stack_list.append(img)
                empty_window_count += 1
                continue

            fs.get(f"{bucket}/{prefix}.tif", tif_path)

        scl_band = np.full((100, 100), NO_DATA, dtype=np.int16)
        with rasterio.open(tif_path) as src:
            img = src.read().astype(np.int16)
            scl_band = src.read(11).astype(np.int16)
        if img.shape != def_shape and img.shape[0] >= 10:
            img = resize_img(img[:10], def_shape)  # Only first 10 bands, exclude SCL
        if scl_band.shape != (100, 100):
            scl_band = resize(scl_band, (100, 100))

        ndvi, evi, ndwi = calculate_indices(img[:10])
        img_with_indices = np.concatenate(
            (img[:10], ndvi[None, :, :], evi[None, :, :], ndwi[None, :, :]),
            axis=0
        )

        img_rgb = np.moveaxis(img[:10], 0, -1) / 10000.0
        img_batch = img_rgb[np.newaxis, ...] # Shape for get_cloud_masks: (1, 100, 100, 10)

        try:
            cloud_mask = cloud_detector.get_cloud_masks(img_batch)[0]
            combined_mask = cloud_mask.astype(bool)
        except Exception as e:
            logging.error(f"Cloud mask failed for {prefix}: {e}")
            combined_mask = np.zeros(img.shape[1:], dtype=bool)

        # Add SCL band to the image
        # Convert all 0 values in SCL to NO_DATA
        scl_band[scl_band == 0] = NO_DATA
        img_with_indices = np.concatenate(
            (img_with_indices, scl_band[None, :, :]),
            axis=0
        )

        # Apply cloud mask to all bands
        img_with_indices[:, combined_mask] = NO_DATA

        meta_list.append({
            "date_range": [s, e],
            "cloud_fraction": float(combined_mask.mean()),
            "mean_ndvi": float(ndvi[ndvi != NO_DATA].mean()) if np.any(ndvi != NO_DATA) else NO_DATA,
            "mean_evi": float(evi[evi != NO_DATA].mean()) if np.any(evi != NO_DATA) else NO_DATA,
            "mean_ndwi": float(ndwi[ndwi != NO_DATA].mean()) if np.any(ndwi != NO_DATA) else NO_DATA 
        })
        stack_list.append(img_with_indices)

    return stack_list, meta_list, empty_window_count

def retrieve_images():
    '''
    For each labeled image, downloads its corresponding time series (36 images, with 18 images before the label date
    and 18 images after the label date) into ~/data/features/
    
    To do this, we attempt to retrieve images from the Google Cloud Bucket. If the image is not there, we request
    the image from Google Earth Engine, processes each image into the correct format (through calculation of 
    vegetation bands, use of cloud mask), add it to Google Cloud Bucket, and retrieve it from there.
    '''
    logging.basicConfig(level=logging.INFO)
    os.makedirs(TMP_DIR, exist_ok=True)
    data = pd.read_csv(LABEL_CSV)
    initialize_earthengine()
    for idx, row in data.iterrows():
        lat, lon = row['y'], row['x']
        uid = row['unique_id']
        date = datetime(int(row['year']), int(row['month']), int(row['day']))
        site_id = f"site_{lat:.2f}_{lon:.2f}_{date.year}_{uid}"
        stack_list, meta_list, empty_window_count = retrieve_time_series_stack(site_id, lat, lon, date)
        stack_arr = np.stack(stack_list)
        T, B, H, W = stack_arr.shape
        expected_shape = (37, len(FINAL_BANDS), 100, 100)
        if stack_arr.shape != expected_shape:
            raise ValueError(
                f"Output stack shape {stack_arr.shape} does not match expected {expected_shape} "
                f"(T={T}, B={B}, H={H}, W={W})"
            )
        reshaped = stack_arr.transpose(1, 0, 2, 3).reshape(T*B, H, W)
        out_tif = os.path.join(DOWNLOAD_DIR, f"{site_id}.tif")
        out_json = os.path.join(DOWNLOAD_DIR, f"{site_id}.json")

        with rasterio.open(out_tif, 'w', driver='GTiff', height=H, width=W, count=T*B,
                        dtype='int16', crs='EPSG:32633',
                        transform=from_origin(lon - 0.0005, lat + 0.0005, 0.0001, 0.0001),
                        nodata=NO_DATA) as dst:
            dst.write(reshaped.astype('int16'))

        with open(out_json, 'w') as f:
            json.dump({
                "site_id": site_id,
                "lat": lat, "lon": lon,
                "year": date.year, 
                "unique_id": uid,
                "bands": FINAL_BANDS,
                "shape": list(stack_arr.shape),
                "empty_window_count": empty_window_count,
                "windows": meta_list
            }, f, indent=2)

        logging.info(f"[DONE] Saved stack to {out_tif}, shape={stack_arr.shape}")

if __name__ == "__main__":
    retrieve_images()