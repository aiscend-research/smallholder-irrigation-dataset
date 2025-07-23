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
from src.utils.utils import load_config, find_project_root
from src.utils.geometries import get_bounding_box
from s2cloudless.cloud_detector import S2PixelCloudDetector

import ee
NO_DATA = -9999

# Set up proxy if needed (adjust protocol if necessary)
# os.environ["HTTP_PROXY"] = "socks5://127.0.0.1:33210"
# os.environ["HTTPS_PROXY"] = "socks5://127.0.0.1:33210"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Initialization
def initialize_earthengine():
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
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;_-")
    return ''.join([c if c in allowed else '_' for c in desc])[:95]

def download_sentinel2_mosaic(lat, lon, start_date, end_date, output_prefix=None, bands=None):
    '''
    Downloads a mosaic image over the specified location and time range.

    Params
        - lat, lon (float): The latitude and longitude of the location of interest
        - start_date, end_date (Date): The start and end dates of the desired time range
        - output_prefix (str): Prefix for the output file
        - bands (array): List of bands to include in the mosaic

    Returns:
        - task (Earth Engine Task): Earth Engine task object that downloads the mosaic image
        - output_prefix (str): The prefix of the output file
    '''
    print(f"Attempting to download images (if found) for {output_prefix}")
    config = load_config()
    bucket = config["earthengine"]["bucket_name"]
    region = get_bounding_box(lat, lon)

    if bands is None:
        bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']

    collection = ee.ImageCollection("COPERNICUS/S2_HARMONIZED") \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
        .select(bands)

    if collection.size().getInfo() == 0:
        return None, output_prefix

    mosaic = collection.mosaic()
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
    while task.active():
        time.sleep(interval)
    return task.status()

def resize_img(img, target_shape):
    return np.stack([resize(img[b], target_shape[1:], preserve_range=True) for b in range(img.shape[0])], axis=0).astype(img.dtype)

# Retrieve ndvi, evi, and ndwi bands
def calculate_indices(img):
    img = img.astype(np.float32) / 10000.0
    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = img
    ndvi = np.where((B8 + B4) != 0, (B8 - B4) / (B8 + B4), NO_DATA)
    evi = np.where((B8 + 6*B4 - 7.5*B2 + 1) != 0, 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1), NO_DATA)
    ndwi = np.where((B8 + B11) != 0, (B8 - B11) / (B8 + B11), NO_DATA)
    return ndvi, evi, ndwi

def retrieve_images():
    '''
    For each labeled image, downloads its corresponding time series (36 images, with 18 images before the label date
    and 18 images after the label date) into ~/data/features/
    
    To do this, we attempt to retrieve images from the Google Cloud Bucket. If the image is not there, we request
    the image from Google Earth Engine, processes each image into the correct format (through calculation of 
    vegetation bands, use of cloud mask), add it to Google Cloud Bucket, and retrieve it from there.
    '''

    # Configuration
    logging.basicConfig(level=logging.INFO)
    LABEL_CSV = os.path.join(project_root, "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv")
    DOWNLOAD_DIR = os.path.join(project_root, "data/features/")
    TMP_DIR = os.path.join(DOWNLOAD_DIR, "_tmp_tif")
    os.makedirs(TMP_DIR, exist_ok=True)

    data = pd.read_csv(LABEL_CSV)
    initialize_earthengine()
    config = load_config()
    bucket = config["earthengine"]["bucket_name"]
    ee_key = os.path.join(project_root, config["earthengine"]["service_account_key"])
    fs = gcsfs.GCSFileSystem(token=ee_key, project="smallholder-irr")

    bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
    def_shape = (len(bands), 100, 100)
    cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)

    # Main Loop
    for idx, row in data.iterrows():
        lat, lon = row['y'], row['x']
        uid = row['unique_id']
        date = datetime(int(row['year']), int(row['month']), int(row['day']))
        windows = get_dense_time_windows(date)
        site_id = f"site_{lat:.2f}_{lon:.2f}_{date.year}_{uid}"
        stack_list, meta_list = [], []

        # Retrieve all 37 images for this particular location & time
        for start, end in windows:
            s, e = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
            prefix = f"{site_id}/s2_{lat:.2f}_{lon:.2f}_{s}_{e}"
            tif_path = os.path.join(TMP_DIR, os.path.basename(prefix) + ".tif")

            # If tif file has already been added to the Google Cloud Bucket, retrieve it
            if fs.exists(f"{bucket}/{prefix}.tif"):
                fs.get(f"{bucket}/{prefix}.tif", tif_path)
            else:
                task, _ = download_sentinel2_mosaic(lat, lon, s, e, prefix, bands)
                
                # No image was found for the location at that time interval
                if task is None:
                    meta_list.append({
                        "date_range": [s, e],
                        "cloud_fraction": 1.0,
                        "mean_ndvi": NO_DATA,
                        "mean_evi": NO_DATA,
                        "mean_ndwi": NO_DATA 
                    })
                    # Create blank tif
                    img = np.full((13, 100, 100), NO_DATA)
                    stack_list.append(img)
                    continue

                wait_for_task(task)
                fs.get(f"{bucket}/{prefix}.tif", tif_path)

            # Read image
            with rasterio.open(tif_path) as src:
                img = src.read().astype(np.int16)  
            if img.shape != def_shape:
                img = resize_img(img, def_shape)

            # Resize
            img_rgb = np.moveaxis(img, 0, -1) / 10000.0
            img_batch = img_rgb[np.newaxis, ...]

            cloud_mask = cloud_detector.get_cloud_masks(img_batch)[0]
            combined_mask = cloud_mask.astype(bool)

            # Set cloudy pixels to -9999 (NO_DATA value)
            img[:, combined_mask] = NO_DATA

            # Add NDVI, EVI, NDWI bands
            ndvi, evi, ndwi = calculate_indices(img)
            img = np.concatenate(
                (img, ndvi[None, :, :], evi[None, :, :], ndwi[None, :, :]),
                axis=0
            )
            bands.extend(["NDVI", "EVI", "NDWI"])

            print(f"Resultant image shape {img.shape} – should be (13, 100, 100)")

            meta_list.append({
                "date_range": [s, e],
                "cloud_fraction": float(combined_mask.mean()),
                "mean_ndvi": float(ndvi[ndvi != NO_DATA].mean()) if np.any(ndvi != NO_DATA) else NO_DATA,
                "mean_evi": float(evi[evi != NO_DATA].mean()) if np.any(evi != NO_DATA) else NO_DATA,
                "mean_ndwi": float(ndwi[ndwi != NO_DATA].mean()) if np.any(ndwi != NO_DATA) else NO_DATA 
            })
            stack_list.append(img)

        # Process the images from 37 time steps
        stack_arr = np.stack(stack_list)
        T, B, H, W = stack_arr.shape
        reshaped = stack_arr.transpose(1, 0, 2, 3).reshape(T*B, H, W)

        # Download .tif and .json files.
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
                "year": date.year, "unique_id": uid,
                "bands": bands,
                "shape": list(stack_arr.shape),
                "windows": meta_list
            }, f, indent=2)

        logging.info(f"[DONE] Saved stack to {out_tif}, shape={stack_arr.shape}")

if __name__ == "__main__":
    retrieve_images()