#!/usr/bin/env python3

import sys, os, json, time, logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from skimage.transform import resize
from skimage.morphology import binary_dilation, footprint_rectangle
import gcsfs
import ee
import requests

S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']

# project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.utils import load_config, find_project_root
from src.utils.geometries import get_ee_bounding_box

config = load_config()

def initialize_earthengine():
    key_path = os.path.join(find_project_root(os.getcwd()), config["earthengine"]["service_account_key"])
    with open(key_path) as f:
        creds = json.load(f)
        service_email = creds['client_email']
    credentials = ee.ServiceAccountCredentials(service_email, key_path)
    ee.Initialize(credentials)
    logging.info("Earth Engine initialized.")

def s2_image_exporter(lat: float, lon: float, start_date: str, end_date: str, file_name: str, out_dir: str, collection: str = "L1C"):
    """
    Download the best quality Sentinel-2 image for a location and time window.
    
    Retrieves a 1km × 1km (100×100 pixels at 10m resolution) Sentinel-2 image centered 
    on the given coordinates, selecting the image with maximum good-quality pixel coverage
    after applying collection-specific quality filtering.
    
    Quality Masking Strategy (GEE no data value = 0):
        - L2A: Uses Scene Classification Layer (SCL) to mask clouds (classes 8,9,10), 
          cloud shadows (class 3), saturated pixels (class 1), and no data (class 0).
          Keeps vegetation, bare soil, water, unclassified, and snow pixels.
        
        - L1C: Uses QA60 band to identify clouds (bit 10) and cirrus (bit 11).
          Pre-filters to exclude ANY image containing opaque clouds in the region
          (to avoid undetectable cloud shadows), then masks remaining cirrus pixels.
    
    Args:
        lat: Latitude of region center (WGS84 decimal degrees)
        lon: Longitude of region center (WGS84 decimal degrees)
        start_date: Start of date range in 'YYYY-MM-DD' format
        end_date: End of date range in 'YYYY-MM-DD' format
        file_name: Output filename (e.g., 's2_image.tif')
        out_dir: Directory path where file will be saved (created if doesn't exist)
        collection: Either 'L1C' (Top-of-Atmosphere) or 'L2A' (Surface Reflectance).
                   Default is 'L1C'.

    Returns:
        None: Always returns None. Check logs for success/failure status.
        Output file is saved to {out_dir}/{file_name} if successful.

    Example:
        >>> s2_image_exporter(
        ...     lat=37.5, 
        ...     lon=-120.5,
        ...     start_date='2023-06-01',
        ...     end_date='2023-08-31',
        ...     file_name='field_123.tif',
        ...     out_dir='./data/images',
        ...     collection='L2A'
        ... )
        # Creates ./data/images/field_123.tif
    """

    # Define region and date range
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(500).bounds()

    # Retieve the best image for the given collection and date range, masking out poor quality pixels. 

    # For L2A data, pick the least cloudy image 
    # and mask out poor quality pixels (cloud shadows, clouds, cirrus, reflectance saturation) using SCL (scene classification) band. 
    # For L1C data, there is no SCL so we only know where there are clouds and cirrus pixels using QA60.
    # Therefore, we not only choose the image with the fewest poor quality pixels,
    # but we first toss any image with any opaque clouds present at all

    assert collection in ["L1C", "L2A"]

    if collection == "L2A":
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(ee.Date(start_date), ee.Date(end_date)))
        
        # Define quality mask for L2A
        def get_quality_mask(img):
            scl = img.select('SCL')
            return scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11)) # vegetation, bare soil, water, unclassified, snow

    elif collection == "L1C":
        col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(region)
            .filterDate(ee.Date(start_date), ee.Date(end_date)))

        # Additional step for L1C:
        # Remove any images that have any clouds at all, 
        # since we have no way of knowing which pixels have cloud shadow
        def check_clouds(img):
            qa60 = img.select('QA60')
            clouds = qa60.bitwiseAnd(1 << 10).neq(0)
            
            # Check if ANY cloud pixels exist in region
            result = clouds.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=region,
                scale=60,
                maxPixels=1e9
            )
            
            # Get the value, default to 0 if None
            cloud_pixels = ee.Number(result.get('QA60', 0))
            has_clouds = cloud_pixels.gt(0)
            
            return img.set('has_clouds', has_clouds)
        
        col = col.map(check_clouds).filter(ee.Filter.eq('has_clouds', ee.Number(0)))
        
        # Define quality mask for L1C
        def get_quality_mask(img):
            qa60 = img.select('QA60')
            return qa60.bitwiseAnd(1 << 10).Or(qa60.bitwiseAnd(1 << 11)).Not() # not cloud or cirrus

    # Common code for both
    # Note if you found no images
    if col.size().getInfo() == 0:
        logging.warning(f"No images found for {lat},{lon} between {start_date} and {end_date}")
        return None

    # Score and pick best
    def score_image(img):
        quality_mask = get_quality_mask(img)
        good_count = quality_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=60,
            maxPixels=1e8
        ).values().get(0)
        return img.set('good_pixel_count', good_count)

    best = col.map(score_image).sort('good_pixel_count', False).first()

    # Apply mask
    quality_mask = get_quality_mask(best)
    masked = best.updateMask(quality_mask).select(S2_BANDS)

    # HTTP download locally
    if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, file_name)

    # save image
    url = masked.getDownloadURL({
        'region': region, 
        'scale': 10, 
        'format': 'GeoTIFF'
    })

    try:
        resp = requests.get(url)
        resp.raise_for_status()
        with open(output_path, 'wb') as f: f.write(resp.content)
        logging.info(f"Successfully downloaded: {output_path}")
        return None
    except Exception as e:
        logging.error(f"Failed to download masked image for {file_name}: {e}")
        return None

def retrieve_time_series_stack(site_id: str, lat: float, lon: float, date: datetime, out_dir: str, start_month: int=1, num_windows: int=36, timestep: int=10, window_buffer: int=3):
    """
    Download and stack Sentinel-2 images over a time series, save result, and clean up.
    
    Args:
        site_id: Unique identifier for this site
        lat, lon: Coordinates
        date: Reference date (year used for time series)
        out_dir: Directory to save final outputs
        start_month: Month to start downloading (1=January)
        num_windows: Number of timesteps to download
        timestep: Days per timestep
        window_buffer: Extra timesteps before/after for augmentation
    
    Saves:
        {site_id}_stack.tif - Stacked images (num_windows, num_bands, H, W)
        {site_id}_metadata.json - Metadata about the stack
    """

    # Create time windows
    step_size = timedelta(days=timestep)
    num_windows = num_windows + (window_buffer * 2)
    start_date = datetime(date.year, start_month, 1) - (step_size * window_buffer)
    time_windows = [(start_date + i * step_size, start_date + (i + 1) * step_size) for i in range(num_windows)]

    # Create temporary directory for individual downloads
    temp_dir = os.path.join(out_dir, "_tmp", site_id)
    os.makedirs(temp_dir, exist_ok=True)

    # Helper function for filenames
    def get_file_path(start, end):
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        file_name = f"s2_{lat:.2f}_{lon:.2f}_{start_str}_{end_str}_masked.tif"
        return os.path.join(temp_dir, file_name), start_str, end_str

    # Download images in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        
        for start, end in time_windows:
            file_path, start_str, end_str = get_file_path(start, end)
            
            if not os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                futures.append(executor.submit(s2_image_exporter, lat, lon, start_str, end_str, file_name, temp_dir))
        
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logging.error(f"Error during export: {e}")

    # Load all images
    images = []
    metadata_windows = []
    
    for start, end in time_windows:
        file_path, start_str, end_str = get_file_path(start, end)
        
        if os.path.exists(file_path):
            try:
                with rasterio.open(file_path) as src:
                    img = src.read()
                    images.append(img)
                    
                    nodata_val = src.nodata if src.nodata is not None else 0
                    nodata_mask = (img == nodata_val).any(axis=0)
                    nodata_fraction = nodata_mask.sum() / (img.shape[1] * img.shape[2])
                    
                    metadata_windows.append({
                        'date_range': [start_str, end_str],
                        'file_exists': True,
                        'masked_fraction': float(nodata_fraction)
                    })
            except Exception as e:
                logging.error(f"Failed to read {file_path}: {e}")
                images.append(None)
                metadata_windows.append({
                    'date_range': [start_str, end_str],
                    'file_exists': False,
                    'masked_fraction': 1.0
                })
        else:
            images.append(None)
            metadata_windows.append({
                'date_range': [start_str, end_str],
                'file_exists': False,
                'masked_fraction': 1.0
            })

    # Find template image
    template_img = None
    template_nodata = 0
    template_crs = None
    template_transform = None
    
    for i, img in enumerate(images):
        if img is not None:
            template_img = img
            file_path, _, _ = get_file_path(*time_windows[i])
            with rasterio.open(file_path) as src:
                template_nodata = src.nodata if src.nodata is not None else 0
                template_crs = src.crs
                template_transform = src.transform
            break
    
    if template_img is None:
        raise ValueError(f"No valid images found for site {site_id}")
    
    template_shape = template_img.shape
    empty_template = np.full(template_shape, template_nodata, dtype=template_img.dtype)
    
    # Build stack
    stack = []
    for img in images:
        if img is None:
            stack.append(empty_template.copy())
        else:
            if img.shape != template_shape:
                logging.warning(f"Image shape mismatch: {img.shape} vs {template_shape}")
                stack.append(empty_template.copy())
            else:
                stack.append(img)
    
    stacked_array = np.stack(stack, axis=0)
    
    # Reshape for saving: (num_windows, num_bands, H, W) -> (num_windows*num_bands, H, W)
    T, B, H, W = stacked_array.shape
    reshaped = stacked_array.transpose(1, 0, 2, 3).reshape(T * B, H, W)
    
    # Save stacked image
    stack_file = os.path.join(out_dir, f"{site_id}_stack.tif")
    with rasterio.open(
        stack_file, 'w',
        driver='GTiff',
        height=H, width=W,
        count=T * B,
        dtype=template_img.dtype,
        crs=template_crs,
        transform=template_transform,
        nodata=template_nodata
    ) as dst:
        dst.write(reshaped)
    
    # Save metadata
    metadata = {
        'site_id': site_id,
        'lat': float(lat),
        'lon': float(lon),
        'year': int(date.year),
        'bands': S2_BANDS,
        'shape': list(stacked_array.shape),
        'num_windows': num_windows,
        'timestep_days': timestep,
        'start_month': start_month,
        'windows': metadata_windows
    }
    
    metadata_file = os.path.join(out_dir, f"{site_id}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    logging.info(f"Saved stack: {stack_file}")
    logging.info(f"Saved metadata: {metadata_file}")
    logging.info(f"Final shape: {stacked_array.shape}")
    logging.info(f"Cleaned up temp directory: {temp_dir}")


def process_row(row, out_dir):
    """Process one row from the CSV - download and stack images for this site."""
    
    site_id = f"site_{row['unique_id']}"
    lat, lon = row['y'], row['x']
    date = datetime(int(row['year']), int(row['month']), int(row['day']))
    
    logging.info(f"Processing {site_id} at ({lat}, {lon})")
    
    retrieve_time_series_stack(
        site_id=site_id,
        lat=lat,
        lon=lon,
        date=date,
        out_dir=out_dir
    )
    
    logging.info(f"Completed {site_id}")