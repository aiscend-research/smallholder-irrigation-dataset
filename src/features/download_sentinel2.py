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
    
    Quality masking (GEE no data value is 0):

    L2A:
    - Select the least cloudy image.
    - Mask cloud shadows, clouds, cirrus, and saturated pixels using the SCL band.
    - Keep only valid surface classes.

    L1C:
    - No SCL available, so only clouds and cirrus can be detected from QA60.
    - Use a custom detector (not very precise) to find cloudy pixels and buffer them by 2 km to also remove likely cloud shadows.
    
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

        def get_quality_mask(img):
            # 1. Get QA60 clouds and cirrus (bad pixels)
            qa60 = img.select('QA60')
            qa60_clouds = qa60.bitwiseAnd(1 << 10).neq(0)
            qa60_cirrus = qa60.bitwiseAnd(1 << 11).neq(0)
            
            # 2. Get custom cloud mask (bad pixels)
            custom_clouds = create_l1c_cloud_mask(img)
            
            # 3. Resample custom clouds to QA60 resolution
            custom_clouds_60m = custom_clouds.reproject(qa60_clouds.projection())
            
            # 4. Combine ALL bad pixels
            bad_pixels = qa60_clouds.Or(qa60_cirrus).Or(custom_clouds_60m.eq(1))
            
            # 5. Reduce to 2km resolution - if ANY pixel in 1km area is bad, whole area is bad
            bad_pixels_coarse = bad_pixels.reduceResolution(
                reducer=ee.Reducer.max(),  # Max = if any pixel is 1 (cloud), result is 1
                maxPixels=2048
            ).reproject(
                crs=qa60_clouds.projection().crs(),
                scale=2000  # 2km resolution
            )
            
            # 6. Resample back to 60m resolution
            bad_pixels_buffered = bad_pixels_coarse.reproject(
                crs=qa60_clouds.projection().crs(),
                scale=60
            )

            # 7. Invert to get good pixels (mask)
            good_pixels = bad_pixels_buffered.Not()
            
            return good_pixels.rename('cloud_mask')

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

def create_l1c_cloud_mask(img: ee.Image, 
                          cloud_threshold: float = 0.55,
                          veg_ndvi_threshold: float = 0.55) -> ee.Image:
    """
    Create a pixel-by-pixel cloud mask for L1C imagery.
    
    Args:
        img: Sentinel-2 L1C image
        cloud_threshold: Cloud probability threshold (0-1). Lower = stricter. Default 0.3
        veg_ndvi_threshold: NDVI threshold for vegetation guard. Higher = less conservative. Default 0.55
    
    Returns:
        ee.Image: Binary mask where 1 = cloud, 0 = clear
    """
    
    # Normalize bands to 0-1
    blue = img.select('B2').divide(10000)
    green = img.select('B3').divide(10000)
    red = img.select('B4').divide(10000)
    nir = img.select('B8').divide(10000)
    swir = img.select('B11').divide(10000)
    
    # Calculate NDVI
    ndvi = nir.subtract(red).divide(nir.add(red).add(1e-6))
    
    # Calculate standard deviation of RGB (clouds are uniform/white)
    vis = ee.Image.cat([blue, green, red])
    vis_std = vis.reduce(ee.Reducer.stdDev())
    
    # 1. Brightness indicator
    vis_mean = img.select(['B2','B3','B4']).divide(10000).reduce(ee.Reducer.mean())
    p_bright = vis_mean.subtract(0.30).divide(0.15).clamp(0, 1)
    
    # 2. SWIR indicator (clouds are dark in SWIR)
    p_swir = swir.subtract(0.12).divide(0.10).clamp(0, 1)
    
    # 3. Blue/SWIR ratio (clouds have high ratio)
    p_ratio = blue.divide(swir.add(1e-6)).subtract(1.2).divide(0.5).clamp(0, 1)
    
    # 4. Whiteness (low std = uniform white color = cloud)
    p_white = ee.Image(1).subtract(vis_std.divide(0.06).clamp(0, 1))
    
    # Combine indicators
    cloud_prob = (p_bright.multiply(0.4)
                  .add(p_swir.multiply(0.2))
                  .add(p_ratio.multiply(0.2))
                  .add(p_white.multiply(0.2)))
    
    # Vegetation guard
    veg_guard = ndvi.lte(veg_ndvi_threshold)
    
    # Apply threshold (tunable!)
    is_cloud = cloud_prob.gt(cloud_threshold).And(veg_guard)
    
    return is_cloud.toUint8().rename('cloud_mask')


def retrieve_time_series_stack(
    site_id: str, 
    lat: float, 
    lon: float, 
    date: datetime, 
    out_dir: str, 
    collection: str="L1C",
    start_month: int=1, 
    num_windows: int=36, 
    timestep: int=10, 
    window_buffer: int=3):
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
    num_windows_buffered = num_windows + (window_buffer * 2)
    start_date = datetime(date.year, start_month, 1) - (step_size * window_buffer)
    time_windows = [(start_date + i * step_size, start_date + (i + 1) * step_size) for i in range(num_windows_buffered)]

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
                futures.append(executor.submit(s2_image_exporter, lat, lon, start_str, end_str, file_name, temp_dir, collection))
        
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
    
    # Reshape for saving: (num_windows_buffered, num_bands, H, W) -> (num_windows_buffered*num_bands, H, W)
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
        'num_windows_buffered': num_windows_buffered,
        'timestep_days': timestep,
        'start_month_unbuffered': start_month,
        'window_buffer': 3,
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

if __name__ == '__main__':

    initialize_earthengine()

    TEST_OUT_DIR = './test_timeseries_output_L1C'

    # Test with a known good location
    test_site_id = "test_site_001"
    test_lat = 37.5
    test_lon = -120.5
    test_date = datetime(2023, 6, 15)

    retrieve_time_series_stack(
        site_id=test_site_id,
        lat=test_lat,
        lon=test_lon,
        date=test_date,
        collection="L1C",
        out_dir=TEST_OUT_DIR,
        start_month=1,
        num_windows=12,  # Use fewer windows for faster testing
        timestep=10,
        window_buffer=1
    )