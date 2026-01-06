#!/usr/bin/env python3

import sys, os, json, logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import rasterio

import pandas as pd
import numpy as np
import rasterio
import ee
import requests

import numpy as np
import rasterio
import ee
import requests

S2_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']

# project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.utils import load_config, find_project_root, get_data_root

config = load_config()

def initialize_earthengine():
    """Initialize Earth Engine with service account credentials."""
    key_path = os.path.join(find_project_root(os.getcwd()), config["earthengine"]["service_account_key"])
    with open(key_path) as f:
        creds = json.load(f)
        service_email = creds['client_email']
    credentials = ee.ServiceAccountCredentials(service_email, key_path)
    ee.Initialize(credentials)
    logging.info("Earth Engine initialized.")

#############################
# Download a single S2 scene
#############################

def s2_image_exporter(lat: float, lon: float, start_date: str, end_date: str, 
                     file_name: str, out_dir: str, collection: str = "L1C"):
    """
    Download best quality Sentinel-2 image for a time window.
    
    Selects the image with the most good-quality pixels after masking clouds,
    cloud shadows, and other bad pixels. Downloads a 1km × 1km region (100×100 
    pixels at 10m resolution) centered on the coordinates.
    
    Always downloads both unmasked and masked versions.
    
    Args:
        lat: Latitude in decimal degrees (WGS84)
        lon: Longitude in decimal degrees (WGS84)
        start_date: Start date as 'YYYY-MM-DD'
        end_date: End date as 'YYYY-MM-DD'
        file_name: Output filename WITHOUT extension (e.g., 's2_image')
        out_dir: Output directory (created if doesn't exist)
        collection: 'L1C' (Top-of-Atmosphere) or 'L2A' (Surface Reflectance)
    
    Returns:
        True if successful, False otherwise. Saves files to:
            - {out_dir}/{file_name}.tif (unmasked - all pixels)
            - {out_dir}/{file_name}_masked.tif (masked - bad pixels set to 0)
    
    Example:
        >>> s2_image_exporter(
        ...     lat=37.5, 
        ...     lon=-120.5,
        ...     start_date='2023-06-01',
        ...     end_date='2023-08-31',
        ...     file_name='field_123',
        ...     out_dir='./data/images',
        ...     collection='L2A'
        ... )
        # Creates:
        #   ./data/images/field_123.tif
        #   ./data/images/field_123_masked.tif
    
    Notes:
        - L2A: Uses SCL band for high-quality masking (clouds, shadows, etc.)
        - L1C: Uses QA60 + custom cloud detector with 2km buffer for shadows
        - GEE nodata value is 0
    """

    # Define region and date range
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(500).bounds()

    # Retrieve the best image for the given collection and date range
    assert collection in ["L1C", "L2A"]

    if collection == "L2A":
        col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(ee.Date(start_date), ee.Date(end_date)))

    elif collection == "L1C":
        col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
            .filterBounds(region)
            .filterDate(ee.Date(start_date), ee.Date(end_date)))

    # Note if you found no images
    if col.size().getInfo() == 0:
        logging.warning(f"No images found for {lat},{lon} between {start_date} and {end_date}")
        return False

    # Score and pick best
    def score_image(img):
        quality_mask = get_quality_mask(img, collection)
        good_count = quality_mask.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=60,
            maxPixels=1e8
        ).values().get(0)
        return img.set('good_pixel_count', good_count)

    best = col.map(score_image).sort('good_pixel_count', False).first()

    # Download unmasked version
    unmasked = best.select(S2_BANDS)
    success = download_ee_image(unmasked, region, f"{file_name}.tif", out_dir, scale=10)
    if not success:
        return False

    # Download masked version
    quality_mask = get_quality_mask(best, collection)
    masked = best.updateMask(quality_mask).select(S2_BANDS)
    success = download_ee_image(masked, region, f"{file_name}_masked.tif", out_dir, scale=10)
    if not success:
        return False

    return True

# Helper functions

def download_ee_image(image: ee.Image, region: ee.Geometry, 
                     file_name: str, out_dir: str, scale: int = 10) -> bool:
    """
    Download Earth Engine image to local file.
    
    Args:
        image: Earth Engine image to download
        region: Geographic region to download
        file_name: Output filename (including extension, e.g., 'image.tif')
        out_dir: Output directory
        scale: Pixel resolution in meters (default 10m)
    
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, file_name)
    
    url = image.getDownloadURL({
        'region': region,
        'scale': scale,
        'format': 'GeoTIFF'
    })
    
    try:
        resp = requests.get(url, timeout=300)  # 5 min timeout
        resp.raise_for_status()
        with open(output_path, 'wb') as f:
            f.write(resp.content)
        logging.info(f"Downloaded: {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to download {output_path}: {e}")
        return False

def get_quality_mask(img, collection):
    """
    Create quality mask for Sentinel-2 image.
    
    Args:
        img: Earth Engine image (S2_HARMONIZED or S2_SR_HARMONIZED)
        collection: 'L1C' or 'L2A'
    
    Returns:
        ee.Image: Binary mask where 1 = good pixel, 0 = bad pixel
        
    L2A approach:
        - Uses Scene Classification Layer (SCL)
        - Keeps: vegetation(4), bare soil(5), water(6), unclassified(7), snow(11)
        
    L1C approach:
        - Uses QA60 (clouds bit 10, cirrus bit 11)
        - Uses custom cloud detector (brightness + SWIR indicators)
        - Applies 2km buffer to catch cloud shadows
    """

    assert collection in ["L1C", "L2A"]

    if collection == "L2A":
        scl = img.select('SCL')
        return scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11)) # vegetation, bare soil, water, unclassified, snow

    elif collection == "L1C":
        # 1. Get QA60 clouds and cirrus (bad pixels)
        qa60 = img.select('QA60')
        qa60_clouds = qa60.bitwiseAnd(1 << 10).neq(0)
        qa60_cirrus = qa60.bitwiseAnd(1 << 11).neq(0)
        
        # 2. Get custom cloud mask (bad pixels)
        custom_clouds = custom_l1c_cloud_mask(img)
        
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

def custom_l1c_cloud_mask(img: ee.Image, 
                          cloud_threshold: float = 0.55,
                          veg_ndvi_threshold: float = 0.55) -> ee.Image:
    """
    Detect clouds in L1C imagery using spectral indicators.
    
    Combines brightness, SWIR darkness, blue/SWIR ratio, and color uniformity
    to identify cloud pixels. More lenient than QA60 alone but less accurate
    than L2A's SCL band.
    
    Args:
        img: Sentinel-2 L1C image
        cloud_threshold: Cloud probability threshold (0-1). Lower = stricter
        veg_ndvi_threshold: NDVI threshold for vegetation guard (pixels with 
                           NDVI > this won't be flagged as clouds)
    
    Returns:
        ee.Image: Binary mask where 1 = cloud, 0 = clear
        
    Method:
        - Brightness: Bright in visible bands
        - SWIR: Dark in SWIR (clouds don't reflect SWIR well)
        - Ratio: High blue/SWIR ratio
        - Whiteness: Low color variation (uniform white = cloud)
        - Guard: Skips pixels with high NDVI (vegetation)
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

#################################
# Download a full time series 
#################################

def retrieve_time_series_stack(
    file_id: str, 
    lat: float, 
    lon: float, 
    date: datetime, 
    out_dir: str, 
    collection: str="L1C",
    start_month: int=1, 
    num_windows: int=36, 
    timestep: int=10, 
    window_buffer: int=3,
    target_size: int=100):
    """
    Download and stack Sentinel-2 images over a time series.
    
    Creates two stacks: unmasked (all pixels) and masked (bad pixels = 0).
    All images are trimmed or padded to a fixed size (target_size × target_size pixels).
    Missing images are filled with nodata. Cleans up temporary files after stacking.
    
    Args:
        file_id: Unique identifier for this site
        lat, lon: Coordinates (WGS84 decimal degrees)
        date: Reference date (year used for time series)
        out_dir: Directory to save final outputs
        collection: 'L1C' or 'L2A'
        start_month: Month to start downloading (1=January, excluding buffer)
        num_windows: Number of timesteps to download (excluding buffer)
        timestep: Days per timestep
        window_buffer: Extra timesteps before/after for augmentation
        target_size: Size in pixels for all images (default 100, i.e., 100×100)
    
    Returns:
        None. Saves files to:
            - {out_dir}/{file_id}_stack.tif - Unmasked stack
            - {out_dir}/{file_id}_stack_masked.tif - Masked stack
            - {out_dir}/{file_id}_metadata.json - Metadata for both
    
    Output shape:
        Stack: (num_windows_buffered, num_bands, target_size, target_size)
        Saved as: (num_windows_buffered * num_bands, target_size, target_size)
    """

    # Create time windows
    step_size = timedelta(days=timestep)
    num_windows_buffered = num_windows + (window_buffer * 2)
    start_date = datetime(date.year, start_month, 1) - (step_size * window_buffer)
    time_windows = [(start_date + i * step_size, start_date + (i + 1) * step_size) 
                    for i in range(num_windows_buffered)]

    # Create temporary directory for individual downloads
    temp_dir = os.path.join(out_dir, "_tmp", file_id)
    os.makedirs(temp_dir, exist_ok=True)

    # Helper function for filenames
    def get_file_paths(start, end):
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        file_name = f"s2_{lat:.2f}_{lon:.2f}_{start_str}_{end_str}"
        base_path = os.path.join(temp_dir, file_name)
        return base_path + ".tif", base_path + "_masked.tif", start_str, end_str

    # Download images in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        
        for start, end in time_windows:
            unmasked_path, masked_path, start_str, end_str = get_file_paths(start, end)
            
            # Check if either file is missing
            if not (os.path.exists(unmasked_path) and os.path.exists(masked_path)):
                file_name_base = f"s2_{lat:.2f}_{lon:.2f}_{start_str}_{end_str}"
                futures.append(executor.submit(
                    s2_image_exporter, lat, lon, start_str, end_str, 
                    file_name_base, temp_dir, collection
                ))
        
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logging.error(f"Error during export: {e}")

    # Create empty template (we know the size!)
    num_bands = len(S2_BANDS)
    empty_template = np.zeros((num_bands, target_size, target_size), dtype=np.uint16)
    
    # Load and resize all images to target size
    images_unmasked = []
    images_masked = []
    metadata_windows = []
    template_crs = None
    template_transform = None
    
    for start, end in time_windows:
        unmasked_path, masked_path, start_str, end_str = get_file_paths(start, end)
        
        # Try to load both versions
        if os.path.exists(unmasked_path) and os.path.exists(masked_path):
            try:
                # Load unmasked
                with rasterio.open(unmasked_path) as src:
                    unmasked_img = src.read()
                    
                    # Get CRS and transform from first valid image
                    if template_crs is None:
                        template_crs = src.crs
                        template_transform = src.transform
                    
                    # Trim or pad to target size
                    unmasked_img = trim_or_pad_image(unmasked_img, target_size, nodata=0)
                
                # Load masked
                with rasterio.open(masked_path) as src:
                    masked_img = src.read()
                    masked_img = trim_or_pad_image(masked_img, target_size, nodata=0)
                    
                    # Calculate nodata fraction from masked version
                    nodata_mask = (masked_img == 0).any(axis=0)
                    nodata_fraction = nodata_mask.sum() / (target_size * target_size)
                
                # Add loaded images
                images_unmasked.append(unmasked_img)
                images_masked.append(masked_img)
                
                metadata_windows.append({
                    'date_range': [start_str, end_str],
                    'file_exists': True,
                    'masked_fraction': float(nodata_fraction)
                })
                
            except Exception as e:
                logging.error(f"Failed to read {unmasked_path} or {masked_path}: {e}")
                # Add empty template on error
                images_unmasked.append(empty_template.copy())
                images_masked.append(empty_template.copy())
                metadata_windows.append({
                    'date_range': [start_str, end_str],
                    'file_exists': False,
                    'masked_fraction': 1.0
                })
        else:
            # Files missing - add empty template directly
            images_unmasked.append(empty_template.copy())
            images_masked.append(empty_template.copy())
            metadata_windows.append({
                'date_range': [start_str, end_str],
                'file_exists': False,
                'masked_fraction': 1.0
            })
    
    # Check if we have at least one valid image
    if template_crs is None:
        raise ValueError(f"No valid images found for site {file_id}")
    
    # Stack everything (no more filling needed!)
    stacked_unmasked = np.stack(images_unmasked, axis=0)
    stacked_masked = np.stack(images_masked, axis=0)
    
    # Reshape for saving: (T, B, H, W) -> (T*B, H, W)
    T, B, H, W = stacked_unmasked.shape
    reshaped_unmasked = stacked_unmasked.transpose(1, 0, 2, 3).reshape(T * B, H, W)
    reshaped_masked = stacked_masked.transpose(1, 0, 2, 3).reshape(T * B, H, W)
    
    # Save unmasked stack
    stack_unmasked_file = os.path.join(out_dir, f"{file_id}_stack.tif")
    with rasterio.open(
        stack_unmasked_file, 'w',
        driver='GTiff',
        height=H, width=W,
        count=T * B,
        dtype=stacked_unmasked.dtype,
        crs=template_crs,
        transform=template_transform,
        nodata=0
    ) as dst:
        dst.write(reshaped_unmasked)
    
    # Save masked stack
    stack_masked_file = os.path.join(out_dir, f"{file_id}_stack_masked.tif")
    with rasterio.open(
        stack_masked_file, 'w',
        driver='GTiff',
        height=H, width=W,
        count=T * B,
        dtype=stacked_masked.dtype,
        crs=template_crs,
        transform=template_transform,
        nodata=0
    ) as dst:
        dst.write(reshaped_masked)
    
    # Save metadata
    metadata = {
        'file_id': file_id,
        'lat': float(lat),
        'lon': float(lon),
        'year': int(date.year),
        'collection': collection,
        'bands': S2_BANDS,
        'shape': list(stacked_masked.shape),
        'target_size': target_size,  
        'num_windows': num_windows,
        'num_windows_buffered': num_windows_buffered,
        'timestep_days': timestep,
        'start_month_unbuffered': start_month,
        'window_buffer': window_buffer,
        'windows': metadata_windows
    }
    
    metadata_file = os.path.join(out_dir, f"{file_id}_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Clean up temporary files
    import shutil
    shutil.rmtree(temp_dir)
    
    logging.info(f"Saved unmasked stack: {stack_unmasked_file}")
    logging.info(f"Saved masked stack: {stack_masked_file}")
    logging.info(f"Saved metadata: {metadata_file}")
    logging.info(f"Final shape: {stacked_masked.shape}")
    logging.info(f"Cleaned up temp directory: {temp_dir}")

# Helper functions

def trim_or_pad_image(img: np.ndarray, target_size: int, nodata: int = 0) -> np.ndarray:
    """
    Trim or pad image to target size.
    
    If image is larger than target, center-crops it.
    If image is smaller than target, pads with nodata.
    
    Args:
        img: Image array with shape (bands, height, width)
        target_size: Target size for height and width
        nodata: Value to use for padding (default 0)
    
    Returns:
        Image with shape (bands, target_size, target_size)
    """
    bands, h, w = img.shape
    
    # If already correct size, return as-is
    if h == target_size and w == target_size:
        return img
    
    # Create output array filled with nodata
    output = np.full((bands, target_size, target_size), nodata, dtype=img.dtype)
    
    # Calculate crop/pad regions
    h_start = max(0, (h - target_size) // 2)
    w_start = max(0, (w - target_size) // 2)
    h_end = h_start + min(h, target_size)
    w_end = w_start + min(w, target_size)
    
    out_h_start = max(0, (target_size - h) // 2)
    out_w_start = max(0, (target_size - w) // 2)
    out_h_end = out_h_start + min(h, target_size)
    out_w_end = out_w_start + min(w, target_size)
    
    # Copy data
    output[:, out_h_start:out_h_end, out_w_start:out_w_end] = \
        img[:, h_start:h_end, w_start:w_end]
    
    return output

def visualize_time_series_stack(out_dir, file_id):
    unmasked_file = f"{out_dir}/{file_id}_stack.tif"
    masked_file = f"{out_dir}/{file_id}_stack_masked.tif"

    # Read stacks
    with rasterio.open(unmasked_file) as src:
        unmasked_data = src.read()

    with rasterio.open(masked_file) as src:
        masked_data = src.read()

    # Reshape to (T, B, H, W)
    num_bands = 10
    T = unmasked_data.shape[0] // num_bands
    H, W = unmasked_data.shape[1], unmasked_data.shape[2]

    unmasked = unmasked_data.reshape(num_bands, T, H, W).transpose(1, 0, 2, 3)
    masked = masked_data.reshape(num_bands, T, H, W).transpose(1, 0, 2, 3)

    print(f"Stack shape: {unmasked.shape}")
    print(f"Number of timesteps: {T}")

    # RGB indices: [red, green, blue] = [B4, B3, B2] = [2, 1, 0]
    rgb_indices = [2, 1, 0]

    # Plot unmasked
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    for t in range(min(T, 15)):  # Show first 15
        rgb = unmasked[t, rgb_indices, :, :]
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = np.clip(rgb / 3000.0, 0, 1)
        
        axes[t].imshow(rgb)
        axes[t].set_title(f'Timestep {t}', fontsize=10)
        axes[t].axis('off')

    for t in range(min(T, 15), 15):
        axes[t].axis('off')

    plt.suptitle('UNMASKED Stack', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Plot masked
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten()

    for t in range(min(T, 15)):
        rgb = masked[t, rgb_indices, :, :]
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = np.clip(rgb / 3000.0, 0, 1)
        
        axes[t].imshow(rgb)
        axes[t].set_title(f'Timestep {t}', fontsize=10)
        axes[t].axis('off')

    for t in range(min(T, 15), 15):
        axes[t].axis('off')

    plt.suptitle('MASKED Stack', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def dataset_download(csv, download_dir, 
                collection='L1C',
                start_month=1,
                num_windows=36,
                timestep=10,
                window_buffer=3,
                target_size=100, 
                subset=False):
    """
    Download and stack images for each site represented by a row in a CSV.
    
    Args:
        csv: Path to csv with columns: x, y, unique_id, site_id, year, month, day
        download_dir: Output directory for all files
        collection: 'L1C' or 'L2A'
        start_month: Month to start time series (1=January, excluding buffer)
        num_windows: Number of timesteps to download (excluding buffer)
        timestep: Days per timestep
        window_buffer: Extra timesteps before/after for augmentation
        target_size: Size in pixels for all images (default 100x100)
    
    Returns:
        str: Success message
    """

    initialize_earthengine()

    # Create a version folder for this download based on datetime and save a metadata file within with all the arguments used
    version_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(download_dir, version_name)
    os.makedirs(out_dir, exist_ok=False)

    # Save metadata for this run
    args_dict = locals().copy()
    metadata_path = os.path.join(out_dir, f"metadata_{version_name}.json")
    with open(metadata_path, "w") as f:
        json.dump(args_dict, f, indent=2)
    
    # Read in the each observation to download data for
    data = pd.read_csv(csv)
    
    if subset == True:
        data = data.head(10)

    logging.info(f"Starting to process {len(data)} rows from {LABEL_CSV}")

    rows = list(data.iterrows())
    for _, row in rows:
    
        # Extract row data
        lat, lon = row['y'], row['x']
        uid = row['unique_id']
        date = datetime(int(row['year']), int(row['month']), int(row['day']))
        
        # Create file naming
        date_str = f"{date.year}.{date.month:02d}.{date.day:02d}"
        sid_raw = str(row['site_id'])
        sid_for_name = sid_raw.replace('id_', '')
        file_id = f"{uid}_{sid_for_name}_{date_str}"
        
        logging.info(f"Processing {file_id} at ({lat:.4f}, {lon:.4f})")

        # Download and stack - pass all parameters through
        retrieve_time_series_stack(
            file_id=file_id,
            lat=lat,
            lon=lon,
            date=date,
            out_dir=out_dir,
            collection=collection,
            start_month=start_month,
            num_windows=num_windows,
            timestep=timestep,
            window_buffer=window_buffer,
            target_size=target_size
        )
        
        # Add unique_id to metadata (for matching back to CSV)
        metadata_file = os.path.join(out_dir, f"{file_id}_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        metadata['unique_id'] = int(uid) if str(uid).isdigit() else uid
        metadata['original_site_id'] = sid_raw
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.info(f"Completed {file_id}")

    get_stats(out_dir)

    return f"Processed {file_id} successfully"

def get_stats(out_dir):
    """
    Compute summary statistics from Sentinel-2 metadata files in a directory.

    Scans the specified output directory for metadata files whose names end with
    "_metadata.json" (excluding files that start with "metadata_"), parses each
    JSON file to extract a list under the "windows" key, and computes:
        - how many samples meet the images coverage criterion (>=50% windows
          with 'file_exists' == True),
        - the median of per-sample average masked fraction across all samples
          (each sample's average is the mean of its windows' 'masked_fraction').

    Parameters
    ----------
    out_dir : str
        Path to the directory containing the per-sample metadata JSON files.

    Returns
    -------
    dict
        Dictionary containing:
            - total_samples (int)
            - samples_with_50pct_images (int)
            - percent_with_50pct_images (float, 0-100)
            - median_of_avg_masked_fraction (float, 0-1)
    Side effects
    ------------
    - Writes a JSON file named "download_stats.json" into out_dir containing
      the same dictionary that is returned.
    - Logs errors when individual metadata files cannot be read or parsed; such
      files are skipped when computing per-sample counts.
    """

    all_metadata_files = [f for f in os.listdir(out_dir) if f.endswith('_metadata.json') and not f.startswith('metadata_')]
    total_samples = len(all_metadata_files)
    samples_with_50pct_images = 0

    per_sample_avg_maskeds = []

    for mf in all_metadata_files:
        path = os.path.join(out_dir, mf)
        try:
            with open(path, 'r') as f:
                meta = json.load(f)
        except Exception as e:
            logging.error(f"Failed to read metadata {path}: {e}")
            continue

        windows = meta.get('windows', [])
        if not windows:
            continue

        num_true = sum(1 for w in windows if w.get('file_exists') is True)
        # masked_fraction may be missing for some windows; default to 1.0 (fully masked)
        masked_vals = [float(w.get('masked_fraction', 1.0)) for w in windows]

        if (num_true / len(windows)) >= 0.5:
            samples_with_50pct_images += 1

        # compute per-sample average masked fraction and collect
        avg_masked = float(sum(masked_vals) / len(masked_vals))
        per_sample_avg_maskeds.append(avg_masked)

    # median across samples of the per-sample average masked fraction
    if len(per_sample_avg_maskeds) > 0:
        median_avg_masked = float(np.median(per_sample_avg_maskeds))
    else:
        median_avg_masked = 0.0

    stats = {
        'total_samples': total_samples,
        'samples_with_50pct_images': samples_with_50pct_images,
        'percent_with_50pct_images': (samples_with_50pct_images / total_samples) * 100 if total_samples > 0 else 0,
        'median_of_avg_masked_fraction': median_avg_masked
    }

    # Save stats
    stats_file = os.path.join(out_dir, "download_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    return stats

if __name__ == '__main__':

    LABEL_CSV    = os.path.join(project_root, "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv")

    data_root = get_data_root()
    DOWNLOAD_DIR = os.path.join(data_root, "features")

    dataset_download(
        csv=LABEL_CSV,
        download_dir=DOWNLOAD_DIR,
        collection='L2A',
        start_month=1,
        num_windows=36,
        timestep=10,
        window_buffer=3,
        target_size=100,
        subset=True  # Set to True to test with just 10 rows
    )