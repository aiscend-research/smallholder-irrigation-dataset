#!/usr/bin/env python3
"""
Download PlanetScope imagery time series for labeled irrigation sites.

This module mirrors the structure of download_sentinel2.py but uses the Planet
Data and Orders API instead of Google Earth Engine. Key differences:
- 4 bands (Blue, Green, Red, NIR) instead of 10
- ~3m resolution instead of 10m (so 333x333 pixels for 1km²)
- Uses UDM2 for cloud masking instead of QA60/SCL
- Async API requires order submission then polling for completion

Usage:
    from src.features.download_planetscope import dataset_download

    dataset_download(
        csv='data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv',
        download_dir='data/features_planet',
        start_month=1,
        num_windows=36,
        timestep=10,
        window_buffer=3,
        target_size=333,  # ~1km at 3m resolution
    )
"""

import sys
import os
import json
import logging
import asyncio
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import rasterio

# Planet SDK imports
try:
    from planet import Session, DataClient, OrdersClient
    from planet.exceptions import APIError
    PLANET_SDK_AVAILABLE = True
except ImportError:
    PLANET_SDK_AVAILABLE = False
    logging.warning("Planet SDK not installed. Run: pip install planet")

# Project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.utils import find_project_root, get_data_root
from src.utils.geometries import bounding_box
from src.features.download_sentinel2 import trim_or_pad_image, get_stats

# PlanetScope band configuration
# PSScene 4-band: Blue, Green, Red, NIR
PS_BANDS = ['blue', 'green', 'red', 'nir']
PS_BAND_INDICES = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


#############################
# Authentication
#############################

def get_planet_api_key():
    """
    Get Planet API key from secrets/planet-api-key.txt.

    Returns:
        str: Planet API key

    Raises:
        ValueError: If API key file not found
    """
    secrets_path = os.path.join(find_project_root(os.getcwd()), 'secrets', 'planet-api-key.txt')

    if not os.path.exists(secrets_path):
        raise ValueError(f"Planet API key not found. Create {secrets_path} with your API key.")

    with open(secrets_path, 'r') as f:
        api_key = f.read().strip()

    if not api_key:
        raise ValueError(f"Planet API key file is empty: {secrets_path}")

    return api_key


def initialize_planet():
    """Initialize Planet API authentication from secrets/planet-api-key.txt."""
    if not PLANET_SDK_AVAILABLE:
        raise ImportError("Planet SDK not installed. Run: pip install planet")

    api_key = get_planet_api_key()
    os.environ['PL_API_KEY'] = api_key  # Planet SDK reads from env
    logging.info("Planet API initialized.")
    return api_key


#############################
# Geometry helpers
#############################

def point_to_aoi(lon: float, lat: float, half_side_km: float = 0.5) -> dict:
    """
    Create a GeoJSON polygon AOI from a point with buffer.

    Uses geodesic calculations for accuracy (via geopy).

    Args:
        lon: Longitude in decimal degrees
        lat: Latitude in decimal degrees
        half_side_km: Half the side length in km (default 0.5 = 1km total)

    Returns:
        dict: GeoJSON Polygon geometry
    """
    from shapely.geometry import box, mapping

    min_lat, min_lon, max_lat, max_lon = bounding_box(lat, lon, half_side_km)
    return mapping(box(min_lon, min_lat, max_lon, max_lat))


#############################
# Search for imagery
#############################

async def search_scenes(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    max_cloud_cover: float = 0.5,
    check_aoi_coverage: bool = True
) -> list:
    """
    Search for PlanetScope scenes in a time window.

    Args:
        lat: Latitude in decimal degrees
        lon: Longitude in decimal degrees
        start_date: Start date as 'YYYY-MM-DD'
        end_date: End date as 'YYYY-MM-DD'
        max_cloud_cover: Maximum cloud cover fraction (0-1)
        check_aoi_coverage: If True, query Planet for actual AOI coverage (slower but accurate)

    Returns:
        list: List of scene items sorted by AOI coverage (highest first).
              Each item has 'aoi_clear_percent' and 'footprint_coverage' added.
    """
    from planet import data_filter

    aoi = point_to_aoi(lon, lat)

    # Convert string dates to datetime objects (Planet SDK requires datetime)
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    async with Session() as sess:
        client = DataClient(sess)

        # Build search filter
        combined_filter = data_filter.and_filter([
            data_filter.geometry_filter(aoi),
            data_filter.date_range_filter("acquired", gte=start_dt, lte=end_dt),
            data_filter.range_filter("cloud_cover", lte=max_cloud_cover),
        ])

        # Collect matching items
        items = []
        async for item in client.search(
            search_filter=combined_filter,
            item_types=["PSScene"]
        ):
            items.append(item)

        if not items:
            return []

        # Calculate actual coverage within AOI for each scene
        if check_aoi_coverage:
            from shapely.geometry import shape

            aoi_shape = shape(aoi)

            for item in items:
                try:
                    # Calculate footprint coverage: what fraction of AOI is covered by scene geometry
                    scene_geom = item.get('geometry')
                    if scene_geom:
                        scene_shape = shape(scene_geom)
                        intersection = aoi_shape.intersection(scene_shape)
                        footprint_coverage = (intersection.area / aoi_shape.area) * 100
                    else:
                        print("No geometry found for item:", item['id'], " . Assuming full coverage.")
                        footprint_coverage = 100.0  # Assume full coverage if geometry not available

                    # Get cloud/clear coverage within AOI using Planet's get_item_coverage
                    # mode="estimate" returns quickly based on browse imagery
                    coverage = await client.get_item_coverage(
                        item_type_id="PSScene",
                        item_id=item['id'],
                        geometry=aoi,
                        mode="estimate"
                    )
                    aoi_clear_pct = coverage.get('clear_percent', 0)

                    item['aoi_clear_percent'] = aoi_clear_pct
                    item['footprint_coverage'] = footprint_coverage
                    # Combined score: effective usable coverage (footprint * clear fraction)
                    item['effective_coverage'] = footprint_coverage * (aoi_clear_pct / 100.0)

                    logging.debug(f"Scene {item['id']}: footprint {footprint_coverage:.1f}%, clear {aoi_clear_pct:.1f}%, effective {item['effective_coverage']:.1f}%")

                except Exception as e:
                    # If coverage check fails, fall back to scene-level stats
                    logging.warning(f"Coverage check failed for {item['id']}: {e}")
                    item['aoi_clear_percent'] = item['properties'].get('clear_percent', 0)
                    item['footprint_coverage'] = 100.0  # Unknown, assume full
                    item['effective_coverage'] = item['aoi_clear_percent']

            # Sort by effective coverage (footprint * clear), highest first
            items.sort(key=lambda x: x.get('effective_coverage', 0), reverse=True)
        else:
            # Fall back to scene-level clear_percent
            for item in items:
                item['aoi_clear_percent'] = item['properties'].get('clear_percent', 0)
                item['footprint_coverage'] = 100.0
                item['effective_coverage'] = item['aoi_clear_percent']
            items.sort(key=lambda x: x['properties'].get('clear_percent', 0), reverse=True)

        return items


def search_scenes_sync(lat: float, lon: float, start_date: str, end_date: str,
                       max_cloud_cover: float = 0.5, check_aoi_coverage: bool = True) -> list:
    """Synchronous wrapper for search_scenes."""
    return asyncio.run(search_scenes(lat, lon, start_date, end_date, max_cloud_cover, check_aoi_coverage))


#############################
# Download a single scene
#############################

async def order_and_download_batch(
    item_ids: list,
    lat: float,
    lon: float,
    output_dir: str,
    order_name: str
) -> dict:
    """
    Order multiple PlanetScope scenes in a single batch, wait, and download all.

    Args:
        item_ids: List of Planet scene IDs
        lat: Latitude for clipping AOI
        lon: Longitude for clipping AOI
        output_dir: Directory to save files
        order_name: Name for the order

    Returns:
        dict: Mapping of item_id -> (unmasked_path, masked_path) for successful downloads
    """
    os.makedirs(output_dir, exist_ok=True)
    aoi = point_to_aoi(lon, lat)

    async with Session() as sess:
        client = OrdersClient(sess)

        # Build batch order request
        order_request = {
            "name": order_name,
            "products": [
                {
                    "item_ids": item_ids,
                    "item_type": "PSScene",
                    "product_bundle": "analytic_sr_udm2" # Surface Reflectance + UDM2 cloud mask
                }
            ],
            "tools": [
                {"clip": {"aoi": aoi}}
            ]
        }

        try:
            # Create order
            order = await client.create_order(order_request)
            order_id = order['id']
            logging.info(f"Created batch order {order_id} with {len(item_ids)} scenes")

            # Poll for completion
            while True:
                order_info = await client.get_order(order_id)
                state = order_info['state']

                if state == 'success':
                    logging.info(f"Order {order_id} completed successfully")
                    break
                elif state in ['failed', 'partial']:
                    logging.error(f"Order {order_id} ended with state: {state}")
                    if state == 'failed':
                        return {}
                    break  # partial = some succeeded, continue to download those

                logging.info(f"Order {order_id} state: {state}, waiting...")
                await asyncio.sleep(30)

            # Download all results
            results = {}
            downloaded_files = []

            for result in order_info.get('_links', {}).get('results', []):
                name = result.get('name', '')
                url = result.get('location', '')

                if name.endswith('.tif'):
                    # Planet returns nested paths like "{order_id}/PSScene/{file}.tif"
                    # Just use the filename to save flat in output_dir
                    filename = os.path.basename(name)
                    local_path = os.path.join(output_dir, filename)
                    await download_file(url, local_path)
                    downloaded_files.append((filename, local_path))

            # Match up SR images with their UDM2 masks
            # Filenames look like: 20230607_072630_16_2439_3B_AnalyticMS_SR_clip.tif
            #                      20230607_072630_16_2439_3B_udm2_clip.tif
            # Scene ID is the part before "_3B_": 20230607_072630_16_2439
            sr_files = {}
            udm2_files = {}

            for filename, path in downloaded_files:
                # Extract scene ID (everything before _3B_)
                if '_3B_' in filename:
                    scene_id = filename.split('_3B_')[0]
                else:
                    # Fallback: use filename without extension
                    scene_id = filename.replace('.tif', '')

                if 'AnalyticMS_SR' in filename or 'SR_clip' in filename:
                    sr_files[scene_id] = path
                elif 'udm2' in filename.lower():
                    udm2_files[scene_id] = path

            # Apply masks and build results
            for scene_id in sr_files:
                sr_path = sr_files[scene_id]
                udm2_path = udm2_files.get(scene_id)

                if udm2_path and os.path.exists(udm2_path):
                    masked_path = sr_path.replace('.tif', '_masked.tif')
                    apply_udm2_mask(sr_path, udm2_path, masked_path)
                    os.remove(udm2_path)
                    results[scene_id] = (sr_path, masked_path)
                else:
                    results[scene_id] = (sr_path, None)

            return results

        except APIError as e:
            logging.error(f"Planet API error: {e}")
            return {}


async def download_file(url: str, output_path: str):
    """Download a file from URL."""
    import aiohttp

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            with open(output_path, 'wb') as f:
                async for chunk in resp.content.iter_chunked(8192):
                    f.write(chunk)
    logging.info(f"Downloaded: {output_path}")


def apply_udm2_mask(image_path: str, udm2_path: str, output_path: str, nodata: int = 0):
    """
    Apply UDM2 cloud mask to image.

    UDM2 bands:
        Band 1: Clear map (1=usable, 0=unusable)
        Band 2: Snow
        Band 3: Shadow
        Band 4: Light haze
        Band 5: Heavy haze
        Band 6: Cloud
        Band 7: Confidence
        Band 8: Unusable data

    Args:
        image_path: Path to surface reflectance image
        udm2_path: Path to UDM2 mask
        output_path: Path for masked output
        nodata: Value to use for masked pixels (default 0)
    """
    with rasterio.open(image_path) as src:
        image = src.read()
        profile = src.profile.copy()

    with rasterio.open(udm2_path) as src:
        udm2 = src.read()

    # Create mask from UDM2
    # Band 1 (index 0) is the clear map: 1=clear, 0=not clear
    if udm2.shape[0] >= 1:
        clear_mask = udm2[0] == 1
    else:
        # Fallback: assume all clear
        clear_mask = np.ones(image.shape[1:], dtype=bool)

    # Apply mask to all bands
    masked_image = image.copy()
    for b in range(masked_image.shape[0]):
        masked_image[b][~clear_mask] = nodata

    # Update profile
    profile.update(nodata=nodata)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(masked_image)

    logging.info(f"Created masked image: {output_path}")


#############################
# Single window download (sync wrapper)
#############################

#############################
# Time series stack download
#############################

async def search_best_scenes_for_windows(
    lat: float,
    lon: float,
    time_windows: list,
    max_cloud_cover: float = 0.5
) -> dict:
    """
    Search for the best scene for each time window.

    Args:
        lat: Latitude
        lon: Longitude
        time_windows: List of (start_date, end_date) tuples
        max_cloud_cover: Maximum cloud cover

    Returns:
        dict: {window_index: {"item_id": str, "cloud_cover": float, "date_range": [str, str]}}
    """
    results = {}

    for i, (start, end) in enumerate(time_windows):
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        items = await search_scenes(lat, lon, start_str, end_str, max_cloud_cover)

        if items:
            best = items[0]
            results[i] = {
                "item_id": best['id'],
                "cloud_cover": best['properties'].get('cloud_cover', 0),
                "date_range": [start_str, end_str]
            }
            logging.info(f"Window {i}: found {len(items)} scenes, best cloud_cover={results[i]['cloud_cover']:.2f}")
        else:
            results[i] = {
                "item_id": None,
                "cloud_cover": None,
                "date_range": [start_str, end_str]
            }
            logging.warning(f"Window {i}: no scenes found for {start_str} to {end_str}")

    return results


def retrieve_time_series_stack(
    file_id: str,
    lat: float,
    lon: float,
    date: datetime,
    out_dir: str,
    start_month: int = 1,
    num_windows: int = 36,
    timestep: int = 10,
    window_buffer: int = 3,
    target_size: int = 333,
    max_cloud_cover: float = 0.5
):
    """
    Download and stack PlanetScope images over a time series using batch ordering.

    Flow:
    1. Search for best scene for each time window
    2. Submit ONE batch order for all scenes
    3. Wait for processing (done in parallel on Planet's side)
    4. Download all results
    5. Stack into final output

    Args:
        file_id: Unique identifier for this site
        lat, lon: Coordinates (WGS84 decimal degrees)
        date: Reference date (year used for time series)
        out_dir: Directory to save final outputs
        start_month: Month to start downloading (1=January, excluding buffer)
        num_windows: Number of timesteps to download (excluding buffer)
        timestep: Days per timestep
        window_buffer: Extra timesteps before/after
        target_size: Size in pixels for all images (default 333 for ~1km at 3m)
        max_cloud_cover: Maximum cloud cover for scene selection

    Returns:
        None. Saves files to:
            - {out_dir}/{file_id}_stack.tif
            - {out_dir}/{file_id}_stack_masked.tif
            - {out_dir}/{file_id}_metadata.json
    """
    # Create time windows (same logic as S2)
    step_size = timedelta(days=timestep)
    num_windows_buffered = num_windows + (window_buffer * 2)
    start_date_dt = datetime(date.year, start_month, 1) - (step_size * window_buffer)
    time_windows = [
        (start_date_dt + i * step_size, start_date_dt + (i + 1) * step_size)
        for i in range(num_windows_buffered)
    ]

    # Create temporary directory for downloads
    temp_dir = os.path.join(out_dir, "_tmp", file_id)
    os.makedirs(temp_dir, exist_ok=True)

    # Step 1: Search for best scene for each window
    logging.info(f"Searching for scenes across {num_windows_buffered} windows...")
    window_scenes = asyncio.run(
        search_best_scenes_for_windows(lat, lon, time_windows, max_cloud_cover)
    )

    # Collect all scene IDs that were found
    item_ids = []
    window_to_item = {}  # Map window index -> item_id
    for i, info in window_scenes.items():
        if info["item_id"]:
            item_ids.append(info["item_id"])
            window_to_item[i] = info["item_id"]

    logging.info(f"Found scenes for {len(item_ids)}/{num_windows_buffered} windows")

    # Step 2 & 3: Submit batch order and download
    downloaded = {}
    if item_ids:
        logging.info(f"Submitting batch order for {len(item_ids)} scenes...")
        downloaded = asyncio.run(
            order_and_download_batch(item_ids, lat, lon, temp_dir, f"batch_{file_id}")
        )
        logging.info(f"Downloaded {len(downloaded)} scenes")

    # Step 4: Build stacks
    num_bands = len(PS_BANDS)
    empty_template = np.zeros((num_bands, target_size, target_size), dtype=np.uint16)

    images_unmasked = []
    images_masked = []
    metadata_windows = []
    template_crs = None
    template_transform = None

    for i, (start, end) in enumerate(time_windows):
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        item_id = window_to_item.get(i)
        paths = downloaded.get(item_id) if item_id else None

        if paths and paths[0] and os.path.exists(paths[0]):
            unmasked_path, masked_path = paths
            try:
                # Load unmasked
                with rasterio.open(unmasked_path) as src:
                    unmasked_img = src.read()
                    if template_crs is None:
                        template_crs = src.crs
                        template_transform = src.transform
                    unmasked_img = trim_or_pad_image(unmasked_img, target_size, nodata=0)

                # Load masked (or use unmasked if no mask)
                if masked_path and os.path.exists(masked_path):
                    with rasterio.open(masked_path) as src:
                        masked_img = src.read()
                        masked_img = trim_or_pad_image(masked_img, target_size, nodata=0)
                else:
                    masked_img = unmasked_img.copy()

                # Calculate nodata fraction
                nodata_mask = (masked_img == 0).any(axis=0)
                nodata_fraction = nodata_mask.sum() / (target_size * target_size)

                images_unmasked.append(unmasked_img)
                images_masked.append(masked_img)
                metadata_windows.append({
                    'date_range': [start_str, end_str],
                    'item_id': item_id,
                    'file_exists': True,
                    'masked_fraction': float(nodata_fraction),
                    'cloud_cover': window_scenes[i].get('cloud_cover')
                })

            except Exception as e:
                logging.error(f"Failed to read window {i}: {e}")
                images_unmasked.append(empty_template.copy())
                images_masked.append(empty_template.copy())
                metadata_windows.append({
                    'date_range': [start_str, end_str],
                    'item_id': item_id,
                    'file_exists': False,
                    'masked_fraction': 1.0
                })
        else:
            # No scene found or download failed for this window
            images_unmasked.append(empty_template.copy())
            images_masked.append(empty_template.copy())
            metadata_windows.append({
                'date_range': [start_str, end_str],
                'item_id': item_id,
                'file_exists': False,
                'masked_fraction': 1.0
            })

    if template_crs is None:
        raise ValueError(f"No valid images found for site {file_id}")

    # Stack everything
    stacked_unmasked = np.stack(images_unmasked, axis=0)
    stacked_masked = np.stack(images_masked, axis=0)

    # Reshape for saving: (T, B, H, W) -> (B, T, H, W) -> (B*T, H, W)
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
        'sensor': 'PlanetScope',
        'bands': PS_BANDS,
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

    # Clean up temp files
    import shutil
    shutil.rmtree(temp_dir)

    logging.info(f"Saved unmasked stack: {stack_unmasked_file}")
    logging.info(f"Saved masked stack: {stack_masked_file}")
    logging.info(f"Saved metadata: {metadata_file}")
    logging.info(f"Final shape: {stacked_masked.shape}")


#############################
# Dataset download
#############################

def dataset_download(
    csv: str,
    download_dir: str,
    start_month: int = 1,
    num_windows: int = 36,
    timestep: int = 10,
    window_buffer: int = 3,
    target_size: int = 333,
    max_cloud_cover: float = 0.5,
    subset: bool = False
):
    """
    Download and stack PlanetScope images for each site in a CSV.

    Mirrors dataset_download() from download_sentinel2.py.

    Args:
        csv: Path to CSV with columns: x, y, unique_id, site_id, year, month, day
        download_dir: Output directory for all files
        start_month: Month to start time series (1=January)
        num_windows: Number of timesteps (excluding buffer)
        timestep: Days per timestep
        window_buffer: Extra timesteps before/after
        target_size: Size in pixels (default 333 for ~1km at 3m)
        max_cloud_cover: Maximum cloud cover for scene selection
        subset: If True, only process first 10 rows (for testing)

    Returns:
        str: Success message
    """
    initialize_planet()

    # Create version folder
    version_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(download_dir, version_name)
    os.makedirs(out_dir, exist_ok=False)

    # Save run metadata
    args_dict = {
        'csv': csv,
        'download_dir': download_dir,
        'start_month': start_month,
        'num_windows': num_windows,
        'timestep': timestep,
        'window_buffer': window_buffer,
        'target_size': target_size,
        'max_cloud_cover': max_cloud_cover,
        'subset': subset,
        'sensor': 'PlanetScope'
    }
    metadata_path = os.path.join(out_dir, f"metadata_{version_name}.json")
    with open(metadata_path, 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Read input data
    data = pd.read_csv(csv)

    if subset:
        data = data.head(10)

    logging.info(f"Starting to process {len(data)} rows from {csv}")

    for _, row in data.iterrows():
        lat, lon = row['y'], row['x']
        uid = row['unique_id']
        date = datetime(int(row['year']), int(row['month']), int(row['day']))

        # Create file naming (same convention as S2)
        date_str = f"{date.year}.{date.month:02d}.{date.day:02d}"
        sid_raw = str(row['site_id'])
        sid_for_name = sid_raw.replace('id_', '')
        file_id = f"{uid}_{sid_for_name}_{date_str}"

        logging.info(f"Processing {file_id} at ({lat:.4f}, {lon:.4f})")

        try:
            retrieve_time_series_stack(
                file_id=file_id,
                lat=lat,
                lon=lon,
                date=date,
                out_dir=out_dir,
                start_month=start_month,
                num_windows=num_windows,
                timestep=timestep,
                window_buffer=window_buffer,
                target_size=target_size,
                max_cloud_cover=max_cloud_cover
            )

            # Add unique_id to metadata
            metadata_file = os.path.join(out_dir, f"{file_id}_metadata.json")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            metadata['unique_id'] = int(uid) if str(uid).isdigit() else uid
            metadata['original_site_id'] = sid_raw

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logging.info(f"Completed {file_id}")

        except Exception as e:
            logging.error(f"Failed to process {file_id}: {e}")
            continue

    # Compute stats
    get_stats(out_dir)

    return f"Processed {len(data)} sites successfully"


#############################
# Parallel batch processing for many sites
#############################

async def process_sites_parallel(
    sites: list,
    out_dir: str,
    start_month: int = 1,
    num_windows: int = 36,
    timestep: int = 10,
    window_buffer: int = 3,
    target_size: int = 333,
    max_cloud_cover: float = 0.5,
    max_concurrent_orders: int = 20,
    poll_interval: int = 60
):
    """
    Process multiple sites with parallel order submission.

    Submits up to max_concurrent_orders at a time, polls for completion,
    downloads results, then submits more.

    Args:
        sites: List of dicts with keys: file_id, lat, lon, date (datetime)
        out_dir: Output directory
        max_concurrent_orders: Max orders to have pending at once
        poll_interval: Seconds between polling for order status
        (other args same as retrieve_time_series_stack)

    Returns:
        dict: {file_id: "success" | "failed" | error_message}
    """
    from collections import deque

    step_size = timedelta(days=timestep)
    num_windows_buffered = num_windows + (window_buffer * 2)

    # Track state
    pending_orders = {}  # order_id -> site_info
    results = {}
    sites_queue = deque(sites)

    async with Session() as sess:
        orders_client = OrdersClient(sess)

        while sites_queue or pending_orders:
            # Submit new orders up to limit
            while sites_queue and len(pending_orders) < max_concurrent_orders:
                site = sites_queue.popleft()
                file_id = site['file_id']
                lat, lon = site['lat'], site['lon']
                date = site['date']

                # Create time windows
                start_date_dt = datetime(date.year, start_month, 1) - (step_size * window_buffer)
                time_windows = [
                    (start_date_dt + i * step_size, start_date_dt + (i + 1) * step_size)
                    for i in range(num_windows_buffered)
                ]

                # Search for scenes
                try:
                    window_scenes = await search_best_scenes_for_windows(
                        lat, lon, time_windows, max_cloud_cover
                    )
                    item_ids = [info["item_id"] for info in window_scenes.values() if info["item_id"]]

                    if not item_ids:
                        logging.warning(f"{file_id}: No scenes found")
                        results[file_id] = "no_scenes"
                        continue

                    # Submit order
                    aoi = point_to_aoi(lon, lat)
                    order_request = {
                        "name": f"batch_{file_id}",
                        "products": [{
                            "item_ids": item_ids,
                            "item_type": "PSScene",
                            "product_bundle": "analytic_sr_udm2"
                        }],
                        "tools": [{"clip": {"aoi": aoi}}]
                    }

                    order = await orders_client.create_order(order_request)
                    order_id = order['id']
                    pending_orders[order_id] = {
                        'file_id': file_id,
                        'lat': lat,
                        'lon': lon,
                        'window_scenes': window_scenes,
                        'time_windows': time_windows
                    }
                    logging.info(f"{file_id}: Submitted order {order_id} with {len(item_ids)} scenes")

                except Exception as e:
                    logging.error(f"{file_id}: Failed to submit order: {e}")
                    results[file_id] = f"submit_error: {e}"

                # Rate limit: small delay between submissions
                await asyncio.sleep(0.25)

            # Poll pending orders
            if pending_orders:
                logging.info(f"Polling {len(pending_orders)} pending orders...")
                completed = []

                for order_id, site_info in pending_orders.items():
                    try:
                        order_info = await orders_client.get_order(order_id)
                        state = order_info['state']

                        if state == 'success':
                            logging.info(f"{site_info['file_id']}: Order complete, downloading...")
                            await _download_and_stack_order(
                                order_info, site_info, out_dir, target_size
                            )
                            results[site_info['file_id']] = "success"
                            completed.append(order_id)

                        elif state in ['failed', 'partial']:
                            logging.error(f"{site_info['file_id']}: Order {state}")
                            results[site_info['file_id']] = state
                            completed.append(order_id)

                    except Exception as e:
                        logging.error(f"Error polling {order_id}: {e}")

                # Remove completed orders
                for order_id in completed:
                    del pending_orders[order_id]

                # Wait before next poll
                if pending_orders:
                    logging.info(f"Waiting {poll_interval}s... ({len(pending_orders)} pending, {len(sites_queue)} queued)")
                    await asyncio.sleep(poll_interval)

    return results


async def _download_and_stack_order(order_info, site_info, out_dir, target_size):
    """Download order results and create stacked outputs."""
    import aiohttp

    file_id = site_info['file_id']
    temp_dir = os.path.join(out_dir, "_tmp", file_id)
    os.makedirs(temp_dir, exist_ok=True)

    # Download all files
    downloaded_files = []
    async with aiohttp.ClientSession() as session:
        for result in order_info.get('_links', {}).get('results', []):
            name = result.get('name', '')
            url = result.get('location', '')

            if name.endswith('.tif'):
                local_path = os.path.join(temp_dir, name)
                async with session.get(url) as resp:
                    with open(local_path, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
                downloaded_files.append((name, local_path))

    # Match SR images with UDM2 masks and apply masking
    sr_files = {}
    udm2_files = {}

    for name, path in downloaded_files:
        # Extract scene ID from filename
        for part in name.split('_'):
            if len(part) > 20:
                scene_id = part
                break
        else:
            continue

        if 'AnalyticMS_SR' in name or 'analytic_sr' in name.lower():
            sr_files[scene_id] = path
        elif 'udm2' in name.lower():
            udm2_files[scene_id] = path

    # Apply masks
    processed = {}
    for scene_id, sr_path in sr_files.items():
        udm2_path = udm2_files.get(scene_id)
        if udm2_path and os.path.exists(udm2_path):
            masked_path = sr_path.replace('.tif', '_masked.tif')
            apply_udm2_mask(sr_path, udm2_path, masked_path)
            os.remove(udm2_path)
            processed[scene_id] = (sr_path, masked_path)
        else:
            processed[scene_id] = (sr_path, None)

    # Build stack (reuse logic from retrieve_time_series_stack)
    window_scenes = site_info['window_scenes']
    time_windows = site_info['time_windows']

    num_bands = len(PS_BANDS)
    empty_template = np.zeros((num_bands, target_size, target_size), dtype=np.uint16)

    images_unmasked = []
    images_masked = []
    metadata_windows = []
    template_crs = None
    template_transform = None

    for i, (start, end) in enumerate(time_windows):
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        item_id = window_scenes[i].get("item_id") if i in window_scenes else None
        paths = processed.get(item_id) if item_id else None

        if paths and paths[0] and os.path.exists(paths[0]):
            try:
                with rasterio.open(paths[0]) as src:
                    unmasked_img = src.read()
                    if template_crs is None:
                        template_crs = src.crs
                        template_transform = src.transform
                    unmasked_img = trim_or_pad_image(unmasked_img, target_size, nodata=0)

                if paths[1] and os.path.exists(paths[1]):
                    with rasterio.open(paths[1]) as src:
                        masked_img = src.read()
                        masked_img = trim_or_pad_image(masked_img, target_size, nodata=0)
                else:
                    masked_img = unmasked_img.copy()

                nodata_mask = (masked_img == 0).any(axis=0)
                nodata_fraction = nodata_mask.sum() / (target_size * target_size)

                images_unmasked.append(unmasked_img)
                images_masked.append(masked_img)
                metadata_windows.append({
                    'date_range': [start_str, end_str],
                    'item_id': item_id,
                    'file_exists': True,
                    'masked_fraction': float(nodata_fraction)
                })
            except Exception as e:
                logging.error(f"Error reading {paths[0]}: {e}")
                images_unmasked.append(empty_template.copy())
                images_masked.append(empty_template.copy())
                metadata_windows.append({
                    'date_range': [start_str, end_str],
                    'item_id': item_id,
                    'file_exists': False,
                    'masked_fraction': 1.0
                })
        else:
            images_unmasked.append(empty_template.copy())
            images_masked.append(empty_template.copy())
            metadata_windows.append({
                'date_range': [start_str, end_str],
                'item_id': item_id,
                'file_exists': False,
                'masked_fraction': 1.0
            })

    if template_crs is None:
        raise ValueError(f"No valid images for {file_id}")

    # Stack and save
    stacked_unmasked = np.stack(images_unmasked, axis=0)
    stacked_masked = np.stack(images_masked, axis=0)

    T, B, H, W = stacked_unmasked.shape
    reshaped_unmasked = stacked_unmasked.transpose(1, 0, 2, 3).reshape(T * B, H, W)
    reshaped_masked = stacked_masked.transpose(1, 0, 2, 3).reshape(T * B, H, W)

    stack_unmasked_file = os.path.join(out_dir, f"{file_id}_stack.tif")
    with rasterio.open(
        stack_unmasked_file, 'w', driver='GTiff',
        height=H, width=W, count=T * B,
        dtype=stacked_unmasked.dtype,
        crs=template_crs, transform=template_transform, nodata=0
    ) as dst:
        dst.write(reshaped_unmasked)

    stack_masked_file = os.path.join(out_dir, f"{file_id}_stack_masked.tif")
    with rasterio.open(
        stack_masked_file, 'w', driver='GTiff',
        height=H, width=W, count=T * B,
        dtype=stacked_masked.dtype,
        crs=template_crs, transform=template_transform, nodata=0
    ) as dst:
        dst.write(reshaped_masked)

    # Save metadata
    metadata = {
        'file_id': file_id,
        'lat': float(site_info['lat']),
        'lon': float(site_info['lon']),
        'sensor': 'PlanetScope',
        'bands': PS_BANDS,
        'shape': list(stacked_masked.shape),
        'target_size': target_size,
        'windows': metadata_windows
    }

    with open(os.path.join(out_dir, f"{file_id}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Cleanup temp
    import shutil
    shutil.rmtree(temp_dir)

    logging.info(f"{file_id}: Saved stacks ({T} windows, {B} bands)")


def dataset_download_parallel(
    csv: str,
    download_dir: str,
    max_concurrent_orders: int = 20,
    **kwargs
) -> dict:
    """
    Download PlanetScope imagery for all sites in CSV using parallel ordering.

    Args:
        csv: Path to CSV with columns: x, y, unique_id, site_id, year, month, day
        download_dir: Output directory
        max_concurrent_orders: How many orders to have pending at once (default 20)
        **kwargs: Passed to process_sites_parallel

    Returns:
        dict: Results per site
    """
    initialize_planet()

    # Create version folder
    version_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(download_dir, version_name)
    os.makedirs(out_dir, exist_ok=False)

    # Read sites
    data = pd.read_csv(csv)

    sites = []
    for _, row in data.iterrows():
        date_str = f"{int(row['year'])}.{int(row['month']):02d}.{int(row['day']):02d}"
        sid = str(row['site_id']).replace('id_', '')
        sites.append({
            'file_id': f"{row['unique_id']}_{sid}_{date_str}",
            'lat': row['y'],
            'lon': row['x'],
            'date': datetime(int(row['year']), int(row['month']), int(row['day']))
        })

    logging.info(f"Processing {len(sites)} sites with max {max_concurrent_orders} concurrent orders")

    # Run parallel processing
    results = asyncio.run(
        process_sites_parallel(sites, out_dir, max_concurrent_orders=max_concurrent_orders, **kwargs)
    )

    # Save results summary
    with open(os.path.join(out_dir, "download_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # Compute stats
    get_stats(out_dir)

    success = sum(1 for v in results.values() if v == "success")
    logging.info(f"Complete: {success}/{len(sites)} successful")

    return results


#############################
# Main
#############################

if __name__ == '__main__':
    LABEL_CSV = os.path.join(
        project_root,
        "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv"
    )

    data_root = get_data_root()
    DOWNLOAD_DIR = os.path.join(data_root, "features_planet")

    # Use parallel downloading for efficiency
    # With 20 concurrent orders and ~30 min processing each:
    # 2500 sites ≈ 125 batches × 30 min = ~63 hours
    # With 50 concurrent orders: ~25 hours

    dataset_download_parallel(
        csv=LABEL_CSV,
        download_dir=DOWNLOAD_DIR,
        max_concurrent_orders=20,  # Adjust based on Planet's limits
        start_month=1,
        num_windows=36,
        timestep=10,
        window_buffer=3,
        target_size=333,
        max_cloud_cover=0.5
    )
