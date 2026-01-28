#!/usr/bin/env python3
"""
Download PlanetScope imagery time series for labeled irrigation sites.

This module mirrors the structure of download_sentinel2.py but uses the Planet
Data and Orders API instead of Google Earth Engine. Key differences:
- 4 bands (Blue, Green, Red, NIR) instead of 10
- 3m resolution instead of 10m
- Uses UDM2 for cloud masking instead of QA60/SCL
- Async API requires order submission then polling for completion

Grid configuration is defined by PS_TARGET_SIZE (334 pixels) and PS_RESOLUTION (3m).
This gives a ~1km² area with slight padding for grid alignment safety.

Usage:
    from src.features.download_planetscope import dataset_download

    dataset_download(
        csv='data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv',
        download_dir='data/features_planet',
        start_month=1,
        num_windows=36,
        timestep=10,
        window_buffer=3,
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
from src.features.download_sentinel2 import get_stats

# PlanetScope 4-band configuration
PS_BANDS = ['blue', 'green', 'red', 'nir']
PS_BAND_INDICES = {'blue': 0, 'green': 1, 'red': 2, 'nir': 3}

# Product type configuration (similar to Sentinel-2's L1C/L2A)
# SR = Surface Reflectance (atmospherically corrected, fewer scenes available)
# TOA = Top of Atmosphere (raw reflectance, more scenes available)
PRODUCT_TYPES = {
    'SR': {
        'bundle': 'analytic_sr_udm2',
        'asset': 'ortho_analytic_4b_sr',
        'description': 'Surface Reflectance (atmospherically corrected)'
    },
    'TOA': {
        'bundle': 'analytic_udm2',
        'asset': 'ortho_analytic_4b',
        'description': 'Top of Atmosphere (uncorrected)'
    }
}

# Grid configuration
PS_RESOLUTION = 3.0  # meters per pixel
PS_TARGET_SIZE = 334  # pixels per side (334 * 3m = 1002m, slightly over 1km for safety)

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
    check_aoi_coverage: bool = True,
    product_type: str = 'SR'
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
        product_type: 'SR' (Surface Reflectance) or 'TOA' (Top of Atmosphere)

    Returns:
        list: List of scene items sorted by AOI coverage (highest first).
              Each item has 'aoi_clear_percent' and 'footprint_coverage' added.
    """
    if product_type not in PRODUCT_TYPES:
        raise ValueError(f"Invalid product_type '{product_type}'. Must be one of: {list(PRODUCT_TYPES.keys())}")
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

        # Filter to only include scenes with the required asset available
        # The '_permissions' field lists assets the user has access to download
        required_asset = PRODUCT_TYPES[product_type]['asset']
        items_with_access = []
        for item in items:
            permissions = item.get('_permissions', [])
            # Check if we have download access to the required asset
            if any(required_asset in str(p) for p in permissions):
                items_with_access.append(item)
            else:
                logging.debug(f"Skipping {item['id']}: no access to {required_asset}")

        if len(items_with_access) < len(items):
            logging.info(f"Filtered {len(items) - len(items_with_access)}/{len(items)} scenes without {product_type} access")

        items = items_with_access

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
                        logging.debug("Footprint coverage for item %s: %.2f", item['id'], footprint_coverage)
                    else:
                        logging.warning("No geometry found for item: %s . Assuming full coverage.", item['id'])
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
                       max_cloud_cover: float = 0.5, check_aoi_coverage: bool = True,
                       product_type: str = 'SR') -> list:
    """Synchronous wrapper for search_scenes."""
    return asyncio.run(search_scenes(lat, lon, start_date, end_date, max_cloud_cover, check_aoi_coverage, product_type))


#############################
# Download a single scene
#############################

def get_utm_epsg(lat: float, lon: float) -> str:
    """Get the appropriate UTM EPSG code for a lat/lon coordinate."""
    zone = int((lon + 180) / 6) + 1
    if lat >= 0:
        return f"EPSG:326{zone:02d}"  # Northern hemisphere
    else:
        return f"EPSG:327{zone:02d}"  # Southern hemisphere


def _build_order_request(
    item_ids: list,
    lat: float,
    lon: float,
    anchor_id: str,
    order_name: str,
    aoi_buffer_m: float = 20.0,
    product_type: str = 'SR'
) -> dict:
    """
    Build a Planet order request with reproject, coregister, and clip tools.

    NOTE: Planet applies tools in a FIXED order regardless of request order:
    1. clip - Clips to AOI (in native projection)
    2. reproject - Reprojects to UTM at 3m resolution
    3. coregister - Sub-pixel alignment to anchor

    Because clip happens before reproject, each scene may end up on a slightly
    different UTM grid. We add a buffer to the AOI to ensure we have enough
    pixels, then align to a common grid during stacking.

    Args:
        item_ids: List of Planet scene IDs
        lat: Latitude for clipping AOI
        lon: Longitude for clipping AOI
        anchor_id: Scene ID to use as coregistration anchor
        order_name: Name for the order
        aoi_buffer_m: Extra buffer in meters to add around AOI (default 20m)
        product_type: 'SR' (Surface Reflectance) or 'TOA' (Top of Atmosphere)

    Returns:
        dict: Order request ready for Planet Orders API
    """
    if product_type not in PRODUCT_TYPES:
        raise ValueError(f"Invalid product_type '{product_type}'. Must be one of: {list(PRODUCT_TYPES.keys())}")

    # Add buffer to AOI to account for grid misalignment after clip->reproject
    half_side_km = 0.5 + (aoi_buffer_m / 1000.0)
    aoi = point_to_aoi(lon, lat, half_side_km=half_side_km)
    utm_epsg = get_utm_epsg(lat, lon)

    product_bundle = PRODUCT_TYPES[product_type]['bundle']

    order_request = {
        "name": order_name,
        "products": [
            {
                "item_ids": item_ids,
                "item_type": "PSScene",
                "product_bundle": product_bundle
            }
        ],
        "tools": [
            {"reproject": {
                "projection": utm_epsg,
                "resolution": 3,  # Consistent 3m pixel size
                "kernel": "near"  # Nearest neighbor to minimize blurring
            }},
            {"coregister": {"anchor_item": anchor_id}},
            {"clip": {"aoi": aoi}}
        ]
    }

    logging.info(f"Order [{product_type}]: reproject to {utm_epsg} at 3m, coregister to {anchor_id}, clip to AOI")
    return order_request


async def _download_and_process_order(
    order_info: dict,
    output_dir: str
) -> dict:
    """
    Download files from a completed Planet order and apply UDM2 masks.

    Args:
        order_info: Order info dict from Planet API (must have 'success' state)
        output_dir: Directory to save downloaded files

    Returns:
        dict: Mapping of scene_id -> (sr_path, masked_path) for downloaded scenes
    """
    import aiohttp

    os.makedirs(output_dir, exist_ok=True)

    # Download all TIF files
    downloaded_files = []
    async with aiohttp.ClientSession() as session:
        for result in order_info.get('_links', {}).get('results', []):
            name = result.get('name', '')
            url = result.get('location', '')

            if name.endswith('.tif'):
                # Planet returns nested paths like "{order_id}/PSScene/{file}.tif"
                # Just use the filename to save flat in output_dir
                filename = os.path.basename(name)
                local_path = os.path.join(output_dir, filename)

                async with session.get(url) as resp:
                    with open(local_path, 'wb') as f:
                        async for chunk in resp.content.iter_chunked(8192):
                            f.write(chunk)
                downloaded_files.append((filename, local_path))

    # Match up analytic images with their UDM2 masks
    # SR filenames look like: 20230607_072630_16_2439_3B_AnalyticMS_SR_clip.tif
    # TOA filenames look like: 20230607_072630_16_2439_3B_AnalyticMS_clip.tif
    # UDM2 filenames:          20230607_072630_16_2439_3B_udm2_clip.tif
    # Scene ID is the part before "_3B_": 20230607_072630_16_2439
    analytic_files = {}
    udm2_files = {}

    for filename, path in downloaded_files:
        # Extract scene ID (everything before _3B_)
        if '_3B_' in filename:
            scene_id = filename.split('_3B_')[0]
        else:
            # Fallback: use filename without extension
            scene_id = filename.replace('.tif', '')

        # Match analytic images (both SR and TOA)
        if 'AnalyticMS' in filename or 'analytic' in filename.lower():
            if 'udm2' not in filename.lower():
                analytic_files[scene_id] = path
        elif 'udm2' in filename.lower():
            udm2_files[scene_id] = path

    # Apply masks and build results
    results = {}
    for scene_id in analytic_files:
        analytic_path = analytic_files[scene_id]
        udm2_path = udm2_files.get(scene_id)

        if udm2_path and os.path.exists(udm2_path):
            masked_path = analytic_path.replace('.tif', '_masked.tif')
            apply_udm2_mask(analytic_path, udm2_path, masked_path)
            os.remove(udm2_path)
            results[scene_id] = (analytic_path, masked_path)
        else:
            results[scene_id] = (analytic_path, None)

    return results


async def order_and_download_batch(
    item_ids: list,
    lat: float,
    lon: float,
    output_dir: str,
    order_name: str,
    anchor_id: str = None,
    product_type: str = 'SR'
) -> dict:
    """
    Order multiple PlanetScope scenes in a single batch, wait, and download all.

    Applies three processing tools (in this order):
    1. reproject - Puts all images on same north-aligned UTM grid at exactly 3m resolution
                   (uses "near" kernel to minimize blurring)
    2. coregister - Fine sub-pixel alignment to anchor (works better after reproject
                    since images are already on same grid)
    3. clip - Clips to AOI

    Args:
        item_ids: List of Planet scene IDs
        lat: Latitude for clipping AOI
        lon: Longitude for clipping AOI
        output_dir: Directory to save files
        order_name: Name for the order
        anchor_id: Scene ID to use as coregistration anchor (default: first in list)
        product_type: 'SR' (Surface Reflectance) or 'TOA' (Top of Atmosphere)

    Returns:
        dict: Mapping of item_id -> (unmasked_path, masked_path) for successful downloads
    """
    os.makedirs(output_dir, exist_ok=True)

    # Use first scene as anchor if not specified
    if anchor_id is None:
        anchor_id = item_ids[0]

    # Build order request using helper
    order_request = _build_order_request(item_ids, lat, lon, anchor_id, order_name, product_type=product_type)

    async with Session() as sess:
        client = OrdersClient(sess)

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

            # Download and process using helper
            return await _download_and_process_order(order_info, output_dir)

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


def align_to_common_grid(
    image: np.ndarray,
    src_transform: rasterio.Affine,
    src_crs,
    target_transform: rasterio.Affine,
    target_crs,
    target_shape: tuple,
    nodata: int = 0
) -> np.ndarray:
    """
    Resample an image to align with a target grid.

    Used to align images that were clipped before reprojection and ended up
    on slightly different UTM grids.

    Args:
        image: Source image array (bands, height, width)
        src_transform: Affine transform of source image
        src_crs: CRS of source image
        target_transform: Affine transform of target grid
        target_crs: CRS of target grid
        target_shape: (height, width) of target grid
        nodata: Nodata value (default 0)

    Returns:
        np.ndarray: Resampled image aligned to target grid
    """
    from rasterio.warp import reproject, Resampling

    dst_shape = (image.shape[0], target_shape[0], target_shape[1])
    dst_image = np.full(dst_shape, nodata, dtype=image.dtype)

    reproject(
        source=image,
        destination=dst_image,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=Resampling.nearest,
        src_nodata=nodata,
        dst_nodata=nodata
    )

    return dst_image


def compute_centered_grid(lat: float, lon: float, crs, target_size: int = PS_TARGET_SIZE, resolution: float = PS_RESOLUTION):
    """
    Compute a reference grid centered on the target lat/lon.

    Args:
        lat: Center latitude
        lon: Center longitude
        crs: Target CRS (from a downloaded image)
        target_size: Size in pixels (default PS_TARGET_SIZE)
        resolution: Pixel size in meters (default PS_RESOLUTION)

    Returns:
        rasterio.Affine: Transform centered on the target location
    """
    from pyproj import Transformer

    # Transform lat/lon to the target CRS
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    center_x, center_y = transformer.transform(lon, lat)

    # Compute top-left corner (origin)
    half_size_m = (target_size * resolution) / 2.0
    origin_x = center_x - half_size_m
    origin_y = center_y + half_size_m  # Y increases downward in image coords

    # Create transform: (resolution, 0, origin_x, 0, -resolution, origin_y)
    transform = rasterio.Affine(resolution, 0, origin_x, 0, -resolution, origin_y)

    logging.info(f"Centered grid: center ({center_x:.1f}, {center_y:.1f}), origin ({origin_x:.1f}, {origin_y:.1f})")

    return transform


#############################
# Time series stack download
#############################

async def search_best_scenes_for_windows(
    lat: float,
    lon: float,
    time_windows: list,
    max_cloud_cover: float = 0.5,
    product_type: str = 'SR'
) -> dict:
    """
    Search for the best scene for each time window.

    Args:
        lat: Latitude
        lon: Longitude
        time_windows: List of (start_date, end_date) tuples
        max_cloud_cover: Maximum cloud cover
        product_type: 'SR' (Surface Reflectance) or 'TOA' (Top of Atmosphere)

    Returns:
        dict: {window_index: {"item_id": str, "cloud_cover": float, "date_range": [str, str]}}
    """
    results = {}

    for i, (start, end) in enumerate(time_windows):
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        items = await search_scenes(lat, lon, start_str, end_str, max_cloud_cover, product_type=product_type)

        if items:
            best = items[0]  # Already sorted by effective_coverage
            results[i] = {
                "item_id": best['id'],
                "cloud_cover": best['properties'].get('cloud_cover', 0),
                "effective_coverage": best.get('effective_coverage', 100),
                "date_range": [start_str, end_str]
            }
            logging.info(f"Window {i}: found {len(items)} scenes, best effective_coverage={results[i]['effective_coverage']:.1f}%")
        else:
            results[i] = {
                "item_id": None,
                "cloud_cover": None,
                "effective_coverage": 0,
                "date_range": [start_str, end_str]
            }
            logging.warning(f"Window {i}: no scenes found for {start_str} to {end_str}")

    return results


async def retrieve_time_series_stack(
    file_id: str,
    lat: float,
    lon: float,
    date: datetime,
    out_dir: str,
    start_month: int = 1,
    num_windows: int = 36,
    timestep: int = 10,
    window_buffer: int = 3,
    target_size: int = PS_TARGET_SIZE,
    max_cloud_cover: float = 0.5,
    center_on_date: bool = False,
    product_type: str = 'SR'
):
    """
    Download and stack PlanetScope images over a time series (async).

    Use this in Jupyter notebooks with: await retrieve_time_series_stack(...)
    For CLI/scripts, use the sync wrapper: retrieve_time_series_stack_sync(...)

    Flow:
    1. Search for best scene for each time window
    2. Submit ONE batch order for all scenes
    3. Wait for processing (done in parallel on Planet's side)
    4. Download all results
    5. Stack into final output

    Args:
        file_id: Unique identifier for this site
        lat, lon: Coordinates (WGS84 decimal degrees)
        date: Reference date (year used for time series, or center date if center_on_date=True)
        out_dir: Directory to save final outputs
        start_month: Month to start downloading (1=January, excluding buffer). Ignored if center_on_date=True.
        num_windows: Number of timesteps to download (excluding buffer)
        timestep: Days per timestep
        window_buffer: Extra timesteps before/after
        target_size: Size in pixels for all images (default PS_TARGET_SIZE)
        max_cloud_cover: Maximum cloud cover for scene selection
        center_on_date: If True, center time windows on `date` instead of using `start_month`.
            This matches Sentinel-2's behavior with 36 windows centered on the labeled date.
        product_type: 'SR' (Surface Reflectance) or 'TOA' (Top of Atmosphere)

    Returns:
        None. Saves files to:
            - {out_dir}/{file_id}_stack.tif
            - {out_dir}/{file_id}_stack_masked.tif
            - {out_dir}/{file_id}_metadata.json
    """
    # Create time windows
    step_size = timedelta(days=timestep)
    num_windows_buffered = num_windows + (window_buffer * 2)

    if center_on_date:
        # Center the time windows on the labeled date
        # The labeled date should fall in window (num_windows // 2) of the core windows
        # With buffer, that becomes window (window_buffer + num_windows // 2)
        center_window = num_windows // 2
        start_date_dt = date - (center_window + window_buffer) * step_size
    else:
        # Legacy behavior: start from beginning of start_month
        start_date_dt = datetime(date.year, start_month, 1) - (step_size * window_buffer)
    time_windows = [
        (start_date_dt + i * step_size, start_date_dt + (i + 1) * step_size)
        for i in range(num_windows_buffered)
    ]

    # Create temporary directory for downloads
    temp_dir = os.path.join(out_dir, "_tmp", file_id)
    os.makedirs(temp_dir, exist_ok=True)

    # Step 1: Search for best scene for each window
    logging.info(f"Searching for {product_type} scenes across {num_windows_buffered} windows...")
    window_scenes = await search_best_scenes_for_windows(lat, lon, time_windows, max_cloud_cover, product_type=product_type)

    # Collect all scene IDs that were found and find best anchor
    item_ids = []
    window_to_item = {}  # Map window index -> item_id
    best_anchor = None
    best_coverage = 0

    for i, info in window_scenes.items():
        if info["item_id"]:
            item_ids.append(info["item_id"])
            window_to_item[i] = info["item_id"]
            # Track best scene for coregistration anchor
            if info.get("effective_coverage", 0) > best_coverage:
                best_coverage = info["effective_coverage"]
                best_anchor = info["item_id"]

    logging.info(f"Found scenes for {len(item_ids)}/{num_windows_buffered} windows")
    if best_anchor:
        logging.info(f"Using {best_anchor} as coregistration anchor ({best_coverage:.1f}% coverage)")

    # Step 2 & 3: Submit batch order and download
    downloaded = {}
    if item_ids:
        logging.info(f"Submitting {product_type} batch order for {len(item_ids)} scenes...")
        downloaded = await order_and_download_batch(
            item_ids, lat, lon, temp_dir, f"batch_{file_id}",
            anchor_id=best_anchor, product_type=product_type
        )
        logging.info(f"Downloaded {len(downloaded)} scenes")

    # Step 4: Build stacks with grid alignment
    # Compute a reference grid CENTERED on the target lat/lon
    num_bands = len(PS_BANDS)
    empty_template = np.zeros((num_bands, target_size, target_size), dtype=np.uint16)

    ref_transform = None
    ref_crs = None
    ref_shape = (target_size, target_size)

    images_unmasked = []
    images_masked = []
    metadata_windows = []

    for i, (start, end) in enumerate(time_windows):
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        item_id = window_to_item.get(i)
        paths = downloaded.get(item_id) if item_id else None

        if paths and paths[0] and os.path.exists(paths[0]):
            unmasked_path, masked_path = paths
            try:
                with rasterio.open(unmasked_path) as src:
                    unmasked_raw = src.read()
                    src_transform = src.transform
                    src_crs = src.crs

                    # First valid image: use its CRS but compute a CENTERED grid
                    if ref_transform is None:
                        ref_crs = src_crs
                        ref_transform = compute_centered_grid(lat, lon, ref_crs, target_size)

                # Align to reference grid (handles slight grid offsets between scenes)
                unmasked_img = align_to_common_grid(
                    unmasked_raw, src_transform, src_crs,
                    ref_transform, ref_crs, ref_shape, nodata=0
                )

                # Load and align masked image
                if masked_path and os.path.exists(masked_path):
                    with rasterio.open(masked_path) as src:
                        masked_raw = src.read()
                        masked_src_transform = src.transform
                        masked_src_crs = src.crs

                    masked_img = align_to_common_grid(
                        masked_raw, masked_src_transform, masked_src_crs,
                        ref_transform, ref_crs, ref_shape, nodata=0
                    )
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

    # Use reference CRS and transform for output
    template_crs = ref_crs
    template_transform = ref_transform

    if template_crs is None:
        logging.warning(f"No valid images found for site {file_id}, skipping stack creation")
        # Clean up temp files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return

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
        'product_type': product_type,
        'product_bundle': PRODUCT_TYPES[product_type]['bundle'],
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


def retrieve_time_series_stack_sync(
    file_id: str,
    lat: float,
    lon: float,
    date: datetime,
    out_dir: str,
    start_month: int = 1,
    num_windows: int = 36,
    timestep: int = 10,
    window_buffer: int = 3,
    target_size: int = PS_TARGET_SIZE,
    max_cloud_cover: float = 0.5,
    center_on_date: bool = False,
    product_type: str = 'SR'
):
    """
    Sync wrapper for retrieve_time_series_stack.

    Use this for CLI/scripts. For Jupyter notebooks, use the async version directly:
        await retrieve_time_series_stack(...)
    """
    return asyncio.run(retrieve_time_series_stack(
        file_id, lat, lon, date, out_dir,
        start_month, num_windows, timestep, window_buffer,
        target_size, max_cloud_cover, center_on_date, product_type
    ))


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
    target_size: int = PS_TARGET_SIZE,
    max_cloud_cover: float = 0.5,
    subset: bool = False,
    product_type: str = 'SR'
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
        target_size: Size in pixels (default PS_TARGET_SIZE)
        max_cloud_cover: Maximum cloud cover for scene selection
        subset: If True, only process first 10 rows (for testing)
        product_type: 'SR' (Surface Reflectance) or 'TOA' (Top of Atmosphere)

    Returns:
        str: Success message
    """
    initialize_planet()

    # Create version folder with product type suffix
    version_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{product_type}"
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
        'sensor': 'PlanetScope',
        'product_type': product_type,
        'product_bundle': PRODUCT_TYPES[product_type]['bundle']
    }
    metadata_path = os.path.join(out_dir, f"metadata_{version_name}.json")
    with open(metadata_path, 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Read input data
    data = pd.read_csv(csv)
    original_count = len(data)

    # Deduplicate by (site_id, year, month, day) to avoid downloading the same stack twice
    # This happens when multiple labelers labeled the same image
    data['_dedup_key'] = (data['site_id'].astype(str) + '_' +
                          data['year'].astype(str) + '_' +
                          data['month'].astype(str) + '_' +
                          data['day'].astype(str))
    data = data.drop_duplicates(subset='_dedup_key', keep='first')
    data = data.drop(columns=['_dedup_key'])

    if len(data) < original_count:
        logging.info(f"Deduplicated: {original_count} rows -> {len(data)} unique site+date combinations "
                     f"(removed {original_count - len(data)} duplicates)")

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
            retrieve_time_series_stack_sync(
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
                max_cloud_cover=max_cloud_cover,
                product_type=product_type
            )

            # Add unique_id to metadata (only if stack was created)
            metadata_file = os.path.join(out_dir, f"{file_id}_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                metadata['unique_id'] = int(uid) if str(uid).isdigit() else uid
                metadata['original_site_id'] = sid_raw

                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)

                logging.info(f"Completed {file_id}")
            else:
                logging.warning(f"No stack created for {file_id}")

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
    target_size: int = PS_TARGET_SIZE,
    max_cloud_cover: float = 0.5,
    max_concurrent_orders: int = 100,
    poll_interval: int = 60,
    product_type: str = 'SR',
    concurrent_scene_searches: int = 10
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
        product_type: 'SR' (Surface Reflectance) or 'TOA' (Top of Atmosphere)
        concurrent_scene_searches: Number of sites to search scenes for in parallel (default 5)
        (other args same as retrieve_time_series_stack)

    Returns:
        dict: {file_id: "success" | "failed" | error_message}
    """
    import shutil
    from collections import deque

    step_size = timedelta(days=timestep)
    num_windows_buffered = num_windows + (window_buffer * 2)

    async def search_site_scenes(site):
        """Search scenes for a single site. Returns (site, window_scenes, time_windows) or (site, None, error)."""
        file_id = site['file_id']
        lat, lon = site['lat'], site['lon']
        date = site['date']

        # Create time windows
        start_date_dt = datetime(date.year, start_month, 1) - (step_size * window_buffer)
        time_windows = [
            (start_date_dt + i * step_size, start_date_dt + (i + 1) * step_size)
            for i in range(num_windows_buffered)
        ]

        try:
            window_scenes = await search_best_scenes_for_windows(
                lat, lon, time_windows, max_cloud_cover, product_type=product_type
            )
            return (site, window_scenes, time_windows, None)
        except Exception as e:
            logging.error(f"{file_id}: Scene search failed: {e}")
            return (site, None, None, str(e))

    # Check for existing stacks (resume info)
    existing_count = sum(1 for s in sites if os.path.exists(os.path.join(out_dir, f"{s['file_id']}_stack.tif")))
    if existing_count > 0:
        logging.info(f"Resume mode: {existing_count}/{len(sites)} stacks already exist, will skip those")

    # Clean up stale _tmp folders from interrupted runs
    tmp_dir = os.path.join(out_dir, "_tmp")
    if os.path.exists(tmp_dir):
        stale_folders = os.listdir(tmp_dir)
        if stale_folders:
            logging.info(f"Cleaning up {len(stale_folders)} stale _tmp folders from interrupted run")
            shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir, exist_ok=True)

    # Track state
    pending_orders = {}  # order_id -> site_info
    results = {}
    sites_queue = deque(sites)

    async with Session() as sess:
        orders_client = OrdersClient(sess)

        while sites_queue or pending_orders:
            # Submit new orders up to limit, with parallel scene searching
            while sites_queue and len(pending_orders) < max_concurrent_orders:
                # Collect batch of sites to search in parallel
                batch = []
                while sites_queue and len(batch) < concurrent_scene_searches and len(pending_orders) + len(batch) < max_concurrent_orders:
                    site = sites_queue.popleft()
                    file_id = site['file_id']

                    # Skip if stack already exists (resume capability)
                    stack_path = os.path.join(out_dir, f"{file_id}_stack.tif")
                    if os.path.exists(stack_path):
                        logging.info(f"{file_id}: Stack already exists, skipping")
                        results[file_id] = "skipped_exists"
                        continue

                    batch.append(site)

                if not batch:
                    break

                # Search scenes for all sites in batch in parallel
                logging.info(f"Searching scenes for {len(batch)} sites in parallel...")
                search_results = await asyncio.gather(*[search_site_scenes(site) for site in batch])

                # Process results and submit orders
                for site, window_scenes, time_windows, error in search_results:
                    file_id = site['file_id']
                    lat, lon = site['lat'], site['lon']

                    if error:
                        results[file_id] = f"search_error: {error}"
                        continue

                    # Collect item IDs and find best anchor
                    item_ids = []
                    best_anchor = None
                    best_coverage = 0
                    for info in window_scenes.values():
                        if info["item_id"]:
                            item_ids.append(info["item_id"])
                            if info.get("effective_coverage", 0) > best_coverage:
                                best_coverage = info["effective_coverage"]
                                best_anchor = info["item_id"]

                    if not item_ids:
                        logging.warning(f"{file_id}: No scenes found")
                        results[file_id] = "no_scenes"
                        continue

                    # Build order request using shared helper
                    try:
                        anchor_id = best_anchor or item_ids[0]
                        order_request = _build_order_request(
                            item_ids, lat, lon, anchor_id, f"batch_{file_id}",
                            product_type=product_type
                        )

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

                    # Small delay between order submissions to avoid rate limits
                    await asyncio.sleep(0.1)

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
                            stack_created = await _download_and_stack_order(
                                order_info, site_info, out_dir, target_size, product_type
                            )
                            if stack_created:
                                results[site_info['file_id']] = "success"
                            else:
                                results[site_info['file_id']] = "no_valid_images"
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


async def _download_and_stack_order(order_info, site_info, out_dir, target_size, product_type='SR'):
    """Download order results and create stacked outputs.

    Returns:
        bool: True if stack was created, None if no valid images found.
    """
    file_id = site_info['file_id']
    temp_dir = os.path.join(out_dir, "_tmp", file_id)
    os.makedirs(temp_dir, exist_ok=True)

    # Download and process using shared helper
    processed = await _download_and_process_order(order_info, temp_dir)

    # Build stack with grid alignment
    # Compute a reference grid CENTERED on the target lat/lon
    window_scenes = site_info['window_scenes']
    time_windows = site_info['time_windows']
    lat = site_info['lat']
    lon = site_info['lon']

    num_bands = len(PS_BANDS)
    empty_template = np.zeros((num_bands, target_size, target_size), dtype=np.uint16)

    ref_transform = None
    ref_crs = None
    ref_shape = (target_size, target_size)

    images_unmasked = []
    images_masked = []
    metadata_windows = []

    for i, (start, end) in enumerate(time_windows):
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')

        item_id = window_scenes[i].get("item_id") if i in window_scenes else None
        paths = processed.get(item_id) if item_id else None

        if paths and paths[0] and os.path.exists(paths[0]):
            try:
                with rasterio.open(paths[0]) as src:
                    unmasked_raw = src.read()
                    src_transform = src.transform
                    src_crs = src.crs

                    # First valid image: use its CRS but compute a CENTERED grid
                    if ref_transform is None:
                        ref_crs = src_crs
                        ref_transform = compute_centered_grid(lat, lon, ref_crs, target_size)

                # Align to reference grid (handles slight grid offsets between scenes)
                unmasked_img = align_to_common_grid(
                    unmasked_raw, src_transform, src_crs,
                    ref_transform, ref_crs, ref_shape, nodata=0
                )

                # Load and align masked image
                if paths[1] and os.path.exists(paths[1]):
                    with rasterio.open(paths[1]) as src:
                        masked_raw = src.read()
                        masked_src_transform = src.transform
                        masked_src_crs = src.crs

                    masked_img = align_to_common_grid(
                        masked_raw, masked_src_transform, masked_src_crs,
                        ref_transform, ref_crs, ref_shape, nodata=0
                    )
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

    # Use reference CRS and transform for output
    template_crs = ref_crs
    template_transform = ref_transform

    if template_crs is None:
        logging.warning(f"No valid images for {file_id}, skipping stack creation")
        # Clean up temp files
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return

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
        'product_type': product_type,
        'product_bundle': PRODUCT_TYPES[product_type]['bundle'],
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
    return True


def dataset_download_parallel(
    csv: str,
    download_dir: str,
    max_concurrent_orders: int = 100,
    product_type: str = 'SR',
    resume_dir: str = None,
    concurrent_scene_searches: int = 10,
    **kwargs
) -> dict:
    """
    Download PlanetScope imagery for all sites in CSV using parallel ordering.

    Args:
        csv: Path to CSV with columns: x, y, unique_id, site_id, year, month, day
        download_dir: Output directory
        max_concurrent_orders: How many orders to have pending at once (default 20)
        product_type: 'SR' (Surface Reflectance) or 'TOA' (Top of Atmosphere)
        resume_dir: If provided, resume downloading into this existing directory
                    (e.g., '20260127_120000_SR'). Skips sites with existing stacks.
        concurrent_scene_searches: Number of sites to search scenes for in parallel (default 5).
                                   Higher values speed up scene searching but may hit API rate limits.
        **kwargs: Passed to process_sites_parallel

    Returns:
        dict: Results per site
    """
    initialize_planet()

    # Create or resume into version folder
    if resume_dir:
        out_dir = os.path.join(download_dir, resume_dir)
        if not os.path.exists(out_dir):
            raise ValueError(f"Resume directory does not exist: {out_dir}")
        logging.info(f"Resuming download into: {out_dir}")
    else:
        version_name = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{product_type}"
        out_dir = os.path.join(download_dir, version_name)
        os.makedirs(out_dir, exist_ok=False)

    # Read sites
    data = pd.read_csv(csv)
    original_count = len(data)

    # Deduplicate by (site_id, year, month, day) to avoid downloading the same stack twice
    # This happens when multiple labelers labeled the same image
    data['_dedup_key'] = (data['site_id'].astype(str) + '_' +
                          data['year'].astype(str) + '_' +
                          data['month'].astype(str) + '_' +
                          data['day'].astype(str))
    data = data.drop_duplicates(subset='_dedup_key', keep='first')
    data = data.drop(columns=['_dedup_key'])

    if len(data) < original_count:
        logging.info(f"Deduplicated: {original_count} rows -> {len(data)} unique site+date combinations "
                     f"(removed {original_count - len(data)} duplicates)")

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

    logging.info(f"Processing {len(sites)} sites with max {max_concurrent_orders} concurrent orders ({product_type})")

    # Run parallel processing
    results = asyncio.run(
        process_sites_parallel(sites, out_dir, max_concurrent_orders=max_concurrent_orders,
                               product_type=product_type,
                               concurrent_scene_searches=concurrent_scene_searches, **kwargs)
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

    # Download Surface Reflectance (SR) - atmospherically corrected, fewer scenes
    # This is the default and preferred for most applications
    dataset_download_parallel(
        csv=LABEL_CSV,
        download_dir=DOWNLOAD_DIR,
        max_concurrent_orders=50,
        product_type='SR',
        start_month=1,
        num_windows=36,
        timestep=10,
        window_buffer=3,
        max_cloud_cover=1.0  # Don't filter by scene-level clouds; rely on AOI effective_coverage
    )

    # Uncomment to also download TOA (Top of Atmosphere) - more scenes available
    # dataset_download_parallel(
    #     csv=LABEL_CSV,
    #     download_dir=DOWNLOAD_DIR,
    #     max_concurrent_orders=50,
    #     product_type='TOA',
    #     start_month=1,
    #     num_windows=36,
    #     timestep=10,
    #     window_buffer=3,
    #     max_cloud_cover=0.5
    # )
