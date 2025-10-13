#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentinel-2 time-series downloader & stacker

Data:
- Sentinel-2 L1C (COPERNICUS/S2_HARMONIZED) and cloud probability (COPERNICUS/S2_CLOUD_PROBABILITY)
- Sentinel-2 SR SCL (COPERNICUS/S2_SR_HARMONIZED): use class 3 (cloud shadow) when available

Method:
- 10-day windows (37 total) around the target date
- Server-side mask = s2cloudless + QA60 (cloud/cirrus) plus SR-SCL cloud shadow
- DOS-style brightness correction on B2-B12
- Compute NDVI, EVI, NDWI

Outputs (per window):
- <prefix>.tif          : 10 corrected bands + SCL
- <prefix>_masked.tif   : same, with cloud/cirrus/shadow set to NO_DATA

Final Output:
- Stacked 37-step time series (masked & unmasked) 
- JSON metadata (bands, shape, per-window stats)
"""

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

# project setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.utils import load_config, find_project_root
from src.utils.geometries import get_ee_bounding_box

config = load_config()
bucket = config["earthengine"]["bucket_name"]
ee_key = os.path.join(project_root, config["earthengine"]["service_account_key"])
fs = gcsfs.GCSFileSystem(token=ee_key, project="smallholder-irr")

LABEL_CSV    = os.path.join(project_root, "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv")
DOWNLOAD_DIR = os.path.join(project_root, "data/features")
TMP_DIR      = os.path.join(DOWNLOAD_DIR, "_tmp_tif")

MAX_PARALLEL_ROWS     = int(os.environ.get("MAX_PARALLEL_ROWS", "1"))
MAX_IN_FLIGHT_EXPORTS = int(os.environ.get("MAX_IN_FLIGHT_EXPORTS", "10"))

NO_DATA     = -9999
NUM_WINDOWS = 37

# 10 reflectance bands to keep
BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']
FINAL_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','NDVI','EVI','NDWI','SCL']

EXPORT_ROOT           = "masked_v2"
REPLACE_EXISTING_SITE = True
REQUIRE_VERSION_TAG   = True
VERSION_TAG           = os.environ.get("VERSION_TAG", "v1")

EXPORT_BOTH       = True
USE_SERVER_MASKED = True
USE_QA60_IN_MASK  = True

USE_SR_SCL_SHADOW = True

DROP_BY_MASK_FRAC = True
MASK_FRAC_THRESH  = 0.80

DROP_BY_COMPONENTS    = True
COMPONENT_DROP_THRESH = 0.80

# server-side mask tuning
W_S2C, W_THICK, W_THIN = 0.6, 0.3, 0.1
S2CLOUDLESS_SMOOTH_RADIUS = 2
QA_FOCAL_RADIUS           = 1
AUTO_S2CLOUDLESS      = True
S2CLOUDLESS_PROB_MIN  = 60
S2CLOUDLESS_PROB_BASE = 70
S2CLOUDLESS_PROB_MAX  = 80
BRIGHT_HIGH           = 0.35
BRIGHT_LOW            = 0.20
CLOUD_MASK_DILATE_PX  = 2
MIN_CLOUD_AREA_PX     = 100
CLOUD_GATE_NDVI_MAX   = 0.45
CLOUD_GATE_B11_MIN    = 0.08
T_THIN                = 0.50

SCL_MASK_CLASSES = [0, 1, 3, 9, 10, 11]
MASK_DILATE_RADIUS = 1

def_shape = (len(BANDS), 100, 100)

# EE init
def initialize_earthengine():
    key_path = os.path.join(find_project_root(os.getcwd()), config["earthengine"]["service_account_key"])
    with open(key_path) as f:
        creds = json.load(f)
        service_email = creds['client_email']
    credentials = ee.ServiceAccountCredentials(service_email, key_path)
    ee.Initialize(credentials)
    logging.info("Earth Engine initialized.")

# helpers
def _site_root_prefix(site_id: str) -> str:
    p = f"{EXPORT_ROOT}/{site_id}" if EXPORT_ROOT else site_id
    return f"{VERSION_TAG}/{p}" if VERSION_TAG else p

def gcs_prefix_exists(prefix: str) -> bool:
    try:
        return len(fs.ls(f"{bucket}/{prefix}")) > 0
    except Exception:
        return False

def gcs_delete_tree(prefix: str):
    try:
        fs.rm(f"{bucket}/{prefix}", recursive=True)
    except FileNotFoundError:
        pass

def sanitize_description(desc: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;_-")
    return ''.join([c if c in allowed else '_' for c in desc])[:95]

def _utm_epsg_from_latlon(lat: float, lon: float) -> str:
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone}" if lat >= 0 else f"EPSG:{32700 + zone}"

def ensure_all_bands(img: ee.Image) -> ee.Image:
    all_bands = ee.List(BANDS)
    present = img.bandNames()
    missing = all_bands.removeAll(present)
    def _adder(b, acc):
        acc = ee.Image(acc)
        b = ee.String(b)
        z = ee.Image.constant(0).rename(b).toUint16()
        return acc.addBands(z)
    zeros = ee.Image(ee.List(missing).iterate(_adder, ee.Image().select()))
    return img.addBands(zeros, overwrite=False).select(BANDS)

def pseudo_atmospheric_correction(image: ee.Image, region: ee.Geometry) -> ee.Image:
    bands = ['B2','B3','B4','B8']
    stats = image.reduceRegion(
        reducer=ee.Reducer.percentile([1]),
        geometry=region, scale=20, maxPixels=1e8
    )
    corrected_imgs = [image.select(b).subtract(ee.Number(stats.get(b))).rename(b) for b in bands]
    corrected = ee.Image.cat(corrected_imgs)
    return image.addBands(corrected, overwrite=True)

def choose_s2c_threshold(raw_toa: ee.Image, region: ee.Geometry) -> ee.Number:
    vis_mean = raw_toa.select(['B2','B3','B4']).divide(10000).reduce(ee.Reducer.mean())
    mean_b = ee.Number(vis_mean.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=region, scale=60, maxPixels=1e8
    ).values().get(0))
    th = ee.Number(S2CLOUDLESS_PROB_BASE)
    th = ee.Number(ee.Algorithms.If(mean_b.gt(BRIGHT_HIGH), S2CLOUDLESS_PROB_MAX, th))
    th = ee.Number(ee.Algorithms.If(mean_b.lt(BRIGHT_LOW),  S2CLOUDLESS_PROB_MIN,  th))
    return th

def _attach_s2cloudless_prob(collection: ee.ImageCollection, prob_col: ee.ImageCollection) -> ee.ImageCollection:
    joined = ee.ImageCollection(ee.Join.saveFirst('s2c').apply(
        primary=collection,
        secondary=prob_col,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
    ))
    def _add_prob(img):
        prob = ee.Image(ee.Algorithms.If(
            img.get('s2c'),
            ee.Image(img.get('s2c')).select('probability'),
            ee.Image.constant(0).toUint8()
        )).rename('S2CLOUDLESS')
        return ee.Image(img).addBands(prob)
    return joined.map(_add_prob)

# attach SR SCL
def _attach_sr_scl(l1c_col: ee.ImageCollection, sr_col: ee.ImageCollection) -> ee.ImageCollection:
    if not USE_SR_SCL_SHADOW:
        return l1c_col
    joined = ee.ImageCollection(ee.Join.saveFirst('sr').apply(
        primary=l1c_col,
        secondary=sr_col,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
    ))
    def _add_scl_sr(img):
        scl_sr = ee.Image(ee.Algorithms.If(
            img.get('sr'),
            ee.Image(img.get('sr')).select('SCL'),
            ee.Image(0)
        )).rename('SCL_SR')
        return ee.Image(img).addBands(scl_sr)
    return joined.map(_add_scl_sr)

def build_weighted_scl(raw_toa: ee.Image,
                       prob_band: ee.Image,
                       qa_cloud: ee.Image,
                       qa_cirrus: ee.Image,
                       region: ee.Geometry) -> ee.Image:
    # returns SCL with 7/9/10
    vis   = raw_toa.select(['B2','B3','B4']).divide(10000)
    blue  = raw_toa.select('B2').divide(10000)
    swir1 = raw_toa.select('B11').divide(10000)
    ndvi  = raw_toa.normalizedDifference(['B8','B4'])
    vis_mean = vis.reduce(ee.Reducer.mean())
    vis_std  = vis.reduce(ee.Reducer.stdDev())

    p_s2c = prob_band.reduceNeighborhood(
        reducer=ee.Reducer.mean(),
        kernel=ee.Kernel.square(S2CLOUDLESS_SMOOTH_RADIUS)
    ).divide(100.0)

    p_bright = vis_mean.subtract(0.35).divide(0.15).clamp(0,1)
    p_swir   = swir1.subtract(0.14).divide(0.10).clamp(0,1)
    p_white  = ee.Image(1).subtract(vis_std.divide(0.08).clamp(0,1))
    p_thick  = p_bright.multiply(0.6).add(p_swir.multiply(0.4)).multiply(p_white).clamp(0,1)

    if USE_QA60_IN_MASK:
        p_cirrus = qa_cirrus.reduceNeighborhood(
            reducer=ee.Reducer.max(),
            kernel=ee.Kernel.square(QA_FOCAL_RADIUS)
        ).unmask(0).toFloat()
    else:
        p_cirrus = ee.Image(0.0)

    p_blue  = blue.subtract(0.18).divide(0.10).clamp(0,1)
    p_ratio = blue.divide(swir1.add(1e-6)).subtract(1.5).divide(0.5).clamp(0,1)
    p_thin  = p_cirrus.max(p_blue.multiply(0.5).add(p_ratio.multiply(0.5))).clamp(0,1)

    veg_guard   = ndvi.lte(CLOUD_GATE_NDVI_MAX)
    swir_guard  = swir1.gte(CLOUD_GATE_B11_MIN)
    thick_guard = veg_guard.multiply(swir_guard)
    thin_guard  = veg_guard.max(p_cirrus.gt(0))

    p_thick = p_thick.updateMask(thick_guard)
    p_thin  = p_thin.updateMask(thin_guard)

    p_comb = (p_s2c.multiply(W_S2C)
              .add(p_thick.multiply(W_THICK))
              .add(p_thin.multiply(W_THIN))) \
              .divide(W_S2C + W_THICK + W_THIN) \
              .clamp(0,1)

    th_prob = ee.Number(ee.Algorithms.If(
        AUTO_S2CLOUDLESS, choose_s2c_threshold(raw_toa, region), S2CLOUDLESS_PROB_BASE
    )).divide(100.0)

    cloud   = p_comb.gte(th_prob)
    thinhit = p_thin.gte(T_THIN)
    cirrus  = thinhit.multiply(cloud.eq(0))

    if CLOUD_MASK_DILATE_PX > 0:
        k = ee.Kernel.square(CLOUD_MASK_DILATE_PX)
        cloud  = cloud.reduceNeighborhood(ee.Reducer.max(), k)
        cirrus = cirrus.reduceNeighborhood(ee.Reducer.max(), k)

    blobs = cloud.add(cirrus)
    conn  = blobs.connectedPixelCount(256, True)
    keep  = conn.gte(MIN_CLOUD_AREA_PX)
    cloud  = cloud.multiply(keep)
    cirrus = cirrus.multiply(keep)

    scl = ee.Image(7).where(cloud, 9).where(cirrus, 10)
    return scl.toUint16()

# per-window export
def export_window_best(lat: float, lon: float, s: str, e: str, prefix_base: str, region: ee.Geometry, out_dir: str):
    start_ee, end_ee = ee.Date(s), ee.Date(e)

    l1c_col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
               .filterBounds(region)
               .filterDate(start_ee, end_ee))
    if l1c_col.size().getInfo() == 0:
        logging.warning(f"[FALLBACK] No images for {lat},{lon} between {s} and {e}")
        return None, None

    prob_col = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
                .filterBounds(region).filterDate(start_ee, end_ee))

    # fetch and join SR SCL
    if USE_SR_SCL_SHADOW:
        sr_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(region).filterDate(start_ee, end_ee)
                  .select(['SCL']))
    else:
        sr_col = ee.ImageCollection([])

    col = _attach_s2cloudless_prob(l1c_col, prob_col)
    col = _attach_sr_scl(col, sr_col)

    def per_image(img):
        img = ee.Image(img)
        raw_toa = ensure_all_bands(img).select(BANDS)

        if USE_QA60_IN_MASK:
            qa60 = img.select('QA60')
            qa_cloud  = qa60.bitwiseAnd(1 << 10).neq(0)
            qa_cirrus = qa60.bitwiseAnd(1 << 11).neq(0)
        else:
            qa_cloud  = ee.Image(0)
            qa_cirrus = ee.Image(0)

        prob = img.select('S2CLOUDLESS')
        scl  = build_weighted_scl(raw_toa, prob, qa_cloud, qa_cirrus, region)

        # SR shadow union
        if USE_SR_SCL_SHADOW:
            shadow_sr = img.select('SCL_SR').eq(3)
            scl = scl.where(shadow_sr, 3)

        cloud_mask = scl.eq(9).max(scl.eq(10)).max(scl.eq(3))  # include shadows
        cf = cloud_mask.reduceRegion(ee.Reducer.mean(), region, 60, maxPixels=1e8).values().get(0)

        return img.addBands(raw_toa, overwrite=True)\
                  .addBands(scl.rename('SCL'))\
                  .set('cloud_frac', ee.Number(ee.Algorithms.If(cf, cf, 1.0)))

    scored = col.map(per_image)
    best = scored.sort('cloud_frac').first()

    raw  = ee.Image(best).select(BANDS)
    scl  = ee.Image(best).select('SCL')

    dos  = pseudo_atmospheric_correction(raw, region)\
             .max(ee.Image(0)).min(ee.Image(10000)).toUint16()

    # mask clouds + cirrus + shadow
    cloud_mask = scl.eq(9).max(scl.eq(10)).max(scl.eq(3))
    masked_ref = dos.where(cloud_mask, NO_DATA).toInt16()
    masked_scl = scl.where(cloud_mask, NO_DATA).toInt16()

    unmasked_img = dos.addBands(scl).toInt16()
    masked_img   = masked_ref.addBands(masked_scl.rename('SCL')).toInt16()

    # HTTP download locally
    if not os.path.exists(out_dir): os.makedirs(out_dir, exist_ok=True)

    un_file = f"{out_dir}/{prefix_base.split('/')[-1]}.tif"
    url_un = unmasked_img.getDownloadURL({
        'region': region, 'scale': 10, 'crs': _utm_epsg_from_latlon(lat, lon), 'format': 'GeoTIFF'
    })
    try:
        resp = requests.get(url_un); resp.raise_for_status()
        with open(un_file, 'wb') as f: f.write(resp.content)
    except Exception as e:
        logging.error(f"Failed to download unmasked image for {prefix_base}: {e}")
        return None, None

    ms_file = f"{out_dir}/{prefix_base.split('/')[-1]}_masked.tif"
    url_ms = masked_img.getDownloadURL({
        'region': region, 'scale': 10, 'crs': _utm_epsg_from_latlon(lat, lon), 'format': 'GeoTIFF'
    })
    try:
        resp = requests.get(url_ms); resp.raise_for_status()
        with open(ms_file, 'wb') as f: f.write(resp.content)
    except Exception as e:
        logging.error(f"Failed to download masked image for {prefix_base}: {e}")
        return None, None

# client helpers
def get_dense_time_windows(center_date: datetime):
    window = timedelta(days=10)
    total  = NUM_WINDOWS
    half   = timedelta(days=5)
    start  = center_date - (total // 2) * window - half
    return [(start + i * window, start + (i + 1) * window) for i in range(total)]

def calculate_indices(img10: np.ndarray):
    def m(a_int):
        return np.ma.masked_equal(a_int, NO_DATA).astype(np.float32) / 10000.0
    B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 = [m(b) for b in img10[:10]]
    ndvi = (B8 - B4) / (B8 + B4)
    evi  = 2.5 * (B8 - B4) / (B8 + 6.0 * B4 - 7.5 * B2 + 1.0)
    ndwi = (B8 - B11) / (B8 + B11)
    def to_int16(ma):
        ma = np.ma.clip(ma, -1.0, 1.0)
        out = np.full(ma.shape, NO_DATA, dtype=np.int16)
        valid = ~np.ma.getmaskarray(ma)
        out[valid] = (ma[valid] * 10000.0).astype(np.int16)
        return out
    return to_int16(ndvi), to_int16(evi), to_int16(ndwi)

def _wipe_slice_to_nodata():
    return np.full((len(FINAL_BANDS), 100, 100), NO_DATA, dtype=np.int16)

# stack builder
def retrieve_time_series_stack(site_id: str, lat: float, lon: float, date: datetime):
    if REQUIRE_VERSION_TAG and (VERSION_TAG is None or VERSION_TAG.strip() == ""):
        raise RuntimeError("VERSION_TAG must be set (bump it for each full run).")

    windows = get_dense_time_windows(date)
    region = get_ee_bounding_box(lat, lon)

    # submit & download per window via HTTP
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for start, end in windows:
            s, e = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
            base = f"{site_id}/s2_{lat:.2f}_{lon:.2f}_{s}_{e}"
            un_path = os.path.join(TMP_DIR, f"{base}.tif")
            ms_path = os.path.join(TMP_DIR, f"{base}_masked.tif")
            if not (os.path.exists(un_path) and os.path.exists(ms_path)):
                futures.append(executor.submit(export_window_best, lat, lon, s, e, base, region,
                                               os.path.join(TMP_DIR, site_id)))
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                logging.error(f"Error during export: {e}")

    stack_after, stack_before, meta_list = [], [], []
    empty_window_count = 0
    ref_crs = ref_transform = None

    for start, end in windows:
        s, e = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
        base = f"{site_id}/s2_{lat:.2f}_{lon:.2f}_{s}_{e}"
        un_path = os.path.join(TMP_DIR, f"{base}.tif")
        ms_path = os.path.join(TMP_DIR, f"{base}_masked.tif")

        if not os.path.exists(un_path):
            empty_window_count += 1
            stack_before.append(_wipe_slice_to_nodata())
            stack_after.append(_wipe_slice_to_nodata())
            meta_list.append({
                "date_range": [s, e],
                "cloud_fraction": 1.0,
                "masked_fraction": 1.0,
                "dropped_by_mask_frac": True if DROP_BY_MASK_FRAC else False,
                "drop_thresh": MASK_FRAC_THRESH,
                "mask_mode": "server" if (USE_SERVER_MASKED and EXPORT_BOTH) else "local",
                "mean_ndvi_after_mask": NO_DATA,
                "mean_evi_after_mask":  NO_DATA,
                "mean_ndwi_after_mask": NO_DATA,
                "mean_ndvi_before_mask": NO_DATA,
                "mean_evi_before_mask":  NO_DATA,
                "mean_ndwi_before_mask": NO_DATA
            })
            continue

        with rasterio.open(un_path) as src:
            if ref_crs is None:
                ref_crs = src.crs
                ref_transform = src.transform
            cube_un = src.read().astype(np.int16)

        if cube_un.shape[0] == len(BANDS) + 1:
            img_un = cube_un[:len(BANDS)]
            scl_un = cube_un[len(BANDS)]
        else:
            img_un = cube_un[:-1]
            scl_un = cube_un[-1]

        if img_un.shape != def_shape:
            img_un = np.stack([resize(img_un[b], def_shape[1:], preserve_range=True)
                               for b in range(img_un.shape[0])], axis=0).astype(np.int16)
        if scl_un.shape != (100, 100):
            scl_un = resize(scl_un, (100, 100), order=0, preserve_range=True,
                            anti_aliasing=False).astype(np.int16)

        band_to_idx = {b: i for i, b in enumerate(BANDS) if i < img_un.shape[0]}
        img10_un = np.stack([
            img_un[band_to_idx[b]] if b in band_to_idx else np.full((100, 100), NO_DATA, dtype=np.int16)
            for b in FINAL_BANDS[:10]
        ], axis=0).astype(np.int16)

        ndvi_un, evi_un, ndwi_un = calculate_indices(img10_un)

        before_slice = np.concatenate(
            (img10_un, ndvi_un[None], evi_un[None], ndwi_un[None], scl_un[None]), axis=0
        ).astype(np.int16)
        stack_before.append(before_slice)

        # pre-mask coverage (from unmasked SCL)
        cloud_frac_pre   = float((scl_un == 9).mean())
        cirrus_frac_pre  = float((scl_un == 10).mean())
        shadow_frac_pre  = float((scl_un == 3).mean())
        union_frac_pre   = float(((scl_un == 9) | (scl_un == 10) | (scl_un == 3)).mean())
        drop_by_components = bool(
            DROP_BY_COMPONENTS and (
                (cloud_frac_pre  >= COMPONENT_DROP_THRESH) or
                (shadow_frac_pre >= COMPONENT_DROP_THRESH) or
                (union_frac_pre  >= COMPONENT_DROP_THRESH)
            )
        )

        if USE_SERVER_MASKED and EXPORT_BOTH and os.path.exists(ms_path):
            with rasterio.open(ms_path) as srcm:
                cube_ms = srcm.read().astype(np.int16)
            if cube_ms.shape[0] == len(BANDS) + 1:
                img_ms = cube_ms[:len(BANDS)]
                scl_ms = cube_ms[len(BANDS)]
            else:
                img_ms = cube_ms[:-1]
                scl_ms = cube_ms[-1]

            if img_ms.shape != def_shape:
                img_ms = np.stack([resize(img_ms[b], def_shape[1:], preserve_range=True)
                                   for b in range(img_ms.shape[0])], axis=0).astype(np.int16)
            if scl_ms.shape != (100, 100):
                scl_ms = resize(scl_ms, (100, 100), order=0, preserve_range=True,
                                anti_aliasing=False).astype(np.int16)

            img10_ms = np.stack([
                img_ms[band_to_idx[b]] if b in band_to_idx else np.full((100, 100), NO_DATA, dtype=np.int16)
                for b in FINAL_BANDS[:10]
            ], axis=0).astype(np.int16)

            img10_ms = np.where(scl_ms[None, :, :] == NO_DATA, NO_DATA, img10_ms)

            ndvi_ms, evi_ms, ndwi_ms = calculate_indices(img10_ms)
            masked_fraction = float((scl_ms == NO_DATA).mean())
            cloud_frac = masked_fraction

            after_slice = np.concatenate(
                (img10_ms, ndvi_ms[None], evi_ms[None], ndwi_ms[None], scl_ms[None]), axis=0
            ).astype(np.int16)

            if drop_by_components or (DROP_BY_MASK_FRAC and (masked_fraction >= MASK_FRAC_THRESH)):
                after_slice = _wipe_slice_to_nodata()

            mask_mode = "server"
        else:
            classes = SCL_MASK_CLASSES
            combined_mask = np.isin(scl_un, classes)
            if MASK_DILATE_RADIUS > 0:
                k = 2 * MASK_DILATE_RADIUS + 1
                foot = footprint_rectangle((k, k))
                combined_mask = binary_dilation(combined_mask, footprint=foot)

            mask3 = combined_mask[None, :, :]
            img10_ms = np.where(mask3, NO_DATA, img10_un)

            ndvi_ms  = ndvi_un.copy();  ndvi_ms[combined_mask]  = NO_DATA
            evi_ms   = evi_un.copy();   evi_ms[combined_mask]   = NO_DATA
            ndwi_ms  = ndwi_un.copy();  ndwi_ms[combined_mask]  = NO_DATA
            scl_ms   = scl_un.copy();   scl_ms[combined_mask]   = NO_DATA

            masked_fraction = float(combined_mask.mean())
            cloud_frac = masked_fraction

            after_slice = np.concatenate(
                (img10_ms, ndvi_ms[None], evi_ms[None], ndwi_ms[None], scl_ms[None]), axis=0
            ).astype(np.int16)

            if drop_by_components or (DROP_BY_MASK_FRAC and (masked_fraction >= MASK_FRAC_THRESH)):
                after_slice = _wipe_slice_to_nodata()

            mask_mode = "local"

        stack_after.append(after_slice)

        def _mean_no_data(a):
            m = a[a != NO_DATA]
            return float(m.mean()) if m.size else NO_DATA

        meta_list.append({
            "date_range": [s, e],
            "cloud_fraction": cloud_frac,
            "masked_fraction": masked_fraction,
            "dropped_by_mask_frac": bool(DROP_BY_MASK_FRAC and (masked_fraction >= MASK_FRAC_THRESH)),
            "drop_thresh": MASK_FRAC_THRESH,
            "mask_mode": mask_mode,
            "mean_ndvi_after_mask": _mean_no_data(after_slice[10]),
            "mean_evi_after_mask":  _mean_no_data(after_slice[11]),
            "mean_ndwi_after_mask": _mean_no_data(after_slice[12]),
            "mean_ndvi_before_mask": _mean_no_data(before_slice[10]),
            "mean_evi_before_mask":  _mean_no_data(before_slice[11]),
            "mean_ndwi_before_mask": _mean_no_data(before_slice[12]),
            # >>> NEW: expose pre-mask component coverage + decision
            "cloud_frac_pre":  cloud_frac_pre,
            "cirrus_frac_pre": cirrus_frac_pre,
            "shadow_frac_pre": shadow_frac_pre,
            "union_frac_pre":  union_frac_pre,
            "dropped_by_components": drop_by_components,
        })

    return stack_after, stack_before, meta_list, empty_window_count, ref_crs, ref_transform

# row I/O
def process_row(row):
    lat, lon = row['y'], row['x']
    uid = row['unique_id']
    logging.info(f"Processing row {uid}")
    date = datetime(int(row['year']), int(row['month']), int(row['day']))

    date_str = f"{date.year}.{date.month:02d}.{date.day:02d}"
    sid_raw = str(row['site_id'])
    sid_for_name = sid_raw.replace('id_', '')
    file_prefix = f"{uid}_{sid_for_name}_{date_str}"

    site_id = f"site_{lat:.2f}_{lon:.2f}_{date.year}_{uid}"

    stack_after, stack_before, meta_list, empty_count, ref_crs, ref_transform = \
        retrieve_time_series_stack(site_id, lat, lon, date)

    arr_after  = np.stack(stack_after)
    arr_before = np.stack(stack_before)
    T, B, H, W = arr_after.shape
    expected_shape = (NUM_WINDOWS, len(FINAL_BANDS), 100, 100)
    if arr_after.shape != expected_shape or arr_before.shape != expected_shape:
        raise ValueError(f"Unexpected shapes for stacks: after={arr_after.shape}, before={arr_before.shape}, expected={expected_shape}")

    reshaped_after  = arr_after.transpose(1, 0, 2, 3).reshape(T*B, H, W)
    reshaped_before = arr_before.transpose(1, 0, 2, 3).reshape(T*B, H, W)

    out_image_tif      = os.path.join(DOWNLOAD_DIR, f"{file_prefix}_image.tif")
    out_image_json     = os.path.join(DOWNLOAD_DIR, f"{file_prefix}_image.json")
    out_unmasked_tif   = os.path.join(DOWNLOAD_DIR, f"{file_prefix}_unmasked.tif")
    out_unmasked_json  = os.path.join(DOWNLOAD_DIR, f"{file_prefix}_unmasked.json")

    write_crs = ref_crs if ref_crs is not None else None
    write_transform = ref_transform if ref_transform is not None else from_origin(
        lon - 0.0005,  # ~100 m at equator fallback
        lat + 0.0005, 0.0001, 0.0001
    )

    with rasterio.open(out_image_tif, 'w', driver='GTiff',
                       height=H, width=W, count=T*B, dtype='int16',
                       crs=write_crs, transform=write_transform, nodata=NO_DATA) as dst:
        dst.write(reshaped_after.astype('int16'))

    with rasterio.open(out_unmasked_tif, 'w', driver='GTiff',
                       height=H, width=W, count=T*B, dtype='int16',
                       crs=write_crs, transform=write_transform, nodata=NO_DATA) as dst:
        dst.write(reshaped_before.astype('int16'))

    base_meta = {
        "site_id": site_id,
        "lat": float(lat), "lon": float(lon),
        "year": int(date.year),
        "unique_id": int(uid) if str(uid).isdigit() else uid,
        "bands": FINAL_BANDS,
        "shape": list(arr_after.shape),
        "empty_window_count": int(empty_count),
        "version_tag": VERSION_TAG,
        "export_root": EXPORT_ROOT
    }

    windows_after, windows_before = [], []
    for w in meta_list:
        windows_after.append({
            "date_range": w["date_range"],
            "cloud_fraction": w["cloud_fraction"],
            "masked_fraction": w["masked_fraction"],
            "dropped_by_mask_frac": w["dropped_by_mask_frac"],
            "drop_thresh": w["drop_thresh"],
            "mean_ndvi": w["mean_ndvi_after_mask"],
            "mean_evi":  w["mean_evi_after_mask"],
            "mean_ndwi": w["mean_ndwi_after_mask"],
            "mask_mode": w["mask_mode"]
        })
        windows_before.append({
            "date_range": w["date_range"],
            "cloud_fraction": w["cloud_fraction"],
            "mean_ndvi": w["mean_ndvi_before_mask"],
            "mean_evi":  w["mean_evi_before_mask"],
            "mean_ndwi": w["mean_ndwi_before_mask"],
            "mask_mode": "before_mask"
        })

    with open(out_image_json, 'w') as f:
        json.dump({**base_meta, "dataset": "after_mask", "windows": windows_after}, f, indent=2)
    with open(out_unmasked_json, 'w') as f:
        json.dump({**base_meta, "dataset": "before_mask", "windows": windows_before}, f, indent=2)

    logging.info(f"[DONE] IMAGE (masked): {out_image_tif} + {out_image_json}")
    logging.info(f"[DONE] UNMASKED      : {out_unmasked_tif} + {out_unmasked_json}")
    return f"Processed row {uid} successfully"

# driver
def retrieve_images():
    logging.basicConfig(level=logging.INFO)
    os.makedirs(TMP_DIR, exist_ok=True)

    # (rasterio warning filter left as-is)
    logger = logging.getLogger("rasterio._env")
    logger.addFilter(lambda record: "Photometric type-related color channels" not in record.getMessage())

    if REQUIRE_VERSION_TAG and (VERSION_TAG is None or VERSION_TAG.strip() == ""):
        raise RuntimeError("VERSION_TAG must be set (bump it each full run).")

    initialize_earthengine()

    data = pd.read_csv(LABEL_CSV)
    logging.info(f"Starting to process {len(data)} rows from {LABEL_CSV}")

    rows = list(data.iterrows())
    for idx, row in rows:
        process_row(row)

if __name__ == "__main__":
    retrieve_images()