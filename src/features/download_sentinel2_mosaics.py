#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sentinel-2 time-series downloader & stacker

Overview
- Per 10-day window, build a server-side cloud mask (SCL: 7/9/10) for each image
  and select the single lowest-cloud scene (no pixel-wise mosaics).
- Export two rasters per window to GCS:
  • <prefix>.tif          : DOS-corrected reflectance (10 bands) + SCL
  • <prefix>_masked.tif   : same, with cloud/cirrus written as NO_DATA
- Locally stack 37 steps and compute NDVI/EVI/NDWI.

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


# Remove if not needed
#os.environ.setdefault("HTTP_PROXY",  "socks5://127.0.0.1:33210")
#os.environ.setdefault("HTTPS_PROXY", "socks5://127.0.0.1:33210")

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

# Concurrency across rowss
MAX_PARALLEL_ROWS = int(os.environ.get("MAX_PARALLEL_ROWS", "1"))
# how many EE exports may be RUNNING at once
MAX_IN_FLIGHT_EXPORTS = int(os.environ.get("MAX_IN_FLIGHT_EXPORTS", "10"))

NO_DATA = -9999

# 10 reflectance bands to keep
BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']

# Final reflectance bands
FINAL_BANDS = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','NDVI','EVI','NDWI','SCL']

# Export control
EXPORT_ROOT           = "masked_v2"
REPLACE_EXISTING_SITE = True
REQUIRE_VERSION_TAG   = True
VERSION_TAG           = os.environ.get("VERSION_TAG", "v1")

EXPORT_BOTH       = True
USE_SERVER_MASKED = True
USE_QA60_IN_MASK  = True

DROP_BY_MASK_FRAC = True        # soft-drop step if NO_DATA fraction too high
MASK_FRAC_THRESH  = 0.80        # threshold on fraction of NO_DATA pixels in AFTER SCL

# Server-side mask tuning
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

# Expected single-image shape
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

# GCS helpers
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

# EE helpers
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

def build_weighted_scl(raw_toa: ee.Image,
                       prob_band: ee.Image,
                       qa_cloud: ee.Image,
                       qa_cirrus: ee.Image,
                       region: ee.Geometry) -> ee.Image:
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

# Export utilities 
def _inflight_exports() -> int:
    try:
        tasks = ee.data.getTaskList()
        return sum(t.get('state') in ('RUNNING', 'READY') for t in tasks)
    except Exception:
        return MAX_IN_FLIGHT_EXPORTS

def _wait_for_export_slot():
    while _inflight_exports() >= MAX_IN_FLIGHT_EXPORTS:
        time.sleep(15)

def wait_for_task(task, label="", poll_s=20):
    last = None
    while True:
        try:
            st = task.status()
            state = st.get("state", "UNKNOWN")
            if state != last:
                logging.info(f"[EE TASK] {label} -> {state}")
                last = state
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                if state != "COMPLETED":
                    logging.error(f"[EE TASK] {label} ended as {state}; details: {st}")
                return st
        except Exception as e:
            logging.warning(f"[EE TASK] {label} status error: {e}")
        time.sleep(poll_s)

def _poll_gcs_exists(gs_path: str, timeout_s=1200, interval_s=10) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if fs.exists(gs_path):
            return True
        time.sleep(interval_s)
    return fs.exists(gs_path)

# Per-window export (single best image)
def export_window_best(lat: float, lon: float, s: str, e: str, prefix_base: str, region: ee.Geometry):
    start_ee, end_ee = ee.Date(s), ee.Date(e)

    base_col = (ee.ImageCollection("COPERNICUS/S2_HARMONIZED")
                .filterBounds(region)
                .filterDate(start_ee, end_ee))
    if base_col.size().getInfo() == 0:
        logging.warning(f"[FALLBACK] No images for {lat},{lon} between {s} and {e}")
        return None, None

    prob_col = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
                .filterBounds(region).filterDate(start_ee, end_ee))
    col = _attach_s2cloudless_prob(base_col, prob_col)

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
        cloud_mask = scl.eq(9).max(scl.eq(10))
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

    cloud_mask = scl.eq(9).max(scl.eq(10))
    masked_ref = dos.where(cloud_mask, NO_DATA).toInt16()
    masked_scl = scl.where(cloud_mask, NO_DATA).toInt16()

    unmasked_img = dos.addBands(scl).toInt16()
    masked_img   = masked_ref.addBands(masked_scl.rename('SCL')).toInt16()

    desc_un = sanitize_description(prefix_base)
    task_un = ee.batch.Export.image.toCloudStorage(
        image=unmasked_img,
        description=f"export_un_{desc_un}",
        bucket=bucket,
        fileNamePrefix=prefix_base,
        region=region,
        scale=10,
        crs=_utm_epsg_from_latlon(lat, lon),
        maxPixels=1e13
    )
    task_un.start()

    task_ms = None
    if EXPORT_BOTH:
        masked_prefix = f"{prefix_base}_masked"
        desc_ms = sanitize_description(masked_prefix)
        task_ms = ee.batch.Export.image.toCloudStorage(
            image=masked_img,
            description=f"export_ms_{desc_ms}",
            bucket=bucket,
            fileNamePrefix=masked_prefix,
            region=region,
            scale=10,
            crs=_utm_epsg_from_latlon(lat, lon),
            maxPixels=1e13
        )
        task_ms.start()

    return task_un, task_ms

# Client-side helpers
def get_dense_time_windows(center_date: datetime):
    window = timedelta(days=10)
    total  = 37
    half   = timedelta(days=5)
    start  = center_date - (total // 2) * window - half
    return [(start + i * window, start + (i + 1) * window) for i in range(total)]

def calculate_indices(img10: np.ndarray):
    img = img10.astype(np.float32) / 10000.0
    B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 = img[:10]

    ndvi = np.full(B2.shape, NO_DATA, dtype=np.int16)
    evi  = np.full(B2.shape, NO_DATA, dtype=np.int16)
    ndwi = np.full(B2.shape, NO_DATA, dtype=np.int16)

    valid_ndvi = (B8 + B4) != 0
    valid_evi  = (B8 + 6*B4 - 7.5*B2 + 1) != 0
    valid_ndwi = (B8 + B11) != 0

    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.zeros_like(B2, dtype=np.float32)

        v.fill(0.0); num = (B8 - B4); den = (B8 + B4)
        v[valid_ndvi] = num[valid_ndvi] / den[valid_ndvi]
        v = np.clip(v, -1.0, 1.0); ndvi[valid_ndvi] = (v[valid_ndvi] * 10000).astype(np.int16)

        v.fill(0.0); num = (B8 - B4) * 2.5; den = (B8 + 6*B4 - 7.5*B2 + 1)
        ok = valid_evi; v[ok] = num[ok] / den[ok]
        v = np.clip(v, -1.0, 1.0); evi[ok] = (v[ok] * 10000).astype(np.int16)

        v.fill(0.0); num = (B8 - B11); den = (B8 + B11)
        ok = valid_ndwi; v[ok] = num[ok] / den[ok]
        v = np.clip(v, -1.0, 1.0); ndwi[ok] = (v[ok] * 10000).astype(np.int16)

    return ndvi, evi, ndwi

def _wipe_slice_to_nodata():
    """Return a full-slice array (14,H,W) filled with NO_DATA."""
    return np.full((len(FINAL_BANDS), 100, 100), NO_DATA, dtype=np.int16)

# Main stack builder 
def retrieve_time_series_stack(site_id: str, lat: float, lon: float, date: datetime):
    if REQUIRE_VERSION_TAG and (VERSION_TAG is None or VERSION_TAG.strip() == ""):
        raise RuntimeError("VERSION_TAG must be set (bump it for each full run).")

    site_root = _site_root_prefix(site_id)
    if REPLACE_EXISTING_SITE and gcs_prefix_exists(site_root):
        logging.info(f"[CLEAN] Removing existing site folder: gs://{bucket}/{site_root}/")
        gcs_delete_tree(site_root)

    windows = get_dense_time_windows(date)
    region = get_ee_bounding_box(lat, lon)

    # Submit exports (skip if both files already exist)
    task_records = []   
    for start, end in windows:
        s, e = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
        base = f"{site_root}/s2_{lat:.2f}_{lon:.2f}_{s}_{e}"

        exist_un = fs.exists(f"{bucket}/{base}.tif")
        exist_ms = fs.exists(f"{bucket}/{base}_masked.tif") if EXPORT_BOTH else True

        if exist_un and exist_ms:
            task_records.append((None, "unmasked", base, True))
            if EXPORT_BOTH:
                task_records.append((None, "masked", base, True))
            continue

        _wait_for_export_slot()
        t_un, t_ms = export_window_best(lat, lon, s, e, base, region)
        if t_un: task_records.append((t_un, "unmasked", base, False))
        if EXPORT_BOTH and t_ms: task_records.append((t_ms, "masked", base, False))

    # Wait for submitted tasks 
    final_states = {}  # (base, kind) -> "COMPLETED"/...
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {}
        for t, kind, base, preexist in task_records:
            if t:
                futures[ex.submit(wait_for_task, t, f"{kind}:{base}")] = (kind, base)
        for fut in as_completed(futures):
            kind, base = futures[fut]
            try:
                st = fut.result()
                final_states[(base, kind)] = st.get("state", "UNKNOWN")
            except Exception as e:
                logging.error(f"[EE WAIT] {kind}:{base} -> {e}")
                final_states[(base, kind)] = "FAILED"

    def should_fetch(base, kind, preexist):
        if preexist:
            return True
        return final_states.get((base, kind)) == "COMPLETED"

    # Poll GCS visibility ONLY for fetchable objects
    fetch_list = []  # (gs_path, local_path)
    for start, end in windows:
        s, e = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
        base = f"{site_root}/s2_{lat:.2f}_{lon:.2f}_{s}_{e}"

        if should_fetch(base, "unmasked", fs.exists(f"{bucket}/{base}.tif")):
            if _poll_gcs_exists(f"{bucket}/{base}.tif", timeout_s=1200):
                fetch_list.append((f"{bucket}/{base}.tif",
                                   os.path.join(TMP_DIR, f"{base}.tif")))
            else:
                logging.warning(f"[GCS VISIBILITY] not visible: {base}.tif")

        if EXPORT_BOTH and should_fetch(base, "masked", fs.exists(f"{bucket}/{base}_masked.tif")):
            if _poll_gcs_exists(f"{bucket}/{base}_masked.tif", timeout_s=1200):
                fetch_list.append((f"{bucket}/{base}_masked.tif",
                                   os.path.join(TMP_DIR, f"{base}_masked.tif")))
            else:
                logging.warning(f"[GCS VISIBILITY] not visible: {base}_masked.tif")

    # Download only visible ones
    os.makedirs(TMP_DIR, exist_ok=True)
    def _download(gs_path: str, local_path: str):
        os.makedirs(os.path.dirname(local_path), exist_ok=True
                    )
        fs.get(gs_path, local_path)

    with ThreadPoolExecutor(max_workers=16) as ex:
        futures = [ex.submit(_download, gs, loc) for gs, loc in fetch_list]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logging.error(f"[DOWNLOAD] {e}")

    # Build stacks
    stack_after, stack_before, meta_list = [], [], []
    empty_window_count = 0
    ref_crs = ref_transform = None

    for start, end in windows:
        s, e = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
        base = f"{site_root}/s2_{lat:.2f}_{lon:.2f}_{s}_{e}"
        un_path = os.path.join(TMP_DIR, f"{base}.tif")
        ms_path = os.path.join(TMP_DIR, f"{base}_masked.tif")

        # Missing both → write an empty slice
        if not os.path.exists(un_path):
            empty_window_count += 1
            stack_before.append(_wipe_slice_to_nodata())
            stack_after .append(_wipe_slice_to_nodata())
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

        # BEFORE (unmasked)
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

        # AFTER
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

            # force reflectance NO_DATA wherever SCL==NO_DATA 
            img10_ms = np.where(scl_ms[None, :, :] == NO_DATA, NO_DATA, img10_ms)

            ndvi_ms, evi_ms, ndwi_ms = calculate_indices(img10_ms)
            masked_fraction = float((scl_ms == NO_DATA).mean())
            cloud_frac = masked_fraction

            after_slice = np.concatenate(
                (img10_ms, ndvi_ms[None], evi_ms[None], ndwi_ms[None], scl_ms[None]), axis=0
            ).astype(np.int16)
            dropped = False

            # SOFT DROP: wipe the whole step if too blank, but keep index
            if DROP_BY_MASK_FRAC and (masked_fraction >= MASK_FRAC_THRESH):
                after_slice = _wipe_slice_to_nodata()
                dropped = True

        else:
            # Local fallback (rare)
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
            dropped = False
            if DROP_BY_MASK_FRAC and (masked_fraction >= MASK_FRAC_THRESH):
                after_slice = _wipe_slice_to_nodata()
                dropped = True

        stack_after.append(after_slice)

        # Per-window metrics
        def _mean_no_data(a):
            m = a[a != NO_DATA]
            return float(m.mean()) if m.size else NO_DATA

        meta_list.append({
            "date_range": [s, e],
            "cloud_fraction": cloud_frac,          # fraction used for "best image" scoring
            "masked_fraction": masked_fraction,    # fraction of NO_DATA in AFTER SCL
            "dropped_by_mask_frac": bool(DROP_BY_MASK_FRAC and (masked_fraction >= MASK_FRAC_THRESH)),
            "drop_thresh": MASK_FRAC_THRESH,
            "mask_mode": "server" if (USE_SERVER_MASKED and EXPORT_BOTH and os.path.exists(ms_path)) else "local",
            "mean_ndvi_after_mask": _mean_no_data(after_slice[10]),
            "mean_evi_after_mask":  _mean_no_data(after_slice[11]),
            "mean_ndwi_after_mask": _mean_no_data(after_slice[12]),
            "mean_ndvi_before_mask": _mean_no_data(before_slice[10]),
            "mean_evi_before_mask":  _mean_no_data(before_slice[11]),
            "mean_ndwi_before_mask": _mean_no_data(before_slice[12]),
        })

    return stack_after, stack_before, meta_list, empty_window_count, ref_crs, ref_transform

# Row processing & I/O
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

    arr_after  = np.stack(stack_after)   # (T, 14, 100, 100)
    arr_before = np.stack(stack_before)  # (T, 14, 100, 100)
    T, B, H, W = arr_after.shape
    expected_shape = (37, len(FINAL_BANDS), 100, 100)
    if arr_after.shape != expected_shape or arr_before.shape != expected_shape:
        raise ValueError(f"Unexpected shapes: after={arr_after.shape}, before={arr_before.shape}, expected={expected_shape}")

    reshaped_after  = arr_after.transpose(1, 0, 2, 3).reshape(T*B, H, W)
    reshaped_before = arr_before.transpose(1, 0, 2, 3).reshape(T*B, H, W)

    out_image_tif  = os.path.join(DOWNLOAD_DIR, f"{file_prefix}_image.tif")
    out_label_tif  = os.path.join(DOWNLOAD_DIR, f"{file_prefix}_label.tif")
    out_image_json = os.path.join(DOWNLOAD_DIR, f"{file_prefix}_image.json")
    out_label_json = os.path.join(DOWNLOAD_DIR, f"{file_prefix}_label.json")

    write_crs = ref_crs if ref_crs is not None else None
    write_transform = ref_transform if ref_transform is not None else from_origin(
        lon - 0.0005,  # ~100 m at equator fallback
        lat + 0.0005, 0.0001, 0.0001
    )

    # BEFORE
    with rasterio.open(out_image_tif, 'w', driver='GTiff',
                       height=H, width=W, count=T*B, dtype='int16',
                       crs=write_crs, transform=write_transform, nodata=NO_DATA) as dst:
        dst.write(reshaped_before.astype('int16'))

    # AFTER
    with rasterio.open(out_label_tif, 'w', driver='GTiff',
                       height=H, width=W, count=T*B, dtype='int16',
                       crs=write_crs, transform=write_transform, nodata=NO_DATA) as dst:
        dst.write(reshaped_after.astype('int16'))

    base_meta = {
        "site_id": site_id,
        "lat": float(lat), "lon": float(lon),
        "year": int(date.year),
        "unique_id": int(uid) if str(uid).isdigit() else uid,
        "bands": FINAL_BANDS,
        "shape": list(arr_after.shape),  # (T,B,H,W)
        "empty_window_count": int(empty_count),
        "version_tag": VERSION_TAG,
        "export_root": EXPORT_ROOT
    }

    windows_before, windows_after = [], []
    for w in meta_list:
        windows_before.append({
            "date_range": w["date_range"],
            "cloud_fraction": w["cloud_fraction"],
            "mean_ndvi": w["mean_ndvi_before_mask"],
            "mean_evi":  w["mean_evi_before_mask"],
            "mean_ndwi": w["mean_ndwi_before_mask"],
            "mask_mode": "before_mask"
        })
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

    with open(out_image_json, 'w') as f:
        json.dump({**base_meta, "dataset": "before_mask", "windows": windows_before}, f, indent=2)
    with open(out_label_json, 'w') as f:
        json.dump({**base_meta, "dataset": "after_mask", "windows": windows_after}, f, indent=2)

    logging.info(f"[DONE] BEFORE: {out_image_tif} + {out_image_json}")
    logging.info(f"[DONE] AFTER : {out_label_tif} + {out_label_json}")
    return f"Processed row {uid} successfully"

# Driver
def retrieve_images():
    logging.basicConfig(level=logging.INFO)
    os.makedirs(TMP_DIR, exist_ok=True)

    if REQUIRE_VERSION_TAG and (VERSION_TAG is None or VERSION_TAG.strip() == ""):
        raise RuntimeError("VERSION_TAG must be set (bump it each full run).")

    initialize_earthengine()

    data = pd.read_csv(LABEL_CSV)
    logging.info(f"Starting to process {len(data)} rows from {LABEL_CSV}")

    rows = list(data.iterrows())
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_ROWS) as executor:
        futures = {executor.submit(process_row, row): idx for idx, row in rows}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                logging.info(result)
            except Exception as e:
                logging.error(f"[ERROR] Row {idx}: {e}")

if __name__ == "__main__":
    retrieve_images()
