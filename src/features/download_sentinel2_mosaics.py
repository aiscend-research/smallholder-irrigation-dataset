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

# Set up proxy if needed (adjust protocol if necessary)
#os.environ["HTTP_PROXY"] = "socks5://127.0.0.1:33210"
#os.environ["HTTPS_PROXY"] = "socks5://127.0.0.1:33210"

import rasterio
from skimage.transform import resize

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.utils import load_config
from src.features.earthengine.mosaic_download_utils import initialize_earthengine, download_sentinel2_mosaic

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def get_dense_time_windows(center_date):
    year_start = datetime(center_date.year, 1, 1)
    windows = []
    window_size = 10  # days
    n_windows = (365 // window_size) + 1
    for i in range(n_windows):
        start = year_start + timedelta(days=i * window_size)
        end = start + timedelta(days=window_size)
        windows.append((start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    return windows

def wait_for_task(task, poll_interval=30):
    if task is None:
        return {"state": "NO_TASK"}
    while task.active():
        logging.info("  EE task still running... waiting...")
        time.sleep(poll_interval)
    status = task.status()
    logging.info(f"  EE task finished with state: {status['state']}")
    if status["state"] != "COMPLETED":
        logging.error(f"  EE export failed with error: {status.get('error_message', 'Unknown error')}")
    return status

def resize_img(img, target_shape):
    """Resize a (bands, H, W) image to target_shape, preserving dtype."""
    bands, t_H, t_W = target_shape
    assert img.shape[0] == bands, f"Band count mismatch. Got {img.shape[0]}, expected {bands}"
    if img.shape == target_shape:
        return img
    resized = np.stack([
        resize(img[b], (t_H, t_W), order=1, preserve_range=True, anti_aliasing=True)
        for b in range(bands)
    ], axis=0).astype(img.dtype)
    return resized

LABEL_CSV = os.path.join(project_root, "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv")
DOWNLOAD_DIR = os.path.join(project_root, "data/features/")
GCS_PROJECT = "smallholder-irr"

# Temporary folder for intermediate single-frame tifs
TMP_DIR = os.path.join(DOWNLOAD_DIR, "_tmp_tif")
os.makedirs(TMP_DIR, exist_ok=True)

labels = pd.read_csv(LABEL_CSV)
assert all(col in labels.columns for col in ["unique_id", "y", "x", "year", "month", "day"]), "Check column names!"

initialize_earthengine()
bucket = load_config()["earthengine"]["bucket_name"]
ee_key = os.path.join(project_root, load_config()["earthengine"]["service_account_key"])
fs = gcsfs.GCSFileSystem(token=ee_key, project=GCS_PROJECT)

bands = ['B2', 'B3', 'B4', 'B8', 'QA60']

for idx, row in labels.iterrows():
    lat = row["y"]
    lon = row["x"]
    unique_id = row["unique_id"]  # Add unique_id for file naming
    target_date = datetime(int(row["year"]), int(row["month"]), int(row["day"]))
    windows = get_dense_time_windows(target_date)

    # Add unique_id at the end of the site id
    site_id = f"site_{lat:.2f}_{lon:.2f}_{target_date.year}_{unique_id}"
    stack_list = []
    meta_list = []

    logging.info(f"== Processing {site_id} ==")

    # We'll track previous shape for blank filling (default is None)
    last_valid_shape = None

    for widx, (start_date, end_date) in enumerate(windows):
        output_prefix = f"s2_{lat:.2f}_{lon:.2f}_{start_date}_{end_date}"
        gcs_prefix = f"{site_id}/{output_prefix}"

        tif_path = os.path.join(TMP_DIR, f"{output_prefix}.tif")
        meta_path = os.path.join(TMP_DIR, f"{output_prefix}.json")

        # Check if file exists in GCS (bucket-relative path)
        if fs.exists(f"{bucket}/{gcs_prefix}.tif"):
            logging.info(f"  [SKIP] File already exists in GCS: gs://{bucket}/{gcs_prefix}.tif")
            # Download from GCS to temporary local
            fs.get(f"{bucket}/{gcs_prefix}.tif", tif_path)
            fs.get(f"{bucket}/{gcs_prefix}.json", meta_path)
        else:
            logging.info(f"[{idx+1}/{len(labels)}-{widx+1}/{len(windows)}] Exporting: ({lat}, {lon}) {start_date}–{end_date}")

            try:
                task, out_prefix = download_sentinel2_mosaic(
                    lat, lon, start_date, end_date,
                    output_prefix=gcs_prefix,
                    bands=bands
                )
                if task is None:
                    logging.warning(f"[MISSING] No S2 images found for ({lat}, {lon}) {start_date}–{end_date}. Writing missing placeholder.")
                    shutil.copy("data/features/blank.tif", tif_path)
                    meta = {
                        "filename": os.path.basename(tif_path),
                        "location": {"lat": lat, "lon": lon},
                        "date_range": [start_date, end_date],
                        "bands": bands,
                        "nodata_cloudy": True,
                        "missing_data": True,
                        "description": "No Sentinel-2 mosaic found for this window.",
                        "source": "COPERNICUS/S2_SR_HARMONIZED"
                    }
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, indent=2)
                    # Optionally upload placeholder to GCS for traceability
                    fs.put(tif_path, f"{bucket}/{gcs_prefix}.tif")
                    fs.put(meta_path, f"{bucket}/{gcs_prefix}.json")
                else:
                    logging.info(f"  Started export for {output_prefix}")
                    wait_for_task(task, poll_interval=30)
                    logging.info(f"  Export complete for {output_prefix}")
                    # Download result from GCS
                    fs.get(f"{bucket}/{gcs_prefix}.tif", tif_path)
                    # Meta will be rewritten below
            except Exception as e:
                logging.error(f"  Failed to export {output_prefix}: {e}")
                shutil.copy("data/features/blank.tif", tif_path)
                meta = {
                    "filename": os.path.basename(tif_path),
                    "location": {"lat": lat, "lon": lon},
                    "date_range": [start_date, end_date],
                    "bands": bands,
                    "nodata_cloudy": True,
                    "missing_data": True,
                    "description": f"Error: {e}",
                    "source": "COPERNICUS/S2_SR_HARMONIZED"
                }
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

        # Read tif to numpy array, handling blank and resizing if needed
        try:
            with rasterio.open(tif_path) as src:
                img = src.read()  # shape: (bands, H, W)
                assert img.shape[0] == len(bands), "Band number mismatch!"
                # Force all images to last_valid_shape if last_valid_shape is set
                if last_valid_shape is not None and img.shape != last_valid_shape:
                    img = resize_img(img, last_valid_shape)
                last_valid_shape = img.shape  # Update the shape for the next placeholder if needed
        except Exception as e:
            logging.error(f"  Could not read tif {tif_path}: {e}")
            # Try to load blank.tif and resize to last_valid_shape or fallback to (bands, 512, 512)
            try:
                with rasterio.open("data/features/blank.tif") as blank_src:
                    blank_img = blank_src.read()
                # Determine target shape
                if last_valid_shape is not None:
                    target_shape = last_valid_shape
                else:
                    target_shape = (len(bands), 512, 512)
                img = resize_img(blank_img, target_shape)
            except Exception as e2:
                logging.error(f"  Could not load or resize blank.tif: {e2}")
                target_shape = last_valid_shape if last_valid_shape is not None else (len(bands), 512, 512)
                img = np.zeros(target_shape, dtype=np.uint16)

        # One final check to force all images to the same shape (last_valid_shape)
        if last_valid_shape is not None and img.shape != last_valid_shape:
            img = resize_img(img, last_valid_shape)
        elif last_valid_shape is None and img.shape != (len(bands), 512, 512):
            img = resize_img(img, (len(bands), 512, 512))

        # Check for missing or nodata frames
        try:
            qa60_band = img[-1]  # Last band is QA60
            nodata = (qa60_band == 1024).all() or (img[0] == 0).all()
        except Exception:
            nodata = True

        meta = {
            "filename": os.path.basename(tif_path),
            "location": {"lat": lat, "lon": lon},
            "date_range": [start_date, end_date],
            "bands": bands,
            "nodata_cloudy": bool(nodata),
            "missing_data": bool(nodata),
            "description": "Sentinel-2 mosaic (dense 10-day), with QA60 cloud band" if not nodata else "No valid image, placeholder.",
            "source": "COPERNICUS/S2_SR_HARMONIZED"
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        stack_list.append(img)
        meta_list.append(meta)

    for i, img in enumerate(stack_list):
        print(f"Frame {i}: shape={img.shape}")
    # Stack all images for this site/year and save as npy+json
    stack_arr = np.stack(stack_list, axis=0)  # shape: (n_time, bands, H, W)
    npy_save_path = os.path.join(DOWNLOAD_DIR, f"{site_id}_stack.npy")
    json_save_path = npy_save_path.replace(".npy", ".json")

    np.save(npy_save_path, stack_arr)
    with open(json_save_path, "w") as f:
        json.dump({
            "site_id": site_id,
            "lat": lat, "lon": lon, "year": target_date.year,
            "unique_id": unique_id,
            "bands": bands,
            "windows": meta_list,
            "shape": list(stack_arr.shape)
        }, f, indent=2)

    logging.info(f"[DONE] Saved stack: {npy_save_path}, shape={stack_arr.shape}")
    # Clean up temporary files
    for ftmp in os.listdir(TMP_DIR):
        if ftmp.startswith(f"s2_{lat:.2f}_{lon:.2f}_"):
            os.remove(os.path.join(TMP_DIR, ftmp))

logging.info("All mosaics processed and stacked!")
