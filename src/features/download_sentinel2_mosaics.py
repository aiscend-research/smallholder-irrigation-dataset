import sys
import os
import pandas as pd
import time
import logging
import gcsfs
import json
import shutil
from datetime import datetime, timedelta

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

def gcs_object_path(rel_site_folder, output_prefix):
    return f"{rel_site_folder}/{output_prefix}.tif"

def gcs_meta_object_path(rel_site_folder, output_prefix):
    return f"{rel_site_folder}/{output_prefix}.json"

LABEL_CSV = os.path.join(project_root, "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv")
DOWNLOAD_DIR = os.path.join(project_root, "data/features/")
GCS_PROJECT = "smallholder-irr"

labels = pd.read_csv(LABEL_CSV)
assert all(col in labels.columns for col in ["y", "x", "year", "month", "day"]), "Check column names!"

initialize_earthengine()
bucket = load_config()["earthengine"]["bucket_name"]
ee_key = os.path.join(project_root, load_config()["earthengine"]["service_account_key"])
fs = gcsfs.GCSFileSystem(token=ee_key, project=GCS_PROJECT)

bands = ['B2', 'B3', 'B4', 'B8', 'QA60']

for idx, row in labels.iterrows():
    lat = row["y"]
    lon = row["x"]
    target_date = datetime(int(row["year"]), int(row["month"]), int(row["day"]))
    windows = get_dense_time_windows(target_date)

    for widx, (start_date, end_date) in enumerate(windows):
        output_prefix = f"s2_{lat:.4f}_{lon:.4f}_{start_date}_{end_date}"
        site_folder = os.path.join(DOWNLOAD_DIR, f"site_{lat:.4f}_{lon:.4f}_{target_date.year}")
        os.makedirs(site_folder, exist_ok=True)

        tif_path = os.path.join(site_folder, f"{output_prefix}.tif")
        meta_path = os.path.join(site_folder, f"{output_prefix}.json")
        rel_site_folder = os.path.relpath(site_folder, DOWNLOAD_DIR)
        gcs_tif = gcs_object_path(rel_site_folder, output_prefix)
        gcs_json = gcs_meta_object_path(rel_site_folder, output_prefix)

        # Check if file exists in GCS
        if fs.exists(gcs_tif, bucket=bucket):
            logging.info(f"  [SKIP] File already exists in GCS: gs://{bucket}/{gcs_tif}")
            continue

        logging.info(f"[{idx+1}/{len(labels)}-{widx+1}/{len(windows)}] Exporting: ({lat}, {lon}) {start_date}–{end_date}")

        try:
            task, out_prefix = download_sentinel2_mosaic(
                lat, lon, start_date, end_date,
                output_prefix=output_prefix,
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
                fs.put(tif_path, gcs_tif, bucket=bucket)
                fs.put(meta_path, gcs_json, bucket=bucket)
                logging.info(f"Uploaded blank placeholder and metadata to GCS: gs://{bucket}/{gcs_tif}")
                continue
            logging.info(f"  Started export for {output_prefix}")
            wait_for_task(task, poll_interval=30)
            logging.info(f"  Export complete for {output_prefix}")
        except Exception as e:
            logging.error(f"  Failed to export {output_prefix}: {e}")
            continue

        # After successful export, download from GCS to local
        try:
            logging.info(f"Downloading from gs://{bucket}/{gcs_tif} to {tif_path} ...")
            fs.get(gcs_tif, tif_path, bucket=bucket)
            logging.info(f"Download complete: {tif_path}")
        except Exception as e:
            logging.error(f"Failed to download {gcs_tif}: {e}")
            continue

        # NODATA/Cloud mask check (basic)
        nodata = False
        try:
            import rasterio
            with rasterio.open(tif_path) as src:
                qa60_band = src.read(src.count)
                nodata = (qa60_band == 1024).all() or (src.read(1) == 0).all()
        except Exception as e:
            logging.warning(f"  Could not check nodata/clouds: {e}")

        meta = {
            "filename": os.path.basename(tif_path),
            "location": {"lat": lat, "lon": lon},
            "date_range": [start_date, end_date],
            "bands": bands,
            "nodata_cloudy": bool(nodata),
            "missing_data": False,
            "description": "Sentinel-2 mosaic (dense 10-day), with QA60 cloud band",
            "source": "COPERNICUS/S2_SR_HARMONIZED"
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        fs.put(meta_path, gcs_json, bucket=bucket)
        logging.info(f"Metadata written and uploaded: {meta_path} (gs://{bucket}/{gcs_json})")

logging.info("All mosaics processed!")
