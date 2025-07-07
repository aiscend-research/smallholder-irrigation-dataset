import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
import gcsfs

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.utils import load_config
from src.features.earthengine.mosaic_utils import initialize_earthengine, download_sentinel2_mosaic

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# -------- Time Window Generator --------
def get_time_windows(target_date):
    # Granular close to label date, coarser farther away
    offsets = [-60, -30, -15, -10, 0, 10, 15, 30, 60]  # in days
    window_size = 10  # window size in days (can make larger for coarse intervals if desired)
    windows = []
    for offset in offsets:
        center = target_date + timedelta(days=offset)
        start = (center - timedelta(days=window_size // 2)).strftime('%Y-%m-%d')
        end = (center + timedelta(days=window_size // 2)).strftime('%Y-%m-%d')
        windows.append((start, end, offset))
    return windows

# -------- Helper: Poll for EE task completion --------
def wait_for_task(task, poll_interval=30):
    """Poll the Earth Engine export task until completion."""
    while task.active():
        logging.info("  EE task still running... waiting...")
        time.sleep(poll_interval)
    status = task.status()
    logging.info(f"  EE task finished with state: {status['state']}")
    if status["state"] != "COMPLETED":
        logging.error(f"  EE export failed with error: {status.get('error_message', 'Unknown error')}")
    return status

# -------- Parameters --------
LABEL_CSV = os.path.join(project_root, "data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv")
DOWNLOAD_DIR = os.path.join(project_root, "data/features/")
GCS_PROJECT = "smallholder-irr"

# -------- Main Workflow --------
# 1. Read label table
labels = pd.read_csv(LABEL_CSV)
assert "y" in labels.columns and "x" in labels.columns and "year" in labels.columns \
    and "month" in labels.columns and "day" in labels.columns, "Check column names!"

# 2. Initialize EE
initialize_earthengine()

# 3. Get bucket and GCS key
bucket = load_config()["earthengine"]["bucket_name"]
ee_key = os.path.join(project_root, load_config()["earthengine"]["service_account_key"])
fs = gcsfs.GCSFileSystem(token=ee_key, project=GCS_PROJECT)

# 4. Process each row
for idx, row in labels.iterrows():
    lat = row["y"]
    lon = row["x"]
    target_date_str = f"{int(row['year']):04d}-{int(row['month']):02d}-{int(row['day']):02d}"
    target_date = datetime.strptime(target_date_str, "%Y-%m-%d")

    windows = get_time_windows(target_date)
    for widx, (start_date, end_date, offset) in enumerate(windows):
        output_prefix = f"s2_{lat:.4f}_{lon:.4f}_{start_date}_{end_date}_off{offset:+d}"
        logging.info(f"[{idx+1}/{len(labels)}-{widx+1}/{len(windows)}] Exporting: ({lat}, {lon}) {start_date}–{end_date}")

        # 5. Export mosaic to GCS
        try:
            task, out_prefix = download_sentinel2_mosaic(lat, lon, start_date, end_date, output_prefix=output_prefix)
            logging.info(f"  Started export for {output_prefix}")
            # Automated polling
            wait_for_task(task, poll_interval=30)
            logging.info(f"  Export complete for {output_prefix}")
        except Exception as e:
            logging.error(f"  Failed to export {output_prefix}: {e}")
            continue

        # 6. Download the result from GCS
        gcs_file = f"{bucket}/{output_prefix}.tif"
        local_file = os.path.join(DOWNLOAD_DIR, f"{output_prefix}.tif")
        os.makedirs(os.path.dirname(local_file), exist_ok=True)

        try:
            logging.info(f"  Downloading from gs://{gcs_file} to {local_file} ...")
            fs.get(gcs_file, local_file)
            logging.info(f"  Download complete: {local_file}")
        except Exception as e:
            logging.error(f"  Failed to download {gcs_file}: {e}")

logging.info(" All mosaics processed!")
