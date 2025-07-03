import sys
import os

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.utils import load_config 
from src.utils.earthengine import initialize_earthengine, download_sentinel2_mosaic
import gcsfs

initialize_earthengine()

task, output_prefix = download_sentinel2_mosaic(
    lat=-14.7,
    lon=32.5,
    start_date="2022-06-10",
    end_date="2022-06-20",
    output_prefix="test_s2_mosaic_export"
)

bucket = load_config()["earthengine"]["bucket_name"]

# === Wait for export to complete manually ===
input("🔄 Press Enter once the export has completed in the GCS bucket...")

# === Download from GCS to local/HPC folder ===
gcs_uri = f"gs://{bucket}/{output_prefix}.tif"
local_path = os.path.join(project_root, "data/features", f"{output_prefix}.tif")
os.makedirs(os.path.dirname(local_path), exist_ok=True)

print(f"⬇️ Downloading from {gcs_uri} to {local_path} ...")

ee_key = os.path.join(project_root, load_config()["earthengine"]["service_account_key"])
fs = gcsfs.GCSFileSystem(token=ee_key, project="smallholder-irr")

gcs_path = f"{bucket}/{output_prefix}.tif"
fs.get(gcs_path, local_path)
print("✅ Download complete.")