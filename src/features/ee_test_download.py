import sys
import os

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.utils import load_config 
import ee
import json
import gcsfs

# === Load config ===
config = load_config()
ee_key = config["earthengine"]["service_account_key"] = os.path.join(
            project_root, config["earthengine"]["service_account_key"]
        )
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ee_key
bucket = config["earthengine"]["bucket_name"]

# === Authenticate using service account ===
with open(ee_key) as f:
    creds = json.load(f)
    service_email = creds['client_email']

credentials = ee.ServiceAccountCredentials(service_email, ee_key)
ee.Initialize(credentials)

# === Define image and region ===
# Sentinel-2 SR collection
collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
    .filterDate("2022-06-10", "2022-06-20") \
    .filterBounds(ee.Geometry.Point(32.5, -14.7)) \
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
    .sort("CLOUDY_PIXEL_PERCENTAGE")

# Grab the least cloudy image
image = collection.first()
image = image.select(['B2', 'B3', 'B4', 'B8'])
region = image.geometry().bounds()
crs = "EPSG:32633"
scale = 10

# === Export image to GCS ===
file_prefix = "test_s2_export"
task = ee.batch.Export.image.toCloudStorage(
    image=image,
    description="export_s2_one_image",
    bucket=bucket,
    fileNamePrefix=file_prefix,
    region=region.getInfo()["coordinates"],
    scale=scale,
    crs=crs,
    maxPixels=1e13
)

task.start()
print("✅ Earth Engine export started. Check task status in the EE Code Editor or wait for download.")

# === Wait for export to complete manually ===
input("🔄 Press Enter once the export has completed in the GCS bucket...")

# === Download from GCS to local/HPC folder ===
gcs_uri = f"gs://{bucket}/{file_prefix}.tif"
local_path = os.path.join(project_root, "data/features", f"{file_prefix}.tif")
os.makedirs(os.path.dirname(local_path), exist_ok=True)

print(f"⬇️ Downloading from {gcs_uri} to {local_path} ...")

fs = gcsfs.GCSFileSystem(project="smallholder-irr")

gcs_path = f"{bucket}/{file_prefix}.tif"
fs.get(gcs_path, local_path)
print("✅ Download complete.")