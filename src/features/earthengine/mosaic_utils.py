import os
import json
import ee
from src.utils.utils import load_config, find_project_root
from src.utils.geometries import get_bounding_box


def initialize_earthengine():
    config = load_config()
    ee_key = os.path.join(find_project_root(os.getcwd()), config["earthengine"]["service_account_key"])

    with open(ee_key) as f:
        creds = json.load(f)
        service_email = creds['client_email']

    credentials = ee.ServiceAccountCredentials(service_email, ee_key)
    ee.Initialize(credentials)
    print("Earth Engine initialized.")


def download_sentinel2_mosaic(lat, lon, start_date, end_date, output_prefix=None):
    """
    Download Sentinel-2 mosaic for a 1km x 1km area at a given location and time window.
    """

    config = load_config()
    bucket = config["earthengine"]["bucket_name"]
    region = get_bounding_box(lat, lon)

    # Select Sentinel-2 SR (surface reflectance) with low cloud
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
        .select(['B2', 'B3', 'B4', 'B8'])  # Blue, Green, Red, NIR

    mosaic = collection.mosaic()

    if not output_prefix:
        output_prefix = f"sentinel2_mosaic_{lat}_{lon}_{start_date.replace('-', '')}"

    task = ee.batch.Export.image.toCloudStorage(
        image=mosaic,
        description=f"export_{output_prefix}",
        bucket=bucket,
        fileNamePrefix=output_prefix,
        region=region.getInfo()['coordinates'],
        scale=10,
        crs="EPSG:32633",
        maxPixels=1e13
    )

    task.start()
    print(f"Export started for ({lat}, {lon}) {start_date} – {end_date}")

    return task, output_prefix