import os
import json
import ee
from src.utils.utils import load_config, find_project_root
from src.utils.geometries import get_bounding_box

def initialize_earthengine():
    """
    Initialize the Google Earth Engine API with service account credentials.
    Looks up the credentials file from the project config.
    """
    config = load_config()
    ee_key = os.path.join(find_project_root(os.getcwd()), config["earthengine"]["service_account_key"])

    with open(ee_key) as f:
        creds = json.load(f)
        service_email = creds['client_email']

    credentials = ee.ServiceAccountCredentials(service_email, ee_key)
    ee.Initialize(credentials)
    print("Earth Engine initialized.")

def sanitize_description(desc):
    """
    Clean a string for use as the Earth Engine task description.
    Keeps only allowed characters and limits length to 95 (Earth Engine max is 100).
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;_-")
    cleaned = ''.join([c if c in allowed else '_' for c in desc])
    return cleaned[:95]  # Safe margin for EE description

def download_sentinel2_mosaic(lat, lon, start_date, end_date, output_prefix=None):
    """
    Export a Sentinel-2 mosaic for a 1km x 1km region centered at (lat, lon)
    over the specified date window to Google Cloud Storage.

    Args:
        lat (float): Latitude of the center point.
        lon (float): Longitude of the center point.
        start_date (str): Start of the time window (YYYY-MM-DD).
        end_date (str): End of the time window (YYYY-MM-DD).
        output_prefix (str, optional): Prefix for exported file names and description.

    Returns:
        (ee.batch.Task, str): The Earth Engine export task object and the output prefix used.
    """
    config = load_config()
    bucket = config["earthengine"]["bucket_name"]
    region = get_bounding_box(lat, lon)

    # Select Sentinel-2 Surface Reflectance collection, filter by region, date, and cloud cover
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(region) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
        .select(['B2', 'B3', 'B4', 'B8'])  # Blue, Green, Red, NIR

    mosaic = collection.mosaic()

    if not output_prefix:
        output_prefix = f"sentinel2_mosaic_{lat}_{lon}_{start_date.replace('-', '')}"

    # Sanitize the output prefix for Earth Engine task description (must be <100 chars and valid chars)
    desc = sanitize_description(output_prefix)

    task = ee.batch.Export.image.toCloudStorage(
        image=mosaic,
        description=f"export_{desc}",  # Safe for Earth Engine API
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
