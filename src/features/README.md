# Feature Downloading with Google Earth Engine

## Table of Contents
- [Feature Downloading with Google Earth Engine](#feature-downloading-with-google-earth-engine)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Google Cloud Setup](#google-cloud-setup)
    - [1. Create a GCP Project](#1-create-a-gcp-project)
    - [2. Create a GCS Bucket](#2-create-a-gcs-bucket)
    - [3. Create a Service Account](#3-create-a-service-account)
  - [Service account and GCS configuration](#service-account-and-gcs-configuration)
  - [Downloading Features](#downloading-features)
    - [Time Window Definition](#time-window-definition)
    - [Sentinel-2 Mosaic Retrieval](#sentinel-2-mosaic-retrieval)
      - [Atmospheric Correction](#atmospheric-correction)
      - [Retrieved Bands](#retrieved-bands)
      - [Handling Missing and Invalid Data](#handling-missing-and-invalid-data)
    - [Data Quality Assessment and Visualization](#data-quality-assessment-and-visualization)
      - [RGB Images Before Cloud Masking](#rgb-images-before-cloud-masking)
      - [RGB Images After Cloud Masking](#rgb-images-after-cloud-masking)
      - [NDVI Before Cloud Masking](#ndvi-before-cloud-masking)
      - [NDVI After Cloud Masking](#ndvi-after-cloud-masking)
    - [Stacking and Output](#stacking-and-output)
    - [Running the Download](#running-the-download)
    - [Dataset Location](#dataset-location)
  - [Creating Pixel-Level Labels](#creating-pixel-level-labels)

---

## Overview

The labels generated using Earth Collect do not include any features that can be used to train a model. We use Google Earth Engine (EE) to download features for model training, leveraging Google Cloud Storage (GCS) for storage and transfer.

---

## Prerequisites
- Access to Google Cloud Platform (GCP)
- Permissions to create projects, buckets, and service accounts
- Earth Engine and GCS APIs enabled

---

## Google Cloud Setup

> **Note:** The following setup is recommended if you plan to run downloads on an HPC (High-Performance Computing cluster) or a remote server, where browser-based authentication is not practical. If you are working locally on your own machine, you may be able to authenticate directly with your Google account using the Earth Engine Python API, and download data without setting up a GCS bucket or service account. See the [Earth Engine Python API authentication guide](https://developers.google.com/earth-engine/guides/python_install) for local setup instructions.

### 1. Create a GCP Project
- Go to [Google Cloud Console](https://console.cloud.google.com)
- Create a new project
- **Register your Google account and GCP project with [Google Earth Engine](https://signup.earthengine.google.com/)** (required to use the Earth Engine API; free for noncommercial use)
- Enable billing (required, but costs are minimal for this use case)
- Enable the following APIs:
  - Earth Engine API
  - Google Cloud Storage
  - Service Usage API

### 2. Create a GCS Bucket
- In the Cloud Console: **Storage > Buckets > Create**
- Choose:
  - Standard storage
  - A single region close to you or your HPC
- Example bucket name: `irr-earthengine-exports`

### 3. Create a Service Account
- Go to **IAM & Admin > Service Accounts**
- Click **Create Service Account**
- Name it (e.g., `earthengine-hpc-access`)
- Under roles, add:
  - Storage Object Admin
  - Earth Engine Resource Writer
- Click **Done**
- Go to your service account, create a JSON key, and download it

---

## Service account and GCS configuration

> **Note:** The configuration below is required for workflows using a service account and GCS bucket (recommended for HPC/remote use). For local-only workflows, you may not need these settings—refer to the [Earth Engine documentation](https://developers.google.com/earth-engine/guides/python_install) for local authentication options.

Store the following information in your `config.yaml` file:

```yaml
earthengine:
  service_account_key: secrets/earthengine-key.json
  bucket_name: irr-earthengine-exports
```

- `service_account_key`: Path to your downloaded service account JSON key
- `bucket_name`: Name of your GCS bucket

**Note:** The key is typically stored in the `secrets/earthengine-key.json` file (or as specified in your config).

---

## Downloading Features

> **Note:** This module builds dense Sentinel-2 time series for irrigation-labeled sites via Google Earth Engine (GEE). It supports 2016–2025 and applies server-side cloud screening. Each time window uses the single best scene (no pixel-wise mosaic) to avoid seam artifacts.

To download features, we first load in all the irrigated images and their (lat, lon, date, ID) data from `data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv`. For each site, we generate a fixed-length time series.

### Time Window Definition

We create **42 consecutive 10-day windows** starting from January 1st of the labeled year (with a 3-window buffer before, so starting in early December of the previous year). The default configuration is:
- `num_windows=36` core windows (full year coverage)
- `window_buffer=3` extra windows before and after
- Total: 42 windows

Each window selects one Sentinel-2 scene with the most good-quality pixels inside the 100×100 region (no pixel-level mosaic within the window).

### Sentinel-2 Mosaic Retrieval

We retrieve Sentinel-2 imagery from either:
- **L1C**: `COPERNICUS/S2_HARMONIZED` (Top-of-Atmosphere reflectance) - available 2016+
- **L2A**: `COPERNICUS/S2_SR_HARMONIZED` (Surface Reflectance) - available 2018+

For every 10-day window in the 42-step series, we search for scenes intersecting the site. Each scene is scored by the number of good-quality pixels after cloud masking, and we select the single best scene for that window (no pixel-wise mosaicking).

**Cloud Masking:**
- **L2A**: Uses the Scene Classification Layer (SCL) band to identify good pixels (vegetation, bare soil, water, unclassified, snow)
- **L1C**: Uses QA60 band (cloud/cirrus flags) combined with a custom spectral cloud detector that identifies bright, uniform pixels with low SWIR reflectance. A 2km buffer is applied around detected clouds to catch shadows.

For each window we download two rasters directly from GEE:
- `<prefix>.tif` - Unmasked: 10 reflectance bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
- `<prefix>_masked.tif` - Masked: Same 10 bands with cloud/bad pixels set to 0

These per-window images are then stacked locally into the final time series.

#### Retrieved Bands

The final `.tif` files contain only the **10 original Sentinel-2 reflectance bands**:

| Band | Name | Resolution | Description |
|------|------|------------|-------------|
| B2 | Blue | 10m | Blue visible |
| B3 | Green | 10m | Green visible |
| B4 | Red | 10m | Red visible |
| B5 | Red Edge 1 | 20m* | Vegetation red edge |
| B6 | Red Edge 2 | 20m* | Vegetation red edge |
| B7 | Red Edge 3 | 20m* | Vegetation red edge |
| B8 | NIR | 10m | Near infrared |
| B8A | NIR Narrow | 20m* | Near infrared narrow |
| B11 | SWIR 1 | 20m* | Short-wave infrared |
| B12 | SWIR 2 | 20m* | Short-wave infrared |

*20m bands are resampled to 10m resolution during download.

**Note:** Vegetation indices (NDVI, EVI, NDWI) are **not** pre-computed in the downloaded data. If needed for modeling, compute them from the raw bands:
- NDVI = (B8 - B4) / (B8 + B4)
- EVI = 2.5 × (B8 - B4) / (B8 + 6×B4 - 7.5×B2 + 10000)
- NDWI = (B8 - B11) / (B8 + B11)

#### Handling Missing and Invalid Data

For a particular window, data may be missing (if there is no satellite imagery within that timeframe) or invalid (if there are clouds covering the image).

- **Cloud/cirrus pixels**: In the masked stack (`*_stack_masked.tif`), cloud and bad pixels are set to **0** (nodata value).

- **Missing Images**: When no satellite imagery is available for a time window, an all-zero slice is written and `file_exists: false` is recorded in the metadata JSON.

- **Masked fraction tracking**: The metadata JSON records `masked_fraction` for each window, indicating what percentage of pixels were masked due to clouds or other quality issues.

### Data Quality Assessment and Visualization

The downloaded time series data can be visualized and analyzed for quality assessment using the provided visualization tools. These tools help researchers understand temporal patterns, data coverage, and seasonal variations in the satellite imagery.

`visualize_tif.py` is used to visualize an image of a specific ID. After downloading that image to the data root, to visualize it:

```shell
python3 visualize_tif.py {uid_of_image_to_visualize}
```

Visualizations of RGB/NDVI before and after masking are downloaded to the data root, under `features/visualization`.  

#### RGB Images Before Cloud Masking
Shows the raw Sentinel-2 RGB composite (B4=Red, B3=Green, B2=Blue) before cloud masking is applied:

![RGB Before Cloud Masking](./readme_figures/uid1_rgb_before_masked.png)

#### RGB Images After Cloud Masking
Demonstrates the improvement in image quality after cloud masking, with cloudy pixels set to transparent:

![RGB After Cloud Masking](./readme_figures/uid1_rgb_after_masked.png)

#### NDVI Before Cloud Masking
Shows NDVI values across all time steps without cloud masking, revealing temporal patterns in vegetation:

![NDVI Before Cloud Masking](./readme_figures/uid1_ndvi_before_masked.png)

#### NDVI After Cloud Masking
Displays clean NDVI time series with cloud-masked pixels removed, providing clear vegetation dynamics:

![NDVI After Cloud Masking](./readme_figures/uid1_ndvi_after_masked.png)

These visualizations help researchers:
- Assess data quality for machine learning training
- Identify optimal time periods for irrigation detection
- Understand seasonal patterns in satellite data coverage
- Validate the effectiveness of cloud masking algorithms

### Stacking and Output

The per-window GeoTIFFs live in GCS; we then build fixed-length local stacks:

- Per step layout (local): 10 reflectance + 3 indices + 1 SCL = 14 bands

- Stack shape: (T, B, H, W) = (37, 14, 100, 100) → flattened to (37×14=518, 100, 100) in the final GeoTIFF.

**Files Stored**: For each labeled image, we save four files to our data folder:

- `data/features/{uid}_{site}_{YYYY.MM.DD}_image.tif` – BEFORE stack (unmasked scene + indices)
- `data/features/{uid}_{site}_{YYYY.MM.DD}_label.tif` – AFTER stack (masked scene + indices)
- `data/features/{uid}_{site}_{YYYY.MM.DD}_image.json` – metadata per step (cloud fraction, mean NDVI/EVI/NDWI, etc.)
- `data/features/{uid}_{site}_{YYYY.MM.DD}_label.json` – same fields for AFTER

### Running the Download
Using a single CPU, the dataset takes a little over a day to download. To run this on the HPC, create a `.sh` file in `src/features` with your desired headers & the commands below:

```shell
source ../../env/bin/activate

## run python 
python3 download_sentinel2_mosaics.py
```

Then use the [Slurm job scheduler](https://slurm.schedmd.com/sbatch.html) to schedule the download.

### Dataset Location
The dataset is located on the cluster at `/home/waves/data/smallholder-irrigation-dataset/data/`. There are three versions: 

|Folder Name|Date Downloaded|
|-----------|-----------|
|features|August 5th, 2025|
|features_v2|October 12, 2025|
|features_v3|November 9, 2025|

A note on the differences between versions:
- `features` downloaded everything locally and also to the Google Cloud Bucket, which made the download speed exponentially slower (took around a week to download all sites)
- `features_v2` downloads everything locally (bypassing Google Cloud Bucket), which makes downloading a lot easier. It takes a little less than a 1 minute to download each row of `latest_irrigation_table.csv`, whereas it took approximately 15 minutes in the previous download.
  - Modifications were made to `latest_irrigation_table.csv` between downloads, so the IDs in `features` do not match with those in `features_v2`
- `features` has some issues with the cloud masking in which a lot of images were mostly blank, save a few small patches of colored pixels. Some updates to improve the cloud masking in `features_v2`. Also, images more than 80% blank are dropped in `features_v2`, whereas they were kept in `features`
- `features_v3` uses the same download logic as `features_v2`, except all time windows begin on January 1st.

## Creating Pixel-Level Labels

For each Sentinel-2 stack, we create pixel-level irrigation labels by rasterizing the labeled polygons. The script creates **separate label files for each labeler** who annotated a given site-date, enabling multi-labeler comparison and consensus analysis.

### Label Band Structure (9 bands)

| Band | Name | Description | Values |
|------|------|-------------|--------|
| 1 | Categorical irrigation | Type of irrigation | 0=none, 1=small-scale, 2=tree_crop, 3=industrial, 4=lawn, 5=covered |
| 2 | Binary irrigation mask | Simple presence/absence | 0=no irrigation, 1=irrigated |
| 3 | Uncertainty: unclear agriculture | Labeler uncertainty flag | 0/1 |
| 4 | Uncertainty: only slightly green | Labeler uncertainty flag | 0/1 |
| 5 | Uncertainty: uneven | Labeler uncertainty flag | 0/1 |
| 6 | Uncertainty: may be natural | Labeler uncertainty flag | 0/1 |
| 7 | Uncertainty: may be fishpond | Labeler uncertainty flag | 0/1 |
| 8 | Certainty score | Labeler confidence | 0=no polygon, 1-5 (5=highest) |
| 9 | Polygon coverage % | Fraction of pixel covered | 0-100 (for mixed pixel analysis) |

**Notes:**
- Bands 1-2 only include polygons with certainty >= 3 (configurable threshold)
- Bands 3-8 include all polygons regardless of certainty
- Band 9 uses 10x supersampling to calculate actual pixel coverage percentage, enabling identification of mixed pixels (partially covered by polygons)

### Mixed Pixel Handling

For pixels that are only partially covered by a polygon:
- **Bands 1-2**: Use center-point approach (pixel labeled based on whether its center falls inside the polygon)
- **Band 9**: Shows actual coverage percentage (0-100%), calculated via supersampling

This allows downstream analysis to:
- Filter to only fully-covered pixels (coverage = 100%)
- Weight samples by coverage confidence
- Identify edge pixels for special handling

### Output Files

For each stack file and each labeler who annotated that site-date, the script creates:
```
{unique_id}_{site_id}_{YYYY.MM.DD}_{operator}_labels.tif
```

For example, if site `5133803` on `2018-10-03` was labeled by KL, AB, and JL:
```
100_5133803_2018.10.03_KL_labels.tif
100_5133803_2018.10.03_AB_labels.tif
100_5133803_2018.10.03_JL_labels.tif
```

**BUG:** The unique ID in the main dataset file do not necessarily correspond to the features and their labels and should therefore be ignored!

**Important:** Images labeled as "no irrigation" (no polygons) also get label files with all zeros, ensuring complete coverage for model training.

### Running the Script

```bash
# From project root
python src/features/create_label_band.py --download_dir data/features --version 20260107_180813

# Or use defaults (latest version)
python src/features/create_label_band.py
```

**From Python:**
```python
from src.features.create_label_band import create_labels
create_labels('data/features', '20260107_180813')
```

### Current Dataset

Labels for the `20260107_180813` feature download are stored alongside the stack files:
- **Total label files**: 3,536
- **With irrigation polygons**: 1,244
- **Empty labels (no irrigation)**: 2,292

---

## Visualization Tools

### GEE Screenshot Visualization

`gee_screenshot_visualization.py` provides functions to overlay labeled polygons on Google Earth Pro screenshots for publication-quality figures.

```python
from src.features.gee_screenshot_visualization import (
    list_available_screenshots,
    plot_screenshot_with_polygons,
    plot_random_screenshot
)

# List available screenshots
screenshots = list_available_screenshots()

# Plot a specific screenshot with polygon overlays from all labelers
plot_screenshot_with_polygons(
    survey='201-225',
    internal_id=13,
    month=7, day=31, year=2023
)

# Or plot a random one
plot_random_screenshot()
```

Screenshots should be placed in `data/labels/GEE_screenshots/` with naming format:
`{survey}_{internal_id}_{MM-DD-YY}.png`
