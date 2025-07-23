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
      - [Handling Missing and Invalid Data](#handling-missing-and-invalid-data)
    - [Stacking and Output](#stacking-and-output)
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

> **Note:** This module downloads dense Sentinel-2 mosaics for irrigation-labeled sites via Google Earth Engine. The pipeline is built for 2016–2025, supporting cloud/shadow filtering and NDVI/EVI/NDWI extraction across time.

To download features, we first load in all the irrigated images and their (lat, lon, date, ID) data from `data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv`. Then for each image, we generate a time series of images at the same location, with the middle of the time series being the date of the labeled image.

### Time Window Definition

For each labeled image, we generate 37 consecutive 10-day intervals around the observation date, with the center of the series being the date of the labeled image.

![time window graphic](./readme_figures/time_window.png)

Each of the 37 time windows corresponds to a single satellite image, which is a mosaic over that particular 10 day interval.

### Sentinel-2 Mosaic Retrieval

The satellite imagery we use for the time series is Sentinel-2 L1C data (available starting June 2015), retrieved through [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_HARMONIZED)

For each of the 37 ten-day windows in the time series, we generate a mosaic image spanning the interval, and save it to our Google Cloud Bucket. Each of these resultant images contains 13 bands: all 10 original Sentinel-2 bands and 3 derived bands to measure vegetation.

- **10 Original Sentinel-2 Bands**: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
- **NDVI: Normalized Difference Vegetation Index**: Measures green vegetation density, computed from NIR and Red bands (B8, B4).

$$
\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}
$$

- **EVI: Enhanced Vegetation Index**: Similar to NDVI, but slightly better in areas with dense canopy or haze. Computed from NIR, Red, and Blue bands (B8, B4, B2)

$$
\text{EVI} = 2.5 \times \frac{\text{NIR} - \text{Red}}{\text{NIR} + 6 \times \text{Red} - 7.5 \times \text{Blue} + 1}
$$

- **NDWI: Normalized Difference Water Index**: Detects moisture changes in vegetation and soil. Computed from NIR and SWIR1 bands (B8, B11).

$$
\text{NDWI} = \frac{\text{NIR} - \text{SWIR}}{\text{NIR} + \text{SWIR}}
$$

#### Handling Missing and Invalid Data

For a particular window, data may be missing (if there is no satellite imagery within that timeframe) or invalid (if there are clouds covering the image)

- **Cloud Detection** After retrieving all bands, we use module `s2cloudless` to detect pixels affected by clouds or cloud shadows. These pixels are set to -9999 across all bands to indicate missing/invalid data.

- **Missing Images** In rare cases, a time window may have no available satellite imagery. When this occurs, all pixels are assigned a value of -9999 to indicate missing data. Additionally, we set `cloud_fraction = 1.0` in the corresponding metadata file.

### Stacking and Output

The 37 tif images are stored on the Google Cloud Bucket, and NOT locally. However, these 37 images are combined and a file with their combined data is stored locally, as described below.

**Shape**: Each of the 37 .tif images is of size (13, 100, 100). We then stack these, resulting in final .tif image of size (37, 13, 100, 100), which we then flatten into shape (37 $\times$ 13, 100, 100). 

**Files Stored**: For each labeled image, we save two files to our data folder:

- `data/features/site_{lat}_{lon}_{year}_{ID}.tif` – The final .tif image of shape (37 $\times$ 13, 100, 100), which is a stack of all 37 retrieved images
- `data/features/site_{lat}_{lon}_{year}_{ID}.json` – Metadata for the .tif image, which includes cloud fraction, average NDVI/EVI/NDWI per frame

Note that lat and lon are the latitude and longitude of center of the labeled image, year is the year the labeled image was taken, and ID is the unique ID of the labeled image.

## Creating Pixel-Level Labels

For each Sentinel-2 image, we classify each pixel as irrigated or not. For irrigated pixels, we also specify the type of irrigation, the labeler's level of certainty, and reasons for any uncertainty. To do this, we overlay labeled polygons on an eight-band `.tif` file, with the following bands.

<img src="readme_figures/band_table.png" alt="table showing band information" width="600" />

The first band specifies the type of irrigation, if any, and the second is a simple binary mask of the first. These bands only include areas as irrigated if they clear a certain threshold of certainty, with the default being >=3. 

The next five bands are binary masks indicating the reasons for any uncertainty of the irrigation classification, with each band corresponding to a different uncertainty explanation. The last band indicates the certainty score, with 5 being high certainty, 1 being low certainty, and 0 indicating no irrigation. These bands include all areas regardless of their level of certainty.

The script will then create a folder `~/data/dataset/labels` containing all labels. For each input image, it will create a label file in format `uniqueID_siteID_date_labeler.tif` where
- `uniqueID` is a unique identifier for the label
-  `siteID` is the ID of the site
-  `date` is the date of the image (format `YYYY.MM.DD`)
-  `labeler` is the labeler's initials

To run this script, navigate to the `src` directory and run

```{bash}
python3 features/create_label_band.py
```

This will create a folder `~/data/dataset/labels` with all corresponding labels.

To run tests for this script, run the following command from this directory:

```{bash}
python -m unittest tests/test_create_label_band.py
```