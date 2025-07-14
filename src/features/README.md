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

**Note:** This section describes the workflow and usage for exporting dense Sentinel-2 mosaics for all label points using the Earth Engine API and Google Cloud Storage. The workflow is designed for dense time series sampling—producing a 10-day interval sequence for each site and each year.

### How it works

- For every point in `latest_irrigation_table.csv`, the script generates ~36 time windows of 10 days for the full year (relative to the label date).

- For each window, it checks if a mosaic already exists in the GCS bucket. If yes, it skips to the next.

- If no data is available for a window (e.g. all cloudy), it skips and logs the window.

- Each valid window triggers an Earth Engine export of a Sentinel-2 surface reflectance mosaic,including the QA60 cloud mask band.

- Downloads all resulting .tif files to the `data/features/` folder.

- Writes out a .json metadata file for each image, recording key info like location, date window, band list, and nodata/cloud flags.

### File location

- Input labels:
`data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv`

- Downloaded features & metadata:
`data/features/s2_{lat}_{lon}_{start}_{end}_off{offset}.tif`
`data/features/s2_{lat}_{lon}_{start}_{end}_off{offset}.json`

## Creating Pixel-Level Labels

For each Sentinel-2 image, we need to classify each pixel within the image as irrigated or not irrigated. To do this, we take the labeled polygons corresponding to the image and classify pixels as irrigated if a polygon overlaps with the center of a pixel, otherwise, we label it as not irrigated.

Here is an example of an input image location and its corresponding labeled polygons:

![Input image location](readme_figures/input_image.png)

Then, the label image would be a binary mask like the following
![Label image example](readme_figures/label_image.png)

To run this script, navigate to the `src` directory and run

```{bash}
python3 features/create_label_band.py
```

This will create a folder `~/data/dataset/labels` with all corresponding labels.

After downloading the Sentinel-2 images, we then create labels for each pixel. To do this, we first iterate through all the `.tif` mosaic files, which are assumed to be located at `data/dataset/images`.

For each file, we extract the file data using (latitude, longitude, offset, start date, and end date) from the filename to retrieve the survey date. Then, we extract the `.tif` metadata, such that we can create labels at the resolution of the original `.tif`.

Then, we must link the (latitude, longitude, survey date) to its corresponding labelled polygons (`.geojson` file). 

Then, we retrieve the polygons from the corresponding `.geojson` file, only retrieving polygons greater than a specified certainty (default is 4+). We store these polygons in a `geopandas.geodataframe.GeoDataFrame`, and rasterize these polygons at the same resolution of the original image, as a binary mask – a given pixel is 1 if the polygon overlaps with it, 0 otherwise. Note that if a polygon only partially overlaps with a pixel, it will count as 1 only if the it overlaps with the center of the pixel.

Then, we save the binary mask into a new file, located in `data/dataset/labels`, with the filename the same as the original image.