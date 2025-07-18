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
    - [How it works](#how-it-works)
    - [Handling missing data (blank images)](#handling-missing-data-blank-images)
    - [Viewing/Exporting](#viewingexporting)
    - [File location](#file-location)

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

> **Note:** This section describes the workflow for exporting dense Sentinel-2 mosaics for all label points using the Earth Engine API and Google Cloud Storage. The pipeline is designed for time series sampling at 10-day intervals across each year, with robust handling for missing or cloudy images.

### Site and Time Window Definition

- Input Table: Loads `data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv`, which contains locations, observation dates, and a unique ID for each sample.

- Time Windows: For each site and year, generates ~36–37 consecutive 10-day intervals spanning the full year, aligned to the site’s observation date.

### Sentinel-2 Mosaic Retrieval

- For each time window:
  - Checks if the corresponding `.tif` file already exists in the Google Cloud Storage (GCS) bucket (matching the folder structure used locally). If so, it skips to the next.

  - If no valid Sentinel-2 image is found (e.g., due to clouds), a blank placeholder TIF is generated and stored (see below).

  - Otherwise, an Earth Engine export task generates a 1km x 1km Sentinel-2 surface reflectance mosaic (including the QA60 cloud mask band), which is then downloaded from GCS.

  - Each time window always produces a `.tif` file (either actual data or a blank placeholder) and a corresponding `.json` metadata file.

### Stacking and Output

- Loads all corresponding `.tif` files (real or blank) and stacks them into a single NumPy array of shape 
  `(n_time, n_bands, height, width)`.

- Saves A `.npy` file containing the full-year stacked mosaic sequence for the site. A comprehensive `.json` 
  metadata file with window definitions, bands, locations, and missing frame info. Both outputs are named with the unique site/sample ID (e.g., `site_-15.04_26.69_2023_1_stack.npy`).

![Mean RGB composite of all time steps for first sample site.](src/features/readme_figures/sentinel2_mosaic_example.png)

### Handling missing data (blank images)

- If no data is available for a window (e.g., persistent clouds), a blank placeholder image is copied into place using `generate_blank_tif.py`. 

- Metadata for missing data windows includes `"missing_data": true`.

### File location

- Input labels:
`data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv`

- Downloaded features & metadata:
`data/features/site_{lat}_{lon}_{year}_ID_stack.npy`
`data/features/site_{lat}_{lon}_{year}_ID_stack.json`

- Blank images:
`data/features/blank.tif`