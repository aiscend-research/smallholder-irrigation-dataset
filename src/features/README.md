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
    - [Label Process Overview](#label-process-overview)

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


## Creating Pixel-Level Labels

For each Sentinel-2 image, we need to classify each pixel within the image as irrigated or not irrigated, and if it is irrigated, specify the type of irrigation and if we are uncertain as to whether or not it is irrigation, we specify uncertainty, which is one of five categories. To do this, we take the labeled polygons corresponding to the image and generate a `.tif` files of six bands. 

<img src="readme_figures/band_table.png" alt="table showing band information" width="500"/>

The first band specifies the type of irrigation, if any, and the next five bands indicate the uncertainty of the irrigation classification, with each band corresponding to a different uncertainty explanation.

To run this script, navigate to the `src` directory and run

```{bash}
python3 features/create_label_band.py
```

This will create a folder `~/data/dataset/labels` with all corresponding labels.

### Label Process Overview

After downloading the Sentinel-2 images, we then create labels for each pixel. To do this, we first iterate through all the `.tif` mosaic files, which are assumed to be located at `data/dataset/images`.

For each file, we extract the file data using (latitude, longitude, offset, start date, and end date) from the filename to retrieve the survey date. Then, we extract the `.tif` metadata, such that we can create labels at the resolution of the original `.tif`.

Then, we must link the (latitude, longitude, survey date) to its corresponding labelled polygons (`.geojson` file). 

Then, we retrieve the polygons from the corresponding `.geojson` file, only retrieving polygons greater than a specified certainty (default is 4+). We store these polygons in a `geopandas.geodataframe.GeoDataFrame`, and rasterize these polygons at the same resolution of the original image, as a binary mask – a given pixel is 1 if the polygon overlaps with it, 0 otherwise. Note that if a polygon only partially overlaps with a pixel, it will count as 1 only if the it overlaps with the center of the pixel.

Then, we save the binary mask into a new file, located in `data/dataset/labels`, with the filename the same as the original image.