# Feature Downloading

This module downloads satellite imagery time series for irrigation-labeled sites. Two data sources are supported:

1. **Sentinel-2** (via Google Earth Engine) - 10m resolution, 10 spectral bands, free
2. **PlanetScope** (via Planet Orders API) - 3m resolution, 4 bands, requires license

## Table of Contents
- [Feature Downloading](#feature-downloading)
  - [Table of Contents](#table-of-contents)
  - [Sentinel-2 (Google Earth Engine)](#sentinel-2-google-earth-engine)
    - [Overview](#overview)
    - [Prerequisites](#prerequisites)
    - [Google Cloud Setup](#google-cloud-setup)
    - [Service account and GCS configuration](#service-account-and-gcs-configuration)
    - [Downloading Features](#downloading-features)
    - [Data Quality Assessment and Visualization](#data-quality-assessment-and-visualization)
    - [Stacking and Output](#stacking-and-output)
    - [Running the Download](#running-the-download)
    - [Dataset Location](#dataset-location)
  - [PlanetScope (Planet Orders API)](#planetscope-planet-orders-api)
    - [Overview](#overview-1)
    - [Prerequisites](#prerequisites-1)
    - [Comparison: Sentinel-2 vs PlanetScope](#comparison-sentinel-2-vs-planetscope)
    - [How the Download Works](#how-the-download-works)
    - [Running the Download](#running-the-download-1)
    - [Key Parameters](#key-parameters)
    - [Output Files](#output-files)
    - [Monitoring Progress](#monitoring-progress)
    - [Troubleshooting](#troubleshooting)
  - [Creating Pixel-Level Labels](#creating-pixel-level-labels)

---

## Sentinel-2 (Google Earth Engine)

### Overview

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

The downloaded time series data can be visualized using the built-in visualization function in `download_sentinel2.py`:

```python
from src.features.download_sentinel2 import visualize_time_series_stack

visualize_time_series_stack(out_dir='path/to/features/version', file_id='10_5130509_2016.09.09')
```

This displays RGB composites (B4=Red, B3=Green, B2=Blue) for both unmasked and masked stacks across the first 15 timesteps.

Additional visualization tools are available in:
- `visualize_tif.py` - General TIF visualization
- `sentinel2_visualization.py` - Detailed analysis with mask overlays
- `image_label_visualization.py` - Overlay irrigation labels on imagery

### Stacking and Output

The per-window GeoTIFFs are downloaded to a temporary directory, then stacked into final outputs:

- **Per-window layout**: 10 reflectance bands
- **Stack shape**: (T, B, H, W) = (42, 10, 100, 100) → saved as (B, T, H, W) flattened to **(420, 100, 100)** in the final GeoTIFF

**Files Stored**: For each labeled image, we save three files:

- `{uid}_{site}_{YYYY.MM.DD}_stack.tif` – Unmasked stack (all pixels)
- `{uid}_{site}_{YYYY.MM.DD}_stack_masked.tif` – Masked stack (cloud pixels = 0)
- `{uid}_{site}_{YYYY.MM.DD}_metadata.json` – Per-window metadata (date ranges, file_exists, masked_fraction)

**Reading the stacks**:
```python
import rasterio
import numpy as np

with rasterio.open('path/to/stack.tif') as src:
    data = src.read()  # Shape: (420, 100, 100)

# Reshape to (T, B, H, W)
num_bands = 10
num_windows = 42
data = data.reshape(num_bands, num_windows, 100, 100).transpose(1, 0, 2, 3)
# Now shape is (42, 10, 100, 100)
```

### Running the Download

To run the download:

```python
from src.features.download_sentinel2 import dataset_download

dataset_download(
    csv='data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv',
    download_dir='data/features',
    collection='L1C',      # or 'L2A' for surface reflectance (2018+)
    start_month=1,         # Start from January
    num_windows=36,        # 36 core windows
    timestep=10,           # 10 days per window
    window_buffer=3,       # 3 extra windows before/after
    target_size=100,       # 100x100 pixels (1km x 1km at 10m)
    subset=False           # Set True to test with 10 rows
)
```

Or run directly:
```shell
python src/features/download_sentinel2.py
```

Downloads are saved to a timestamped version folder (e.g., `data/features/sentinel2/20260107_180813/`).

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

---

## PlanetScope (Planet Orders API)

### Overview

PlanetScope provides higher resolution (3m) imagery than Sentinel-2, which can be valuable for detecting small-scale irrigation features. The download uses Planet's asynchronous Orders API with parallel processing for efficiency.

**Key characteristics:**
- **Resolution**: 3m (vs Sentinel-2's 10m)
- **Bands**: 4 (Blue, Green, Red, NIR)
- **Time series**: Same 42-window structure as Sentinel-2
- **Processing**: Surface Reflectance (SR) or Top of Atmosphere (TOA)
- **Coregistration**: All scenes aligned to best anchor for sub-pixel accuracy

### Prerequisites

1. **Planet Account**: Requires an active Planet license with Data API access
2. **API Key**: Store your Planet API key in `secrets/planet-api-key.txt` (single line, no trailing newline)
3. **Python Package**: Install the Planet SDK: `pip install planet`

To get your API key:
1. Log into [Planet Explorer](https://www.planet.com/explorer/)
2. Go to Account Settings → API Key
3. Copy the key and save to `secrets/planet-api-key.txt`

### Comparison: Sentinel-2 vs PlanetScope

| Feature | Sentinel-2 | PlanetScope |
|---------|------------|-------------|
| Resolution | 10m | 3m |
| Spectral bands | 10 | 4 (BGRNIR) |
| Stack size | 420 bands (10×42) | 168 bands (4×42) |
| Grid size | 100×100 pixels (~1km²) | 334×334 pixels (~1km²) |
| Cloud masking | SCL/QA60 | UDM2 |
| API | Google Earth Engine (sync) | Planet Orders API (async) |
| Cost | Free | Requires license |
| Availability | Global, 5-day revisit | Global, daily revisit |

### How the Download Works

The PlanetScope download uses an asynchronous parallel pipeline optimized for throughput:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PARALLEL DOWNLOAD PIPELINE                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Site 1     │    │   Site 2     │    │   Site 3     │   ...    │
│  │ Scene Search │    │ Scene Search │    │ Scene Search │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                   │
│         └─────────┬─────────┴─────────┬─────────┘                   │
│                   ▼                   ▼                             │
│         ┌─────────────────────────────────────┐                     │
│         │     Submit Orders to Planet API      │                    │
│         │   (up to max_concurrent_orders)      │                    │
│         └─────────────────┬───────────────────┘                     │
│                           │                                         │
│                           ▼                                         │
│         ┌─────────────────────────────────────┐                     │
│         │   Quick Poll: Check for completed    │◄──────┐            │
│         │   orders after each search batch     │       │            │
│         └─────────────────┬───────────────────┘       │            │
│                           │                           │            │
│              ┌────────────┴────────────┐              │            │
│              ▼                         ▼              │            │
│     ┌────────────────┐        ┌────────────────┐     │            │
│     │ Order Ready?   │        │ Still Pending  │─────┘            │
│     │ Download &     │        │ Continue...    │                   │
│     │ Stack          │        └────────────────┘                   │
│     └────────────────┘                                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Step-by-step process:**

1. **Scene Search** (parallel):
   - For each site, search Planet's catalog for scenes in each 10-day window
   - `concurrent_scene_searches` sites are searched in parallel (default: 10)
   - Each scene is scored by `effective_coverage = footprint_coverage × clear_percent`
   - The best scene per window is selected

2. **Order Submission**:
   - A batch order is submitted to Planet containing all 42 scenes for one site
   - Orders include processing tools: clip to AOI, reproject to UTM 3m, coregister
   - Up to `max_concurrent_orders` can be pending at Planet simultaneously

3. **Quick Poll** (after each search batch):
   - Check all pending orders for completion
   - Download and stack any completed orders immediately
   - This keeps the pipeline flowing without waiting

4. **Download & Stack**:
   - Downloaded scenes are aligned to a common grid centered on the site
   - Scenes are coregistered to the clearest anchor scene
   - Final stacks are saved as GeoTIFFs

5. **Resume Capability**:
   - If interrupted, use `resume_dir` to continue where you left off
   - Sites with existing `*_stack.tif` files are skipped

### Running the Download

**Recommended: Parallel mode with quick polling**

```python
from src.features.download_planetscope import dataset_download_parallel

results = dataset_download_parallel(
    csv='data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv',
    download_dir='data/features/planetscope',
    max_concurrent_orders=100,      # Orders pending at Planet at once
    concurrent_scene_searches=10,   # Sites to search in parallel
    product_type='SR',              # 'SR' (Surface Reflectance) or 'TOA'
    max_cloud_cover=1.0,            # 0-1, use 1.0 for maximum coverage
    start_month=1,                  # Start from January
    num_windows=36,                 # 36 core windows
    timestep=10,                    # 10 days per window
    window_buffer=3,                # 3 extra windows before/after (total: 42)
)
```

**Command line with caffeinate (macOS)**:

```bash
caffeinate -i -s python3 -c "
from src.features.download_planetscope import dataset_download_parallel

results = dataset_download_parallel(
    csv='data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv',
    download_dir='data/features/planetscope',
    max_concurrent_orders=100,
    concurrent_scene_searches=10,
    product_type='SR',
    max_cloud_cover=1.0
)
print(f'Completed: {sum(1 for v in results.values() if v == \"success\")} successful')
"
```

**Resume an interrupted download**:

```python
results = dataset_download_parallel(
    csv='data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv',
    download_dir='data/features/planetscope',
    resume_dir='20260127_161535_SR',  # Existing folder to resume into
    max_concurrent_orders=100,
    concurrent_scene_searches=10,
    product_type='SR',
    max_cloud_cover=1.0
)
```

**Sequential mode** (slower, but simpler for debugging):

```python
from src.features.download_planetscope import dataset_download

dataset_download(
    csv='data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv',
    download_dir='data/features/planetscope',
    product_type='SR',
    max_cloud_cover=0.5,
    subset=True  # Only process first 10 rows for testing
)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_concurrent_orders` | 100 | Maximum orders pending at Planet at once. Higher = faster but may hit quotas. |
| `concurrent_scene_searches` | 10 | Sites to search in parallel. Higher = faster scene searching but more API calls. |
| `product_type` | 'SR' | 'SR' (Surface Reflectance, atmospherically corrected) or 'TOA' (Top of Atmosphere, more scenes available) |
| `max_cloud_cover` | 0.5 | Maximum cloud cover fraction (0-1). Use 1.0 to maximize scene availability; clouds are masked in the stack. |
| `start_month` | 1 | Month to start the time series (1=January) |
| `num_windows` | 36 | Number of core 10-day windows |
| `window_buffer` | 3 | Extra windows before and after (total windows = num_windows + 2×buffer) |
| `resume_dir` | None | Existing version folder to resume into. Sites with existing stacks are skipped. |

### Output Files

For each labeled site, the following files are created in the version folder:

```
data/features/planetscope/20260127_161535_SR/
├── 1_5130509_2016.09.09_stack.tif        # Unmasked stack (168 bands)
├── 1_5130509_2016.09.09_stack_masked.tif # Cloud-masked stack (bad pixels = 0)
├── 1_5130509_2016.09.09_metadata.json    # Per-window metadata
├── 2_5130509_2017.07.15_stack.tif
├── ...
├── download_results.json                  # Summary of all download results
└── metadata_20260127_161535_SR.json       # Run configuration
```

**Stack structure**:
- Shape: `(168, 334, 334)` = 4 bands × 42 windows × 334×334 pixels
- Bands are interleaved: [B_t0, G_t0, R_t0, NIR_t0, B_t1, G_t1, ...]
- Data type: uint16 (surface reflectance scaled by 10,000)
- NoData value: 0

**Reading the stacks**:
```python
import rasterio
import numpy as np

with rasterio.open('path/to/stack.tif') as src:
    data = src.read()  # Shape: (168, 334, 334)

# Reshape to (T, B, H, W)
num_bands = 4
num_windows = 42
data = data.reshape(num_bands, num_windows, 334, 334).transpose(1, 0, 2, 3)
# Now shape is (42, 4, 334, 334)
```

**Metadata JSON structure**:
```json
{
  "file_id": "1_5130509_2016.09.09",
  "lat": -15.4567,
  "lon": 28.1234,
  "product_type": "SR",
  "num_windows": 42,
  "windows": [
    {
      "window_index": 0,
      "date_range": ["2015-12-12", "2015-12-22"],
      "item_id": "20151215_073012_0c43",
      "cloud_cover": 0.02,
      "effective_coverage": 98.5
    },
    ...
  ]
}
```

### Monitoring Progress

**Watch the log in real-time**:
```bash
tail -f data/features/planetscope/download_*.log | grep -E "(Submitted|Order complete|Quick poll|downloaded)"
```

**Check progress summary**:
```bash
# Count orders and completions
echo "Orders submitted:" && grep -c "Submitted order" data/features/planetscope/*.log
echo "Orders completed:" && grep -c "Order complete" data/features/planetscope/*.log
echo "Total stacks:" && ls data/features/planetscope/*_SR/*_stack.tif 2>/dev/null | wc -l
```

**Progress messages to look for**:
- `Searching scenes for X sites in parallel...` - Scene search batch starting
- `Submitted order <uuid> with N scenes` - Order sent to Planet
- `Quick poll: checking N pending orders...` - Checking for completed orders
- `Order complete, downloading...` - Downloading a ready order
- `Quick poll: downloaded N stacks, M still pending` - Batch download summary

### Troubleshooting

**Rate Limits (429 errors)**:
```
Retrying: caught <class 'planet.exceptions.TooManyRequests'>: max rate reached: retry-in 200ms
```
This is normal. The Planet SDK handles rate limits automatically with exponential backoff. The download will continue after a brief pause.

**No scenes found for window**:
```
WARNING - Window 5: no scenes found for 2016-02-01 to 2016-02-11
```
Some time windows may have no available imagery. These windows will be all-zeros in the stack with `item_id: null` in metadata.

**Coverage check failures**:
```
WARNING - Coverage check failed for <scene_id>: failed to get thumbnail preview
```
Occasional coverage check failures are normal. The download continues and falls back to scene-level cloud statistics.

**Order failed/partial**:
```
ERROR - <file_id>: Order failed
```
Planet may reject orders if scenes are unavailable. The site will be marked as failed in `download_results.json`. Consider retrying with `product_type='TOA'` for better scene availability.

**Resume after interruption**:
If the download is interrupted, restart with the `resume_dir` parameter pointing to the existing version folder. Sites with existing `*_stack.tif` files will be skipped.

---

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
# Sentinel-2 (default)
python src/features/create_label_band.py --download_dir data/features --version 20260107_180813

# PlanetScope
python src/features/create_label_band.py --sensor planetscope --version 20260127_161535_SR

# Or use defaults (latest version, Sentinel-2)
python src/features/create_label_band.py
```

**From Python:**
```python
from src.features.create_label_band import create_labels

# Sentinel-2
create_labels('data/features', '20260107_180813', sensor='sentinel2')

# PlanetScope
create_labels('data/features/planetscope', '20260127_161535_SR', sensor='planetscope')
```

### Current Dataset

Labels for the `20260107_180813` feature download are stored alongside the stack files:
- **Total label files**: 3,536
- **With irrigation polygons**: 1,244
- **Empty labels (no irrigation)**: 2,292

---

## Visualization Tools

The visualization module (`src/features/visualization/`) provides publication-quality figures for satellite data, labels, and time series. **All functions support both Sentinel-2 and PlanetScope** via a unified `sensor` parameter.

### Satellite Visualization (Both Sensors)

`satellite_visualization.py` provides functions to visualize RGB imagery with irrigation mask overlays. Use `sensor='sentinel2'` (default) or `sensor='planetscope'`.

```python
from src.features.visualization.satellite_visualization import (
    SENSOR_CONFIG,           # Sensor configuration dictionary
    get_features_dir,        # Get data directory
    find_stack_for_site,     # Find stack for site/date
    find_labels_for_stack,   # Find label files
    load_rgb_from_stack,     # Load RGB image
    load_label_mask,         # Load label band
    plot_satellite_with_mask # Plot RGB with mask overlay
)

# Plot Sentinel-2 with irrigation mask
plot_satellite_with_mask(stack_path, label_path, sensor='sentinel2')

# Plot PlanetScope with irrigation mask
plot_satellite_with_mask(stack_path, label_path, sensor='planetscope')

# Get data directory for each sensor
s2_dir = get_features_dir(sensor='sentinel2')   # data/features/sentinel2/20260107_180813
ps_dir = get_features_dir(sensor='planetscope') # data/features/planetscope/20260127_161535_SR
```

### EVI Time Series Visualization

`evi_timeseries_visualization.py` extracts and plots EVI time series showing irrigation patterns.

```python
from src.features.visualization.evi_timeseries_visualization import (
    extract_evi_timeseries,   # Extract EVI for all pixels
    plot_evi_timeseries,      # Plot irrigated vs non-irrigated
    plot_clustered_timeseries # Auto-cluster temporal patterns
)

# Sentinel-2 EVI time series
plot_evi_timeseries(stack_path, label_path, sensor='sentinel2')

# PlanetScope EVI time series
plot_evi_timeseries(stack_path, label_path, sensor='planetscope')

# Cluster-based exploration (no labels needed)
plot_clustered_timeseries(stack_path, sensor='sentinel2', n_clusters=4)
```

### Combined Multi-Source Visualization

`combined_visualization.py` orchestrates comparisons across data sources.

```python
from src.features.visualization.combined_visualization import (
    find_all_available_sources,   # Check data availability
    plot_combined_comparison,     # Full multi-panel figure
    plot_sensor_comparison,       # Side-by-side S2 vs PS
    find_sites_with_both_sensors  # Sites with both sensors
)

# Find sites with both Sentinel-2 and PlanetScope
common_sites = find_sites_with_both_sensors(limit=10)

# Create combined comparison figure
fig = plot_combined_comparison(
    site_id='id_5119273',
    year=2021, month=9, day=16,
    show_evi=True  # Include EVI panels
)

# Side-by-side sensor comparison only
fig = plot_sensor_comparison(
    site_id='id_5119273',
    year=2021, month=9, day=16
)
```

### GEE Screenshot Visualization

`gee_screenshot_visualization.py` overlays labeled polygons on Google Earth Pro screenshots.

```python
from src.features.visualization.gee_screenshot_visualization import (
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

### Sensor Configuration

Both sensors share a unified configuration in `SENSOR_CONFIG`:

```python
SENSOR_CONFIG = {
    'sentinel2': {
        'n_bands': 10,
        'band_indices': {'blue': 0, 'green': 1, 'red': 2, 'nir': 6, ...},
        'default_version': '20260107_180813',
        'data_dir': 'features',
        'normalization': 3000.0,
    },
    'planetscope': {
        'n_bands': 4,
        'band_indices': {'blue': 0, 'green': 1, 'red': 2, 'nir': 3},
        'default_version': '20260127_161535_SR',
        'data_dir': 'features/planetscope',
        'normalization': 3000.0,
    }
}
```

### Visualization Notebook

See `src/features/visualization/feature_visualization.ipynb` for interactive examples of all visualization functions.
