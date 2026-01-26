# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository defines and executes a sampling protocol for creating a smallholder **dry season** irrigation dataset in arid/semi-arid regions of Sub-Saharan Africa (primarily Zambia). The workflow involves:
1. **Sampling**: Generate AOIs and sampling grids automatically
2. **Labeling**: Manual annotation using Earth Collect and Google Earth Pro
3. **Feature Extraction**: Download satellite data from Google Earth Engine aligned with sampling locations
4. **Data Processing**: Clean and integrate labels with satellite features for ML training

**Key Constraints**:
- **Temporal scope**: Dry season only (June-October, 5 months)
- **Primary labelers**: DSB, JL, KL, MV (four UCSB undergraduates)
- **Test set**: KL's labels exclusively (highest precision=0.75, recall=0.57 from QC analysis)

## Environment Setup

### Installation
```bash
# Clone and navigate to repository
git clone https://github.com/your-username/smallholder-irrigation-dataset.git
cd smallholder-irrigation-dataset

# Create virtual environment (choose one)
python -m venv irr-venv
source irr-venv/bin/activate  # On Mac/Linux

# OR use conda
conda create --name smh_irr_labels python=3.12
conda activate smh_irr_labels

# Install dependencies
pip install -r requirements.txt
```

### Configuration
All project paths and settings are in `config.yaml`. The utility function `get_data_root()` automatically detects whether code is running locally or on the cluster and returns the appropriate data root:
- Local: `data/` in repository root
- Cluster (UCSB GRIT ERI): `/home/waves/data/smallholder-irrigation-dataset/data/`

**Important**: The cluster data folder is NOT synchronized with the GitHub `data/` folder. Manual syncing is required.

## Running Tests

```bash
# Run all tests
python -m unittest discover tests

# Run specific test file
python -m unittest src/processing/tests/test_polygons_to_geojson.py
python -m unittest src/features/tests/test_create_label_band.py
```

## Key Workflows

### 1. Data Processing (Survey Labels)

The processing workflow converts Earth Collect `.zip` survey files and Google Earth Pro `.kml` polygon files into usable formats.

**File Naming Convention**: `<INITIALS>_<ID-RANGE>.<ext>` (e.g., `AB_1-25.zip`, `AB_1-25.kml`)
- For edited files: `<EDITOR_INITIALS>_<OriginalFileName>` (e.g., `JL_AB_1-25.zip`)
- For revised files: `<INITIALS>_v2_<ID-RANGE>.<ext>` (e.g., `AB_v2_1-25.kml`)

**Process a single pair of files**:
```bash
# Option 1: Run all steps individually
python src/processing/survey_to_csv.py data/labels/labeled_surveys/random_sample/raw/JL_26-50.zip
python src/processing/polygons_to_geojson.py data/labels/labeled_surveys/random_sample/raw/JL_26-50.kml
python src/processing/merge_survey_and_polygons.py data/labels/labeled_surveys/random_sample/processed/JL_26-50.csv

# Option 2: Use batch script (after chmod +x)
./src/processing/process_file_pair.sh JL_26-50
```

**Batch process all files**:
```bash
python src/processing/batch_process.py data/labels/labeled_surveys/random_sample/raw/
```

This also outputs `latest_irrigation_table.csv` and a GeoJSON with bounding boxes.

**Key Output Files**:
- `latest_irrigation_table.csv` - Image-level data with coverage statistics
- `latest_polygons_table.csv` - Polygon-level data with areas and categories

**Important Data Concepts**:
- **Irrigation certainty scale**: 1=no irrigation, 2-5=irrigation with increasing certainty
- **High-certainty coverage** (`percent_coverage_hc`): Includes polygons with certainty ≥ 3 (defined in `src/processing/merge_survey_and_polygons.py`)
- **Polygon categories**: `small-scale`, `industrial`, `tree_crop`, `lawn`, `covered`
- **Deduplication**: Some images were labeled by multiple labelers (e.g., survey 101-125 for QC). When aggregating, deduplicate by `image_id` (site_id + date), keeping labels with priority: KL > MV > DSB > JL

**Check for warnings in latest surveys**:
```bash
./src/processing/check_for_warnings.sh
./src/processing/remove_obsolete_surveys.sh  # Remove old survey versions
```

### 2. Feature Download (Google Earth Engine)

Download Sentinel-2 time series data for labeled sites. Each site gets **42 consecutive 10-day windows** (36 core + 3 buffer each side) starting from January 1st.

**Prerequisites**:
- Google Cloud Platform project with Earth Engine API enabled
- Service account key stored at `secrets/earthengine-key.json`

**Run download**:
```bash
python src/features/download_sentinel2.py
```

**Output Files** (per labeled image, saved to timestamped version folder):
- `{uid}_{site}_{YYYY.MM.DD}_stack.tif` – unmasked 42-window stack (10 bands × 42 windows = 420 bands)
- `{uid}_{site}_{YYYY.MM.DD}_stack_masked.tif` – cloud-masked stack (bad pixels = 0)
- `{uid}_{site}_{YYYY.MM.DD}_metadata.json` – per-window metadata (date_range, file_exists, masked_fraction)

**Create pixel-level labels** (9-band labels with per-labeler files):
```bash
# Run on latest feature download
python src/features/create_label_band.py --download_dir data/features --version 20260107_180813

# Or from Python
from src.features.create_label_band import create_labels
create_labels('data/features', '20260107_180813')
```

Output: `{uid}_{site}_{YYYY.MM.DD}_{operator}_labels.tif` with 9 bands:
- Band 1: Categorical irrigation type (1-5)
- Band 2: Binary irrigation mask
- Bands 3-7: Uncertainty flags
- Band 8: Certainty score (1-5)
- Band 9: Polygon coverage % (0-100, for mixed pixel analysis)

### 3. Label Quality Control (Inter-Rater Comparison)

Assess labeling consistency by comparing a ground truth labeler against other labelers. Located in `src/labels/` with the main notebook at `notebooks/labeler_comparison.ipynb`.

**Run the comparison**:
```python
from src.labels.label_comparison import LabelComparison

comparison = LabelComparison(
    irrigation_table_path='data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv',
    polygons_path='data/labels/labeled_surveys/random_sample/latest_polygons.geojson',
    image_boundaries_path='data/labels/labeled_surveys/random_sample/latest_irrigation_data.geojson',
    gt_operator='AB',                              # Ground truth labeler
    comparison_operators=['DSB', 'JL', 'KL', 'MV', 'PS'],  # Comparison labelers
    min_certainty=4,                               # Filter polygons by certainty
    date_tolerance_days=1,                         # Match images ±1 day
    output_dir='outputs/labeler_comparison'        # Save figures/CSVs here
)

# Generate all plots and metrics
for op in comparison.comparison_operators:
    comparison.plot_confusion_matrix(op)           # Image-level detection confusion matrix
    comparison.plot_detection_metrics_bar(op)      # Image-level precision/recall/F1 bar chart
    comparison.plot_area_metrics_bar(op)           # Area overlap precision/recall/IoU bar chart
    comparison.plot_area_histograms(op)            # Per-image metric distributions
    comparison.print_summary(op)                   # Print summary statistics

# Generate summary tables with weighted averages
detection_table, area_table = comparison.generate_summary_tables()
```

**Two Levels of Metrics**:

1. **Image-Level Detection**: Binary classification - did the labeler detect ANY irrigation?
   - TP: Both GT and comparison saw irrigation
   - FP: Only comparison saw irrigation
   - FN: Only GT saw irrigation
   - TN: Neither saw irrigation
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)

2. **Area Overlap**: How much do the labeled polygon areas agree?
   - For each image, union all GT polygons and all comparison polygons
   - Precision = intersection_area / comp_area (% of marked area that was correct)
   - Recall = intersection_area / gt_area (% of GT area that was found)
   - IoU = intersection_area / union_area
   - Overall metrics sum areas across all images before computing ratios

**Output Files** (saved to `output_dir`):
- `{op}_confusion_matrix.png` - Image detection confusion matrix
- `{op}_detection_metrics.png` - Image detection bar chart
- `{op}_area_metrics.png` - Area overlap bar chart
- `{op}_area_histograms.png` - Per-image metric distributions
- `{site_id}_{date}.png` - Side-by-side polygon comparison plots
- `image_detection_metrics.csv` - Summary table with weighted averages
- `area_overlap_metrics.csv` - Summary table with weighted averages

### 4. Dataset Analysis Notebooks

Located in `notebooks/`:

- **`results.ipynb`**: Dataset description and quality control results
  - QC comparison metrics (Anna Boser and Peter Siame as ground truth)
  - Summary statistics by labeler (locations, images, polygons)
  - Distribution histograms (images per location, polygons per image, polygon sizes, irrigation coverage)
  - Monthly breakdown (June-October only)
  - KL-only subsection for test set characterization

- **`splits.ipynb`**: Test set sizing analysis
  - Sample size requirements for statistical power
  - Monthly/yearly breakdown for KL's labels
  - Temporal change detection capacity (locations with both irrigated and non-irrigated images)
  - Comparison of KL-only vs all-labelers data

### 5. Machine Learning Pipeline

Run experiments on multi-temporal Sentinel-2 imagery for irrigation classification.

**Configure and run experiment**:
```bash
# Edit experiment.yaml to configure data, model, and parameters
python src/modeling/run_experiment.py
```

**Results** are saved in `./experiments/{experiment_name}/`:
- `model.pkl` – trained model
- `metrics.json` – evaluation metrics
- `visualization.png` – prediction plots
- `run.log` – execution log
- Copy of experiment config

**Dataset Structure**:
- **Images**: 10 spectral bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12) × 42 time steps = 420 bands per `*_stack.tif`
- **Labels**: 9-band `*_labels.tif` with irrigation labels (type, presence, uncertainty flags, certainty, coverage %)
- **Metadata**: `*_metadata.json` with location and per-window info (date_range, file_exists, masked_fraction)

## Architecture Notes

### Data Flow
1. **Sampling** (`src/sampling/`): Generate 1km grid points over agricultural lands using GFSAD Cropland Extent data → Output: GeoJSON/GeoPackage with sample locations
2. **Label Generation** (`src/labels/`): Use `surveys_with_locations.py` to create Earth Collect surveys from sampling locations → Manual labeling in Google Earth Pro/Earth Collect
3. **Processing** (`src/processing/`): Convert `.zip` surveys and `.kml` polygons to CSV/GeoJSON → Merge and validate → Pool into `latest_irrigation_table.csv`
4. **Quality Control** (`src/labels/label_comparison.py`): Compare labels across labelers → Compute inter-rater metrics → Generate summary tables and visualizations
5. **Feature Download** (`src/features/`): Read `latest_irrigation_table.csv` → Download Sentinel-2 time series from GEE → Apply cloud masking → Create 42-step stacks with 10 reflectance bands each
6. **Pixel Labeling** (`src/features/create_label_band.py`): Overlay labeled polygons on downloaded features → Create 9-band label `.tif` files per labeler (includes coverage % for mixed pixel analysis)
7. **Modeling** (`src/modeling/`): Spatial-aware data splitting → Flatten multi-temporal data → Train ML models (Random Forest, Gradient Boosting) → Evaluate and visualize

### Utility Functions (`src/utils/`)
- `utils.py`: Contains critical helper functions:
  - `find_project_root()`: Recursively finds the project root by locating `config.yaml`
  - `load_config()`: Loads project configuration
  - `get_data_root()`: Determines local vs. cluster data location
  - `save_data()`: Saves data with auto-generated `.json` metadata
- `geometries.py`: Geospatial utility functions
- `figures.py`: Plotting and visualization utilities

**Important**: Never hardcode file paths. Always use `get_data_root()` and path helpers.

### Multi-Temporal Sentinel-2 Data Structure
- **Time Series**: 42 windows × 10 days each (36 core + 3 buffer each side), starting from January 1st
- **Bands per window** (10 total):
  - 10 Sentinel-2 reflectance bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
  - Note: Vegetation indices (NDVI, EVI, NDWI) are NOT pre-computed; compute from raw bands if needed
- **Missing Data Handling**:
  - Cloud/bad pixels: Set to 0 in masked stack
  - Missing imagery: All-zero slice with `file_exists: false` in metadata
  - `masked_fraction`: Tracked per window in metadata JSON

### Cross-Validation for ML
The modeling pipeline uses file-list based organization (no file duplication). Each experiment can define its own CV structure via `cv_structure_name` in `experiment.yaml`.

**CV Folder Structure**:
```
data/modeling/splits/
└── {cv_structure_name}/
    ├── train/
    │   ├── fold_1/
    │   │   ├── train_files.txt
    │   │   └── val_files.txt
    │   └── fold_2/...
    ├── test/
    │   └── test_files.txt
    ├── manifest.csv
    └── cv_metadata.json
```

Spatial-aware splitting prevents data leakage by grouping at the site level.

## Git Workflow

- Work on feature branches named with initials (e.g., `jl-working`, `ab-dev`)
- Commit frequently with clear messages, especially when collaborating
- Include data changes in commits (labeling data is NOT in `.gitignore`)
- Open weekly pull requests with descriptions of changes
- Main branch: `main`

## Data Storage Best Practices

- Use `save_data()` utility function to auto-generate `.json` metadata
- Every data file should have either:
  - An associated `.json` metadata file, OR
  - A `.README` file in the same directory
- Delete unused files but always retain raw data for reproducibility
- Store data in `data/{module_name}/` corresponding to source module (e.g., `data/sampling/`, `data/labels/`, `data/features/`)
- Small, useful files can be added to GitHub; large files stay on cluster only

## Key Dependencies

- Python 3.8+ (recommended 3.12)
- Core: geopandas, rasterio, numpy, pandas, PyYAML
- ML: torch, scikit-learn, joblib
- Visualization: matplotlib, seaborn
- Other: geopy, tqdm, rapidfuzz
