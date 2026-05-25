# Smallholder Dry Season Irrigation Dataset
This is a test
## Overview
This repository defines and executes the sampling protocol for the smallholder dry season irrigation dataset in arid/semi-arid regions of Sub-Saharan Africa with a single rainy season. The goal is to ensure consistent, reproducible, and well-documented sampling that aligns with data download, labeling processes, and final dataset creation.

## Workflow
Generating this dataset requires five main steps. First, the area of interest and places to be sampled must be defined. Second, these regions are manually labeled for smallholder dry season irrigation presence using Earth Collect and Google Earth Pro, using any high resolution dry season imagery available at that location. Third, label quality is assessed through inter-rater comparison. Fourth, satellite data is downloaded from Google Earth Engine and aligned with the sampling locations. Finally, the data is processed and cleaned to create the final dataset.

1. **Sampling:**
   - Generate AOIs and sampling grids automatically.
   - Export to Collect surveys for manual annotation.

2. **Labeling:**
   - Use Collect survey tools to generate field labels.
   - Store labels with associated metadata.

3. **Quality Control:**
   - Compare labels across multiple labelers using `src/labels/label_comparison.py`.
   - Compute image-level detection metrics (precision, recall, F1).
   - Compute area overlap metrics (IoU, precision, recall).
   - Generate summary tables with weighted averages.
   - See `notebooks/labeler_comparison.ipynb` for interactive analysis.

4. **Feature Extraction:**
   - Download satellite data from Google Earth Engine (Sentinel-2) and Planet (PlanetScope).
   - Sentinel-2: 10m resolution, 10 spectral bands, free via GEE.
   - PlanetScope: 3m resolution, 4 bands (Blue, Green, Red, NIR), requires Planet license.
   - Both create time series stacks aligned with sampling locations.

5. **Data Processing:**
   - Clean and integrate labels with satellite features.
   - Prepare final datasets for analysis or model training.

## Repository Structure
```
.
├── config.yaml                # Project configuration file
├── CONTRIBUTING.md            # Contribution guidelines
├── LICENSE                    # License file
├── README.md                  # This file
├── CLAUDE.md                  # Developer guide for Claude Code
├── requirements.txt           # Python dependencies
├── notebooks/                 # Jupyter notebooks for data exploration and prototyping
│   └── labeler_comparison.ipynb  # Inter-rater comparison analysis
├── src/                       # Main source code folder
│   ├── processing/            # Scripts to clean, merge, and convert survey and polygon data
│   ├── sampling/              # Grid-based sampling code
│   ├── labels/                # Label generation, formatting, and quality control
│   │   ├── label_comparison.py      # LabelComparison class for inter-rater metrics
│   │   └── inter_rater_comparison.py # Helper functions for comparison
│   ├── features/              # Feature extraction and satellite data download
│   ├── modeling/              # ML model training and evaluation
│   └── utils/                 # Shared utility functions (e.g., figures, geometries)
```

## Getting Started

### Prerequisites
- Python 3.8 or later
- Google Earth Engine API access
- Collect survey tools (e.g., ODK Collect)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/smallholder-irrigation-dataset.git
   cd smallholder-irrigation-dataset
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv irr-venv
   source irr-venv/bin/activate
   pip install -r requirements.txt
   ```

   Alternatively, you can use `conda` to create a new environment and install the dependencies:
   ```bash
   conda create --name smh_irr_labels python=3.12
   conda activate smh_irr_labels
   pip install -r requirements.txt
   ```
3. Configure settings in `config.yaml`.

## Configuration
All project paths, sampling parameters, and GEE download settings are specified in `config.yaml` for easy management. 

### Data paths
Data is assumed to be stored locally, under data/ in the root repository. However, if it is stored elsewhere, this path can be specified as server_data_root in the configuration file, and if this directory can be found the data location will be updated accordingly (see utils).

## Contribution Guidelines
If you wish to contribute, please review `CONTRIBUTING.md` for details on our code of conduct, submission process, coding standards, and coding guidelines.
