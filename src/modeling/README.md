## Overview

This folder contains a **machine learning pipeline for running experiments on multi-temporal Sentinel-2 satellite imagery** for irrigation classification. The pipeline is built for a specific dataset structure—each sample consists of a `.tif` image file, a corresponding `.tif` mask file, and `.json` metadata files, with support for spatial-aware data splitting and 8-band irrigation label structure.

---

## Dataset Structure

- **Images:** Each `.tif` image contains 14 spectral bands × 37 time steps (total 518 bands per sample). Can be adjusted as needed.
- **Masks:** Each `.tif` mask contains 8 bands representing different irrigation classification targets.
- **Metadata:** Each `.json` file contains location and temporal information.
- **File Naming:** Consistent convention `{unique_id}_{site_id}_{date}_{type}.tif`

### 8-Band Irrigation Label Structure
- **Band 1**: Per-pixel irrigation type classification (0-5)
- **Band 2**: Per-pixel irrigation presence (0-1) 
- **Bands 3-7**: Binary uncertainty explanation masks
- **Band 8**: Irrigation certainty score (0-4)

Example layout:
```
training_data/
├── 1_5168346_2023.09.06_image.tif
├── 1_5168346_2023.09.06_label.tif
├── 1_5168346_2023.09.06_image.json
└── 1_5168346_2023.09.06_label.json
```

---

## Repository Structure

```
├── run_experiment.py          # Main experiment runner (config-driven)
├── experiment.yaml            # Config file for experiments
├── custom_dataset.py          # PyTorch Dataset for multi-temporal Sentinel-2 data
├── ml_pipeline/               # Core ML pipeline
│   ├── data_splitting.py      # Spatial-aware data splitting with cross validation
│   ├── build_features.py      # Data flattening
│   ├── ml_model.py            # Model training
│   ├── evaluation.py          # Model metrics
│   └── visualization.py       # Visualize predictions
```

## Pipeline Components

### Core Components
- **custom_dataset.py:**  
  Loads `.tif` image/mask files, reshapes to (14, 37, H, W) for images, (8, H, W) for masks. Supports band and time selection and metadata extraction.

- **data_splitting.py:**  
  `IrrigationDataSplitter` class for spatial-aware data splitting with stratified sampling. Prevents spatial data leakage by grouping at the site level. Supports both one-shot train/val/test splits and K-fold cross-validation with held-out test sets.
 
  
- **build_features.py:**  
  `flatten_dataset()` function flattens multi-temporal data into 2D tables for ML (one row per pixel). Includes temporal interpolation imputation for handling missing values. Supports both per-pixel and per-band-time flattening modes.

- **ml_model.py:**  
  Model training (Random Forest, Gradient Boosting) and inference. Supports multi-label (multi-band) targets.

- **evaluation.py:**  
This file provides evaluation utilities for smallholder irrigation models, including pixel-level metrics, image-level summaries, and detailed breakdowns by metadata factors. It also supports visualizing feature importances across bands and timesteps.

- **visualization.py:**  
Plots model predictions and ground truth masks for selected samples.

### Experiment Runner
- **run_experiment.py:**  
  Main experiment runner that loads configuration from YAML and orchestrates the entire pipeline. Supports both single train/val experiments and K-fold cross-validation. Handles temporary file staging for efficient data loading.

## Data Splitting

The pipeline supports cross-validation experiments with file-list based organization (no file duplication). Each experiment can have its own CV structure.

### CV Folder Structure
```
data/modeling/splits/
└── irrigation_cv/                    # ← cv_structure_name
    ├── train/
    │   ├── fold_1/
    │   │   ├── train_files.txt       # ← File list for training
    │   │   └── val_files.txt         # ← File list for validation
    │   ├── fold_2/
    │   │   ├── train_files.txt
    │   │   └── val_files.txt
    │   └── ...
    ├── test/
    │   └── test_files.txt            # ← Held-out test set
    ├── manifest.csv                  # ← File metadata
    └── cv_metadata.json              # ← CV metadata
```

### Different Experiment Configurations

For different experiments (e.g., different bands), use different `cv_structure_name`:

```yaml
# Experiment 1: Band 2 (irrigation presence)
name: "rf_band2_experiment"
data:
  cv_structure_name: "band2_cv"
  label_bands: [2]

# Experiment 2: Band 1 (irrigation type)  
name: "rf_band1_experiment"
data:
  cv_structure_name: "band1_cv"
  label_bands: [1]
```

This creates separate CV structures:
```
data/modeling/splits/
├── band2_cv/              # For irrigation presence experiments
└── band1_cv/              # For irrigation type experiments
```

---

## Experiment Workflow

1. **Configure experiment in `experiment.yaml`.**
2. **Run**  
   ```
   python run_experiment.py
   ```
3. **Results**  
   For each experiment, a new subfolder in `./experiments/` contains:
   - Copy of the experiment config
   - Trained model (`model.pkl`)
   - Evaluation metrics (`metrics.json`)
   - Plots (`visualization.png`)
   - Log file (`run.log`)

---

## Usage Notes

- `-9999` in images is treated as invalid/masked pixels and are ignored.
- Georeferencing is not required for ML (Rasterio warnings are safe to ignore).

## Best Practices

- Commit code before running experiments.
- Use file lists for cross-validation to save disk space.
- Choose appropriate bands based on your classification task.
- Validate data splits visually using the built-in visualization tools.