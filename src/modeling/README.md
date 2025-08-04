## Overview

This folder contains a **machine learning pipeline for running experiments on multi-temporal Sentinel-2 satellite imagery** for irrigation classification. The pipeline is built for a specific dataset structure—each sample consists of a `.tif` image file, a corresponding `.tif` mask file, and a `.json` metadata file, with support for spatial-aware data splitting and 8-band irrigation label structure.

---

## Dataset Structure

- **Images:** Each `.tif` image contains 14 spectral bands × 37 time steps (total 518 bands per sample).
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
data/modeling/
├── 1_5168346_2023.09.06_image.tif
├── 1_5168346_2023.09.06_label.tif
├── 1_5168346_2023.09.06_image.json
└── ...
```

---

## Repository Structure

```
├── run_experiment.py          # Main experiment runner (config-driven)
├── final_test.py              # (WIP) Test a final/best model
├── experiment.yaml            # Config file for experiments
├── custom_dataset.py          # PyTorch Dataset for multi-temporal Sentinel-2 data
├── test_code.py               # (gitignored) Test script for validation
├── ml_pipeline/               # Core ML pipeline
│   ├── data_splitting.py      # Spatial-aware data splitting
│   ├── build_features.py      # Data flattening
│   ├── ml_model.py            # Model training
│   ├── evaluation.py          # Model metrics
│   └── visualization.py       # Visualize predictions
```

## Pipeline Components

### Core Components
- **custom_dataset.py:**  
  Loads `.tif` image/mask files, reshapes to (14, 37, H, W) for images, (8, H, W) for masks. Supports band selection and metadata extraction.

- **data_splitting.py:**  
  `IrrigationDataSplitter` class for spatial-aware data splitting with stratified sampling. Prevents spatial data leakage and maintains class balance. 

- **build_features.py:**  
  `get_datasets()` factory function creates train/val/test datasets. Supports different label band configurations.
  `flatten_dataset(dataset)` flattens all pixels and time/band features into a 2D table for ML 
  (one row per pixel).

- **ml_model.py:**  
  Model training (Random Forest, Gradient Boosting) and inference. Supports multi-label (multi-band) targets.

- **evaluation.py:**  
  Reports per-band accuracy and F1 scores for irrigation classification.

- **visualization.py:**  
  Plots model predictions and ground truth masks for selected samples.

### Experiment Runner
- **run_experiment.py:**  
  Main experiment runner that loads configuration from YAML and orchestrates the entire pipeline.

---

### Data Splitting

#### Folder Structure
```
splits/irrigation_binary_structure/
├── train/
│   ├── 1_5168346_2023.09.06_image.tif
│   ├── 1_5168346_2023.09.06_label.tif
│   ├── 1_5168346_2023.09.06_image.json
│   └── ...
├── val/
│   ├── 2_5168347_2023.09.06_image.tif
│   ├── 2_5168347_2023.09.06_label.tif
│   ├── 2_5168347_2023.09.06_image.json
│   └── ...
└── test/
    ├── 3_5168348_2023.09.06_image.tif
    ├── 3_5168348_2023.09.06_label.tif
    ├── 3_5168348_2023.09.06_image.json
    └── ...
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

- `-9999` in images and `-1` in masks are treated as invalid/masked pixels and are ignored.
- Georeferencing is not required for ML (Rasterio warnings are safe to ignore).

## Best Practices

- Commit code before running experiments.

---