## Overview

This folder contains a **machine learning pipeline for running experiments on multi-temporal satellite imagery** for crop classification. The pipeline is built for a specific dataset structure—each sample consists of a `.tif` image file and a corresponding `.tif` mask file, with directories for training and validation splits.

---

## Dataset Structure

- **Images:** Each `.tif` image contains 14 spectral bands × 37 time steps (total 518 bands per sample).
- **Masks:** Each `.tif` mask contains 8 bands, each representing a different target/class label.
- **Folders:** 
  - `training/` — contains paired image and mask files for training
  - `validation/` — contains paired image and mask files for validation

Example layout:
```
example-data/
├── training/
│   ├── image_001.tif
│   ├── mask_001.tif
│   └── ...
├── validation/
│   ├── image_001.tif
│   ├── mask_001.tif
│   └── ...
```

---

## Repository Structure

```
├── run_experiment.py          # Main experiment runner (config-driven)
├── final_test.py              # (WIP) Test a final/best model
├── experiments.yaml           # Config file for experiments
├── custom_dataset.py          # PyTorch Dataset for image/mask pairs
├── custom_datamodule.py       # (optional) Data module for advanced workflows(Deep Learning)
├── ml_pipeline/               # Core ML pipeline
│   ├── build_features.py      # Data flattening
│   ├── ml_model.py            # Model training
│   ├── evaluation.py          # Model metrics
│   └── visualization.py       # Visualize predictions
```

---

## Pipeline Components

- **custom_dataset.py:**  
  Loads `.tif` image/mask files, reshapes to (14, 37, H, W) for images, (8, H, W) for masks. Allows band selection by name.
- **build_features.py:**  
  `flatten_dataset(dataset)` flattens all pixels and time/band features into a 2D table for ML (one row per pixel).
- **ml_model.py:**  
  Model training (Random Forest, Gradient Boosting), supports multi-label (multi-band) targets.
- **evaluation.py:**  
  Reports per-band accuracy and F1.
- **visualization.py:**  
  Plots model predictions and ground truth masks for selected samples.

---

## Experiment Workflow

1. **Configure experiment in `experiments.yaml`.**
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