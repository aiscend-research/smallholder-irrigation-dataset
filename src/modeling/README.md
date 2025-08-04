<<<<<<< Updated upstream
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

=======
# Irrigation Classification Modeling Pipeline

##Overview

This folder contains a comprehensive machine learning pipeline for irrigation classification using multi-temporal Sentinel-2 satellite imagery. The pipeline supports both terratorch datasets and custom irrigation datasets with advanced spatial-aware data splitting.

## Project Structure

```
src/modeling/
├── README.md                    # This comprehensive guide
├── run_experiment.py            # Main experiment runner
├── ml_pipeline/
├── custom_dataset.py            # Custom dataset for irrigation data
├── custom_datamodule.py         # Custom datamodule for PyTorch Lightning
├── experiment.yaml              # Default experiment configuration
├── experiment_custom.yaml       # Custom datamodule configuration example
├── test_code.py                 # Full functionality tests (requires dependencies)
├── ml_pipeline/                 # Core ML pipeline modules
│   ├── __init__.py
│   ├── build_features.py        # Data loading and datamodule factory
│   ├── data_splitting.py        # Spatial-aware data splitting
│   ├── integrate_splits.py      # Advanced data splitting and config generation
│   ├── ml_model.py              # Model training (Random Forest, Gradient Boosting)
│   ├── evaluation.py            # Model evaluation and metrics
│   └── visualization.py         # Plotting and visualization
└── __pycache__/                 # Python cache files
```

## Key Features

### Dual Datamodule Support
- **Terratorch Datamodule**: For standard multi-temporal crop classification datasets
- **Custom Datamodule**: For smallholder irrigation data with 8-band label structure

### Advanced Data Splitting
- **Spatial-aware splitting**: Prevents data leakage by splitting by location
- **Stratified sampling**: Maintains class balance across splits
- **8-band label support**: Handles complex irrigation label structure
- **Cross-validation**: K-fold cross-validation with spatial awareness

### Comprehensive Experimentation
- **Configuration-driven**: YAML-based experiment configuration
- **Multiple model types**: Random Forest, Gradient Boosting
- **Band-specific experiments**: Test different label bands (1-8)
- **Uncertainty-aware**: Experiments with uncertainty masks and certainty scores

## Quick Start

### 1. Install Dependencies
```bash
# Install required packages
pip install pytorch-lightning gdown torch torchvision rasterio scikit-learn pandas numpy matplotlib seaborn albumentations terratorch

# Or use the installation script (if available)
python install_dependencies.py
```

### 2. Test Your Setup
```bash
# Test full functionality (requires all dependencies)
python test_code.py
```

### 3. Run Experiments

#### Using Terratorch Datamodule (Default)
```bash
python run_experiment.py
```

#### Using Custom Datamodule
```bash
python run_experiment.py experiment_custom.yaml
```

#### Using Advanced Data Splitting
```bash
# Generate experiment configurations with spatial-aware splits
python ml_pipeline/integrate_splits.py

# Run specific experiments
python run_experiment.py experiment_configs/experiment_band_2.yaml
```

## Data Formats

### Terratorch Format
Standard multi-temporal crop classification dataset format used by the terratorch library.

### Custom Irrigation Format
Your custom dataset should have the following structure:
```
dataset_path/
├── site_12.34_56.78_2023_123.tif    # Sentinel-2 time series (518 bands)
├── site_12.34_56.78_2023_123.json   # Metadata file
├── site_12.35_56.79_2023_124.tif
├── site_12.35_56.79_2023_124.json
└── ...
```

**TIF files**: Sentinel-2 multi-temporal data with shape `(518, H, W)` where:
- 518 = 14 spectral bands × 37 timesteps (stored as T*B, H, W)
- 14 bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12, NDVI, EVI, NDWI, SCL
- H, W = 100x100 pixels

**JSON files**: Metadata containing site information, coordinates, band names, shape, etc.

### 8-Band Irrigation Label Structure
Your irrigation labels have 8 bands with the following structure:
- **Band 1**: Per-pixel irrigation type classification (0-5: no irrigation, small-scale, tree crop, industrial, lawn, covered)
- **Band 2**: Per-pixel irrigation presence (0-1: binary classification)
- **Bands 3-7**: Binary uncertainty explanation masks (unclear agriculture, slightly green, uneven, naturally green, fishpond)
- **Band 8**: Irrigation certainty score (0-4: no irrigation to probably irrigated)

## Configuration

### Basic Configuration (`experiment.yaml`)
```yaml
name: 'irrigation_classification'
data:
  dataset_path: "path/to/your/dataset"
  datamodule:
    type: "terratorch"  # or "custom"
    batch_size: 8
    num_workers: 2
model:
  type: "random_forest"
  hyperparameters:
    random_forest:
      n_estimators: 100
      random_state: 42
```

### Custom Datamodule Configuration (`experiment_custom.yaml`)
```yaml
data:
  dataset_path: "path/to/your/custom/dataset"
  datamodule:
    type: "custom"
    batch_size: 4
    num_workers: 0
    custom_params:
      train_files: ["site_12.34_56.78_2023_123", "site_12.35_56.79_2023_124"]
      val_files: ["site_12.36_56.80_2023_125"]
      test_files: ["site_12.37_56.81_2023_126"]
      label_bands: [2]  # Use irrigation presence (band 2)
```

## Advanced Data Splitting

### Spatial-Aware Splitting
The `ml_pipeline/integrate_splits.py` script provides advanced data splitting capabilities:

```bash
python integrate_splits.py
```

This generates:
- **Spatial stratified splits**: By location to prevent data leakage
- **Cross-validation configurations**: K-fold CV with spatial awareness
- **Band-specific experiments**: Different label bands (1, 2, 8)
- **Uncertainty-aware experiments**: Using uncertainty masks and certainty scores

### Generated Configurations
```
experiment_configs/
├── experiment_main.yaml         # Main experiment with band 2
├── experiment_band_1.yaml       # Multi-class irrigation types
├── experiment_band_8.yaml       # Certainty score classification
cv_configs/                      # Cross-validation configurations
├── experiment_cv_fold_1.yaml
├── experiment_cv_fold_2.yaml
└── ...
uncertainty_configs/             # Uncertainty-aware experiments
├── experiment_certain_only.yaml
└── experiment_uncertainty_features.yaml
```

## Testing

### Full Functionality Tests (`test_code.py`)
Tests complete pipeline with all dependencies:
- Data splitting functionality
- Custom dataset loading
- Custom datamodule setup
- Build features factory

## Expected Outputs

For each experiment run, a new subfolder is created in the specified output directory (default: `./experiments`):

```
experiments/
└── irrigation_classification_20241201_143022/
    ├── experiment.yaml          # Copy of experiment configuration
    ├── model.pkl               # Trained model (serialized)
    ├── metrics.json            # Evaluation metrics
    ├── visualization.png       # Prediction plots and confusion matrices
    ├── config.yaml            # Snapshot of configuration
    └── run.log                # Complete experiment log
```

## Machine Learning Pipeline Components

### `build_features.py`
- **`get_datamodule()`**: Factory function for creating datamodules
- **`flatten_dataset()`**: Converts multi-dimensional imagery to tabular format for classical ML

### `data_splitting.py`
- **`IrrigationDataSplitter`**: Spatial-aware data splitting class
- **`spatial_stratified_split()`**: Location-based train/val/test splitting
- **`cross_validation_split()`**: K-fold CV with spatial awareness
- **`experiment_with_bands()`**: Band-specific experiment generation

### `ml_model.py`
- **`train_randomForest()`**: Random Forest training
- **`train_gradientBoosting()`**: Gradient Boosting training

### `evaluation.py`
- **`model_metrics()`**: Accuracy, F1 score, and other metrics

### `visualization.py`
- **`print_confusion_matrix()`**: Confusion matrix visualization
- **`plot_ml_predictions()`**: Pixel-wise prediction visualization

## Important Notes

### Data Requirements
- **Terratorch**: Uses standard terratorch dataset format
- **Custom**: Requires .tif files (518 bands) + .json metadata files
- **Labels**: Currently implemented as placeholder zeros - implement proper label loading

### Performance Considerations
- **Memory**: Large datasets may require reduced batch sizes
- **Workers**: Use `num_workers: 0` for debugging
- **Spatial splitting**: Ensures no data leakage between train/test sets

### Best Practices
1. **Commit code** before running experiments
2. **Test with small subsets** first
3. **Use spatial splitting** for irrigation classification
4. **Validate data format** before running experiments
5. **Check file paths** and permissions

## Troubleshooting

### Common Issues
1. **File not found**: Check file paths in configuration
2. **Band index errors**: Verify `label_bands` contains valid indices (1-8)
3. **Memory issues**: Reduce `batch_size` or `num_workers`
4. **Import errors**: Install missing dependencies

### Debugging Tips
- Use `num_workers: 0` for easier debugging
- Start with small file subsets
- Check file permissions and paths
- Verify data format matches expectations
- Run `test_code_simple.py` first

## References

- **Terratorch**: https://github.com/microsoft/torchgeo
- **PyTorch Lightning**: https://lightning.ai/docs/pytorch/
- **Sentinel-2**: https://sentinel.esa.int/web/sentinel/missions/sentinel-2
- **Irrigation Classification**: Based on smallholder irrigation dataset methodology 
>>>>>>> Stashed changes
