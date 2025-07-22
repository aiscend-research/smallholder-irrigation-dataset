
## Overview
This folder contains a machine learning pipeline for running experiments on multi-temporal satellite imagery. The dataset format is the TerraTorch (TorchGeo-style) format.

## Warning
The pipeline is specifically designed for the multi-temporal-crop-dataset as used by terratorch, so it may not work for other datasets. This sample dataset is currently located in the cluster under the path `/home/waves/data/smallholder-irrigation-dataset/data/modeling/test_data`. Code in the modeling folder has not been updated to run with this file path.

## About the test dataset
The test dataset is from one of the example multi-temporal-crop datasets used by terratorch. Can be found at (https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification). 

## Structure
```
├── run_experiment.py          # Entry point to run an experiment
├── final_test.py              # Entry point to evaluate a best model's performance to  the test dataset (in progress)
├── experiments.yaml           # Specify experiment details
├── README.md                  # This file
├── ml_pipeline/               # Core ML pipeline modules
│   ├── __init__.py            # Makes ml_pipeline a Python package
│   ├── build_features.py      # Data loading and flattening
│   ├── ml_model.py            # Model training
│   ├── evaluation.py          # Model prediction and metrics
│   └── visualization.py       # Plotting predictions and confusion matrix
```
### Machine Learning Pipeline:

#### `build_features.py`
`get_datamodule(dataset_path)`: Loads a TerraTorch-style datamodule from the given path.
`flatten_dataset(dataset)`: Converts multi-dimensional satellite imagery into a **tabular format** where each pixel becomes a row with multi-temporal features. This is so that you can run classical ML models like Random Forests. Flattening removes spatial context, so this is intended for pixel-wise classification only.

 #### `ml_model.py`
 Contains a `train_randomForest` function that uses sklearn to train a randomForest model on training data. Contains a similar function for a gradient boosted model.

 #### `evaluation.py`
 Contains the `model_metrics` function that use sklearn to provide metrics such as the accuracy and the f1 score.

 #### `visualization.py`
 `print_confusion_matix(y_true, y_pred)`: Displays a confusion matrix using `matplotlib`.
`plot_ml_predictions(dataset, clf, class_names, colors)`: Visualizes the model’s pixel-wise predictions vs ground truth masks for selected samples, using color-coded masks. Class colors and names are dataset-dependent and should be passed in as arguments. This function will have to changed depending on the dataset.

### Running Experiments Using the ML Pipeline

This pipeline is designed to make it easy to run, track, and reproduce machine learning experiments. The workflow is configuration-driven, so you can specify all experiment details in a YAML file and run them with a single script.

#### **Configuration: `experiment.yaml`**

- This file contains all the settings for a single experiment, including:
  - **Experiment name** (for tracking)
  - **Data settings** (train/validation/test subset sizes)
  - **Model type** (e.g., `random_forest`, `gradient_boosting`)
  - **Model hyperparameters** (grouped by model type)
  - **Visualization options** (class colors, number of samples to plot)
  - **Output directory** (where results will be saved)

The `experiment.yaml` file is listed in the `.gitignore`, so users may tweak it at their will for a given experiment, but the default configurations will remain untouched. 

#### **Running Experiments: `run_experiment.py`**

- This script loads the experiment configuration from `experiment.yaml` and runs the full ML pipeline. Dataset path is specified in this file. 

#### **Expected Outputs**

For each experiment run, a new subfolder is created in the specified output directory (default: `./experiments`). The name of the folder corresponds to the experiment name as specified in `experiment.yaml, concatenated with a datetime stamp. 

This folder contains:
- `experiment.yaml`: A copy of the experiment configurations used to devise this experiment
- `model.pkl`: The trained model, serialized with joblib.
- `metrics.json`: Evaluation metrics (accuracy, F1 score, etc.) on the validation set.
- `visualization.png`: Plots of model predictions and confusion matrices.
- `config.yaml`: A snapshot of the exact configuration used for this run.
- `run.log`: A complete log of the experiment, including all print statements and errors.

This structure ensures that every experiment is fully reproducible: you can always trace back from results to the exact code and configuration used.

#### **Best Practices**

- **Commit your code before running experiments.** This ensures you can always match results to the code that produced them. 