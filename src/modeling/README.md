
## Overview
This folder contains a machine learning pipeline for running experiments on multi-temporal satellite imagery. The dataset format is the TerraTorch (TorchGeo-style) format.

## Warning
The pipeline is specifically designed for the multi-temporal-crop-dataset as used by terratorch, so it may not work for other datasets. This sample dataset is currently located in the cluster under the path `/home/waves/data/smallholder-irrigation-dataset/data/modeling/test_data`. Code in the modeling folder has not been updated to run with this file path.

## About the test dataset
The test dataset is from one of the example multi-temporal-crop datasets used by terratorch. Can be found at (https://huggingface.co/datasets/ibm-nasa-geospatial/multi-temporal-crop-classification). 

## Structure
```
├── main.py                    # Entry point to run the full pipeline
├── experiments.yaml           # Specify experiment details
├── prototyping.ipynb          # Notebook for interactive experimentation
├── README.md                  # This file
├── ml_pipeline/               # Core ML pipeline modules
│   ├── __init__.py            # Makes ml_pipeline a Python package
│   ├── build_features.py      # Data loading and flattening
│   ├── ml_model.py            # Model training
│   ├── evaluation.py          # Model prediction and metrics
│   └── visualization.py       # Plotting predictions and confusion matrix
├── experiments/               # Stores output of experiments
│    WORK IN PROGRESS
```
### `experiments.yaml`
Contains specfic configurations for each model to be run. 

### `main.py`
Loads in model configurations from the `experiments.yaml` file and runs the pipeline on each model. Saves the output to the experiments folder. 

### `experiments` 
Folder where results from experiments are saved, as of now included in the gitignore. Stucture is a work in progress. 

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
