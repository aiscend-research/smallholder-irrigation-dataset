
## Overview
This folder contains a modular pipeline for training and evaluating a Random Forest classifier on multi-temporal satellite imagery using datasets in the TerraTorch (TorchGeo-style) format.

## Warning
The pipeline is specifically designed for the multi-temporal-crop-dataset as used by terratorch, so many things will need to changed for a different dataset.

## Structure
```
├── main.py                    # Entry point to run the full pipeline
├── prototyping.ipynb          # Notebook for interactive experimentation
├── README.md                  # This file
├── rf_pipeline/               # Core ML pipeline modules
│   ├── __init__.py            # Makes rf_pipeline a Python package
│   ├── build_features.py      # Data loading and flattening
│   ├── rf_model.py            # Model training
│   ├── evaluation.py          # Model prediction and metrics
│   └── visualization.py       # Plotting predictions and confusion matrix
```

#### `build_features.py`
`get_datamodule(dataset_path)`: Loads a TerraTorch-style datamodule from the given path.
`flatten_dataset(dataset)`: Converts multi-dimensional satellite imagery into a **tabular format** where each pixel becomes a row with multi-temporal features. This is so that you can run classical ML models like Random Forests. Flattening removes spatial context, so this is intended for pixel-wise classification only.


 #### `rf_model.py`
 Contains a `train_randomForest` function that uses sklearn to train a randomForest model on training data.

 #### `evaluation.py`
 Contains `predict` and `model_metrics` functions that use sklearn to predict and provide metrics such as the accuracy and the f1 score.

 #### `visualization.py`
 `print_confusion_matix(y_true, y_pred)`: Displays a confusion matrix using `matplotlib`.
`plot_rf_predictions(dataset, clf, class_names, colors)`: Visualizes the model’s pixel-wise predictions vs ground truth masks for selected samples, using color-coded masks. Class colors and names are dataset-dependent and should be passed in as arguments. This function will have to changed depending on the dataset.