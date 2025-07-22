import os
import sys
import yaml
import json
import shutil
from datetime import datetime
from joblib import dump
from ml_pipeline.build_features import get_datamodule ,flatten_dataset
from ml_pipeline.ml_model import train_model
from ml_pipeline.evaluation import model_metrics
from ml_pipeline.visualization import print_confusion_matix, plot_ml_predictions

#In order to access the get_data_root function form utils 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.utils import get_data_root

def load_experiment(config_path="experiment.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_experiment(exp_cfg, config_path):
    # Timestamped experiment name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = exp_cfg["name"]
    run_name = f"{base_name}_{timestamp}"

    # Resolve paths
    base_dir = exp_cfg["output"]["base_dir"]
    experiment_dir = os.path.join(base_dir, run_name)

    # Skip if directory already exists (prevent duplicates)
    if os.path.exists(experiment_dir):
        print(f"Skipping: {run_name} already exists.")
        return

    os.makedirs(experiment_dir, exist_ok=True)

    model_path = os.path.join(experiment_dir, "model.pkl")
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    visualization_path = os.path.join(experiment_dir, "visualization.png")
    config_snapshot_path = os.path.join(experiment_dir, "experiment.yaml")
    log_path = os.path.join(experiment_dir, "run.log")

    # Copy the config file used for this experiment
    shutil.copyfile(config_path, config_snapshot_path)

    # Log file setup
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    original_stdout = sys.stdout
    with open(log_path, 'w') as log_file:
        sys.stdout = log_file
        try:
            print(f"[{timestamp}] Starting experiment: {base_name}")
            print(f"Saving outputs to: {experiment_dir}")

            # Load and prepare data
            #dataset_path = os.path.join(get_data_root(), "modeling", "test") #for data on the cluster 
            dataset_path = exp_cfg["data"]["dataset_path"]
            train_size = exp_cfg["data"].get("train_subset_size", None) # Default to use all data
            val_size = exp_cfg["data"].get("val_subset_size", None)

            datamodule = get_datamodule(dataset_path)
            datamodule.setup("fit")
            train_dataset = datamodule.train_dataset
            datamodule.setup("val")
            val_dataset = datamodule.val_dataset # Because this is an experiment, we only want to use validation data

            X_train, y_train = flatten_dataset(train_dataset) # Warning -- this will need to be changed for models that don't use flattened data
            X_val, y_val = flatten_dataset(val_dataset)

            if train_size:
                X_train, y_train = X_train[:train_size], y_train[:train_size]
            if val_size:
                X_val, y_val = X_val[:val_size], y_val[:val_size]

            # Train model
            model_type = exp_cfg["model"]["type"].lower()
            hyperparams = exp_cfg["model"].get("hyperparameters", {}).get(model_type, {}) # get the hyperparameters specific to this model

            clf = train_model(X_train, y_train, model_type, **hyperparams)

            dump(clf, model_path) # Warning -- with deep learning models we will want to do this in epochs
            print(f"Model saved to {model_path}")

            # Predict and evaluate
            y_pred = clf.predict(X_val)
            metrics = model_metrics(y_pred, y_val)
            metrics_dict = {
                        "accuracy": metrics[0],
                        "f1_score": metrics[1]}
            with open(metrics_path, "w") as f:
                json.dump(metrics_dict, f, indent=2)
            print("Metrics:", metrics)

            # Visualization
            class_names = train_dataset.class_names
            colors = exp_cfg["visualization"]["colors"]
            num_samples = exp_cfg["visualization"].get("num_samples", 2)

            print_confusion_matix(y_val, y_pred)
            plot_ml_predictions(
                val_dataset, clf, class_names, colors,
                num_samples=num_samples, save_path=visualization_path
            )
            print(f"[{timestamp}] Experiment complete.")

        finally:
            sys.stdout = original_stdout
            print(f"✅ Logged output to {log_path}")


if __name__ == "__main__":
    config_path = "experiment.yaml"
    experiments = load_experiment(config_path)
    # If the config is a list of experiments, iterate; else, wrap in a list
    if isinstance(experiments, list):
        for exp_cfg in experiments:
            run_experiment(exp_cfg, config_path)
    else:
        run_experiment(experiments, config_path)