import os
import sys
import yaml
import json
import shutil
from datetime import datetime
from joblib import dump
from ml_pipeline.ml_model import train_model
from ml_pipeline.evaluation import model_metrics
from ml_pipeline.visualization import plot_ml_predictions
from custom_dataset import MultiTemporalCropDataset

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

            # Prepare data
            train_dir = exp_cfg["data"]["train_dir"]
            val_dir = exp_cfg["data"]["val_dir"]

            train_dataset = MultiTemporalCropDataset(
                image_dir=train_dir,
                label_dir=train_dir,
                label_bands=exp_cfg["data"]["label_bands"]
            )
            val_dataset = MultiTemporalCropDataset(
                image_dir=val_dir,
                label_dir=val_dir,
                label_bands=exp_cfg["data"]["label_bands"]
            )

            from ml_pipeline.build_features import flatten_dataset
            X_train, y_train = flatten_dataset(train_dataset)
            X_val, y_val = flatten_dataset(val_dataset)

            # Select only first two label bands for ML training/validation
            # This restricts ML training to the first two bands, reserving other bands for post-hoc analysis
            y_train = y_train[:, :2]
            y_val = y_val[:, :2]

            # Train model
            model_type = exp_cfg["model"]["type"].lower()
            hyperparams = exp_cfg["model"].get("hyperparameters", {}).get(model_type, {}) # get the hyperparameters specific to this model

            clf = train_model(X_train, y_train, model_type, **hyperparams)

            dump(clf, model_path) # Warning -- with deep learning models we will want to do this in epochs
            print(f"Model saved to {model_path}")

            # Predict and evaluate
            y_pred = clf.predict(X_val)
            metrics = model_metrics(y_pred, y_val)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print("Metrics:", metrics)

            # Visualization
            num_samples = exp_cfg["visualization"].get("num_samples", 2)

            plot_ml_predictions(
                val_dataset, clf,
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