import os
import sys
import yaml
import json
from datetime import datetime
from joblib import dump
from ml_pipeline.build_features import get_datamodule ,flatten_dataset
from ml_pipeline.ml_model import train_randomForest, train_GradientBoosting
from ml_pipeline.evaluation import model_metrics
from ml_pipeline.visualization import print_confusion_matix, plot_ml_predictions

def load_experiments(config_path="experiments.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["experiments"]

def run_experiment(exp_cfg):
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
    config_snapshot_path = os.path.join(experiment_dir, "config.yaml")
    log_path = os.path.join(experiment_dir, "run.log")

    # Log file setup
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    original_stdout = sys.stdout
    with open(log_path, 'w') as log_file:
        sys.stdout = log_file
        try:
            print(f"[{timestamp}] Starting experiment: {base_name}")
            print(f"Saving outputs to: {experiment_dir}")

            # Save config snapshot
            with open(config_snapshot_path, "w") as f:
                yaml.dump(exp_cfg, f)

            # Load data
            dataset_path = exp_cfg["data"]["dataset_path"]
            train_size = exp_cfg["data"].get("train_subset_size", None)
            test_size = exp_cfg["data"].get("test_subset_size", None)

            datamodule = get_datamodule(dataset_path)
            datamodule.setup("fit")
            train_dataset = datamodule.train_dataset
            datamodule.setup("test")
            test_dataset = datamodule.test_dataset

            X_train, y_train = flatten_dataset(train_dataset)
            X_test, y_test = flatten_dataset(test_dataset)

            if train_size:
                X_train, y_train = X_train[:train_size], y_train[:train_size]
            if test_size:
                X_test, y_test = X_test[:test_size], y_test[:test_size]

            # Train model
            model_type = exp_cfg["model"]["type"].lower()
            hyperparams = exp_cfg["model"].get("hyperparameters", {})

            if model_type == "random_forest":
                clf = train_randomForest(X_train, y_train, **hyperparams)
            elif model_type == "gradient_boosting":
                clf = train_GradientBoosting(X_train, y_train, **hyperparams)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            dump(clf, model_path)
            print(f"Model saved to {model_path}")

            # Predict and evaluate
            y_pred = clf.predict(X_test)
            metrics = model_metrics(y_pred, y_test)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print("Metrics:", metrics)

            # Visualization
            class_names = train_dataset.class_names
            colors = exp_cfg["visualization"]["colors"]
            num_samples = exp_cfg["visualization"].get("num_samples", 2)

            print_confusion_matix(y_test, y_pred)
            plot_ml_predictions(
                test_dataset, clf, class_names, colors,
                num_samples=num_samples, save_path=visualization_path
            )
            print(f"Visualizations saved to {visualization_path}")
            print(f"[{timestamp}] Experiment complete.")

        finally:
            sys.stdout = original_stdout
            print(f"✅ Logged output to {log_path}")

def main():
    experiments = load_experiments()
    for exp_cfg in experiments:
        run_experiment(exp_cfg)

if __name__ == "__main__":
    main()