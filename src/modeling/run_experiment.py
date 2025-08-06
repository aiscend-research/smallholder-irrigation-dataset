import os
import sys
import yaml
import json
import shutil
from datetime import datetime
from joblib import dump
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from ml_pipeline.ml_model import train_model
from ml_pipeline.evaluation import model_metrics
from ml_pipeline.evaluation import export_feature_importances
import glob  # Place at the top of the file if not already present
from ml_pipeline.visualization import plot_ml_predictions
from custom_dataset import MultiTemporalCropDataset

def load_experiment(config_path="experiment.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_experiment(exp_cfg, config_path):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = exp_cfg["name"]
    run_name = f"{base_name}_{timestamp}"

    base_dir = exp_cfg["output"]["base_dir"]
    experiment_dir = os.path.join(base_dir, run_name)

    if os.path.exists(experiment_dir):
        print(f"Skipping: {run_name} already exists.")
        return

    os.makedirs(experiment_dir, exist_ok=True)

    model_path = os.path.join(experiment_dir, "model.pkl")
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    visualization_path = os.path.join(experiment_dir, "visualization.png")
    config_snapshot_path = os.path.join(experiment_dir, "experiment.yaml")
    log_path = os.path.join(experiment_dir, "run.log")

    shutil.copyfile(config_path, config_snapshot_path)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    original_stdout = sys.stdout
    with open(log_path, 'w') as log_file:
        sys.stdout = log_file
        try:
            print(f"[{timestamp}] Starting experiment: {base_name}")
            print(f"Saving outputs to: {experiment_dir}")

            data_dir = exp_cfg["data"]["data_dir"]
            image_bands = exp_cfg["data"]["image_bands"]

            print("Loading full dataset...")
            full_dataset = MultiTemporalCropDataset(data_dir=data_dir, image_band_names=image_bands)
            total_samples = len(full_dataset)
            print(f"Dataset loaded. Total samples: {total_samples}")

            train_indices = list(range(8))
            val_indices = list(range(8, 10))
            train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

            print(f"Loaded full dataset: {total_samples} samples")
            print(f"Train dataset length: {len(train_dataset)}")
            print(f"Val dataset length: {len(val_dataset)}")
            if len(train_dataset) > 0:
                sample_train = train_dataset[0]
                print(f"First train sample image shape: {sample_train['image'].shape}, mask shape: {sample_train['mask'].shape}")
            if len(val_dataset) > 0:
                sample_val = val_dataset[0]
                print(f"First val sample image shape: {sample_val['image'].shape}, mask shape: {sample_val['mask'].shape}")

            from ml_pipeline.build_features import flatten_dataset

            print("Flattening train dataset...")
            # Wrap flatten_dataset with tqdm progress bar if possible
            X_train, y_train = flatten_dataset(train_dataset)
            print(f"[DEBUG] X_train shape: {X_train.shape}")
            print(f"[DEBUG] y_train shape: {y_train.shape}")
            if X_train.size:
                print(f"[DEBUG] X_train dtype: {X_train.dtype}  min: {np.nanmin(X_train)}  max: {np.nanmax(X_train)}")
                print(f"[DEBUG] X_train NaNs: {np.isnan(X_train).sum()}")
            else:
                print("[DEBUG] X_train is EMPTY")
            if y_train.size:
                print(f"[DEBUG] y_train dtype: {y_train.dtype}  min: {np.nanmin(y_train)}  max: {np.nanmax(y_train)}")
                print(f"[DEBUG] y_train NaNs: {np.isnan(y_train).sum()}")
            else:
                print("[DEBUG] y_train is EMPTY")
            print(f"After flatten_dataset(train_dataset):")
            print(f"  X_train shape: {X_train.shape}")
            print(f"  y_train shape: {y_train.shape}")
            if isinstance(X_train, np.ndarray):
                print(f"  X_train dtype: {X_train.dtype}")
            if isinstance(y_train, np.ndarray):
                print(f"  y_train dtype: {y_train.dtype}")
            print(f"  X_train min: {np.nanmin(X_train) if X_train.size > 0 else 'EMPTY'}")
            print(f"  X_train max: {np.nanmax(X_train) if X_train.size > 0 else 'EMPTY'}")
            print(f"  y_train min: {np.nanmin(y_train) if y_train.size > 0 else 'EMPTY'}")
            print(f"  y_train max: {np.nanmax(y_train) if y_train.size > 0 else 'EMPTY'}")

            print("Flattening val dataset...")
            X_val, y_val = flatten_dataset(val_dataset)
            print(f"[DEBUG] X_val shape: {X_val.shape}")
            print(f"[DEBUG] y_val shape: {y_val.shape}")
            if X_val.size:
                print(f"[DEBUG] X_val dtype: {X_val.dtype}  min: {np.nanmin(X_val)}  max: {np.nanmax(X_val)}")
                print(f"[DEBUG] X_val NaNs: {np.isnan(X_val).sum()}")
            else:
                print("[DEBUG] X_val is EMPTY")
            if y_val.size:
                print(f"[DEBUG] y_val dtype: {y_val.dtype}  min: {np.nanmin(y_val)}  max: {np.nanmax(y_val)}")
                print(f"[DEBUG] y_val NaNs: {np.isnan(y_val).sum()}")
            else:
                print("[DEBUG] y_val is EMPTY")
            print(f"After flatten_dataset(val_dataset):")
            print(f"  X_val shape: {X_val.shape}")
            print(f"  y_val shape: {y_val.shape}")
            if isinstance(X_val, np.ndarray):
                print(f"  X_val dtype: {X_val.dtype}")
            if isinstance(y_val, np.ndarray):
                print(f"  y_val dtype: {y_val.dtype}")
            print(f"  X_val min: {np.nanmin(X_val) if X_val.size > 0 else 'EMPTY'}")
            print(f"  X_val max: {np.nanmax(X_val) if X_val.size > 0 else 'EMPTY'}")
            print(f"  y_val min: {np.nanmin(y_val) if y_val.size > 0 else 'EMPTY'}")
            print(f"  y_val max: {np.nanmax(y_val) if y_val.size > 0 else 'EMPTY'}")

            y_train = y_train[:, :2]
            y_val = y_val[:, :2]
            print(f"y_train (first two bands) shape: {y_train.shape}")
            print(f"y_val (first two bands) shape: {y_val.shape}")

            model_type = exp_cfg["model"]["type"].lower()
            hyperparams = exp_cfg["model"].get("hyperparameters", {}).get(model_type, {})

            print("Training model...")
            clf = train_model(X_train, y_train, model_type, **hyperparams)
            print("Model training complete.")

            dump(clf, model_path)
            print(f"Model saved to {model_path}")

            print("Running predictions...")
            y_pred = clf.predict(X_val)
            print(f"y_pred shape: {y_pred.shape}")

            metrics = model_metrics(y_pred, y_val)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print("Metrics:", metrics)

            num_samples = exp_cfg["visualization"].get("num_samples", 2)
            print("Generating prediction visualizations...")
            plot_ml_predictions(
                val_dataset, clf,
                num_samples=num_samples, save_path=visualization_path
            )

            save_feat_imp = exp_cfg.get("model", {}).get("save_feature_importance", False)
            if save_feat_imp and hasattr(clf, "estimators_"):
                BAND_NAMES = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI", "EVI", "NDWI", "SCL"]
                N_TIMESTEPS = 37
                # --- Export and plot all feature importance CSVs in the experiment_dir ---
                export_feature_importances(clf, BAND_NAMES, N_TIMESTEPS, experiment_dir)
                fi_csvs = glob.glob(os.path.join(experiment_dir, "feature_importance*.csv"))
                print(f"Found feature importance CSV files: {fi_csvs}")
                if not fi_csvs:
                    print(f"Warning: No feature importance files found in {experiment_dir}.")
                else:
                    from ml_pipeline.evaluation import plot_feature_importance_from_df
                    for fi_csv in fi_csvs:
                        try:
                            base = os.path.splitext(os.path.basename(fi_csv))[0]
                            png_path = os.path.join(experiment_dir, base + ".png")
                            plot_feature_importance_from_df(
                                fi_csv,
                                band_names=BAND_NAMES,
                                num_timesteps=N_TIMESTEPS,
                                save_path=png_path
                            )
                            print(f"Plotted feature importance for {fi_csv} to {png_path}")
                        except Exception as e:
                            print(f"Failed to plot feature importance for {fi_csv}: {e}")
            elif save_feat_imp:
                print("Warning: Requested to save feature importances, but model does not support 'estimators_'. Skipping feature importance export.")
            print(f"[{timestamp}] Experiment complete.")

        finally:
            sys.stdout = original_stdout
            print(f"Logged output to {log_path}")

if __name__ == "__main__":
    config_path = "experiment.yaml"
    experiments = load_experiment(config_path)
    if isinstance(experiments, list):
        for exp_cfg in experiments:
            run_experiment(exp_cfg, config_path)
    else:
        run_experiment(experiments, config_path)