import os
import sys
import yaml
import json
import shutil
from datetime import datetime
from joblib import dump
import numpy as np
from tqdm import tqdm
from ml_pipeline.ml_model import train_model
from ml_pipeline.evaluation import model_metrics, metrics_over_factors, plot_metrics_over_factors, export_feature_importances, plot_band_time_importance
import glob  # Place at the top of the file if not already present
from ml_pipeline.visualization import plot_ml_predictions
from custom_dataset import MultiTemporalCropDataset, SHORT_BAND_NAMES
from ml_pipeline.build_features import (
    flatten_dataset,
    compute_nan_stats_for_dataset,
    time_interpolate_features,
)
from ml_pipeline.evaluation import plot_band_importance, plot_time_importance


# --- Suppress Rasterio NotGeoreferencedWarning if present ---
import warnings
try:
    from rasterio.errors import NotGeoreferencedWarning
    warnings.filterwarnings(
        "ignore",
        category=NotGeoreferencedWarning,
        module=r"rasterio"
    )
except Exception:
    # rasterio might not be installed in some environments; ignore if import fails
    pass

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
        return

    os.makedirs(experiment_dir, exist_ok=True)

    model_path = os.path.join(experiment_dir, "model.pkl")
    metrics_path = os.path.join(experiment_dir, "metrics.json")
    visualization_path = os.path.join(experiment_dir, "prediction_visualization.png")
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

            data_cfg   = exp_cfg.get("data", {})
            train_dir  = data_cfg.get("train_dir") 
            val_dir    = data_cfg.get("val_dir")   
            image_bands = data_cfg.get("image_bands")

            if not image_bands:
                BAND_NAMES = SHORT_BAND_NAMES
            else:
                BAND_NAMES = image_bands

            print("Loading train dataset...")
            train_dataset = MultiTemporalCropDataset(data_dir=train_dir, image_band_names=image_bands)
            print(f"Train dataset loaded. Samples: {len(train_dataset)}")
            print("Loading val dataset...")
            val_dataset = MultiTemporalCropDataset(data_dir=val_dir, image_band_names=image_bands)
            print(f"Val dataset loaded. Samples: {len(val_dataset)}")
            print(f"Loaded datasets -> train: {len(train_dataset)} | val: {len(val_dataset)}")
            if len(train_dataset) > 0:
                sample_train = train_dataset[0]
                print(f"First train sample image shape: {sample_train['image'].shape}, mask shape: {sample_train['mask'].shape}")
            if len(val_dataset) > 0:
                sample_val = val_dataset[0]
                print(f"First val sample image shape: {sample_val['image'].shape}, mask shape: {sample_val['mask'].shape}")

            N_TIMESTEPS = sample_train["image"].shape[1]

            # --- NaN statistics (per band x time) saved under experiment_dir/nan_stats ---
            nan_stats_dir = os.path.join(experiment_dir, "nan_stats")
            compute_nan_stats_for_dataset(
                train_dataset,
                os.path.join(nan_stats_dir, "train"),
                split_name="train",
                save_per_sample=False
            )
            compute_nan_stats_for_dataset(
                val_dataset,
                os.path.join(nan_stats_dir, "val"),
                split_name="val",
                save_per_sample=False
            )

            print("Flattening train dataset...")
            X_train, y_train = flatten_dataset(train_dataset)
            # if X_train.size:
            #     print(f"[DEBUG] X_train dtype: {X_train.dtype}  min: {np.nanmin(X_train)}  max: {np.nanmax(X_train)}")
            #     print(f"[DEBUG] X_train NaNs: {np.isnan(X_train).sum()}")
            # else:
            #     print("[DEBUG] X_train is EMPTY")
            # if y_train.size:
            #     print(f"[DEBUG] y_train dtype: {y_train.dtype}  min: {np.nanmin(y_train)}  max: {np.nanmax(y_train)}")
            #     print(f"[DEBUG] y_train NaNs: {np.isnan(y_train).sum()}")
            # else:
            #     print("[DEBUG] y_train is EMPTY")
            # print(f"After flatten_dataset(train_dataset):")
            # print(f"  X_train shape: {X_train.shape}")
            # print(f"  y_train shape: {y_train.shape}")
            # if isinstance(X_train, np.ndarray):
            #     print(f"  X_train dtype: {X_train.dtype}")
            # if isinstance(y_train, np.ndarray):
            #     print(f"  y_train dtype: {y_train.dtype}")
            # print(f"  X_train min: {np.nanmin(X_train) if X_train.size > 0 else 'EMPTY'}")
            # print(f"  X_train max: {np.nanmax(X_train) if X_train.size > 0 else 'EMPTY'}")
            # print(f"  y_train min: {np.nanmin(y_train) if y_train.size > 0 else 'EMPTY'}")
            # print(f"  y_train max: {np.nanmax(y_train) if y_train.size > 0 else 'EMPTY'}")

            print("Flattening val dataset...")
            X_val, y_val = flatten_dataset(val_dataset)
            # print(f"[DEBUG] X_val shape: {X_val.shape}")
            # print(f"[DEBUG] y_val shape: {y_val.shape}")
            # if X_val.size:
            #     print(f"[DEBUG] X_val dtype: {X_val.dtype}  min: {np.nanmin(X_val)}  max: {np.nanmax(X_val)}")
            #     print(f"[DEBUG] X_val NaNs: {np.isnan(X_val).sum()}")
            # else:
            #     print("[DEBUG] X_val is EMPTY")
            # if y_val.size:
            #     print(f"[DEBUG] y_val dtype: {y_val.dtype}  min: {np.nanmin(y_val)}  max: {np.nanmax(y_val)}")
            #     print(f"[DEBUG] y_val NaNs: {np.isnan(y_val).sum()}")
            # else:
            #     print("[DEBUG] y_val is EMPTY")
            # print(f"After flatten_dataset(val_dataset):")
            # print(f"  X_val shape: {X_val.shape}")
            # print(f"  y_val shape: {y_val.shape}")
            # if isinstance(X_val, np.ndarray):
            #     print(f"  X_val dtype: {X_val.dtype}")
            # if isinstance(y_val, np.ndarray):
            #     print(f"  y_val dtype: {y_val.dtype}")
            # print(f"  X_val min: {np.nanmin(X_val) if X_val.size > 0 else 'EMPTY'}")
            # print(f"  X_val max: {np.nanmax(X_val) if X_val.size > 0 else 'EMPTY'}")
            # print(f"  y_val min: {np.nanmin(y_val) if y_val.size > 0 else 'EMPTY'}")
            # print(f"  y_val max: {np.nanmax(y_val) if y_val.size > 0 else 'EMPTY'}")


            # Preserve full label tensors for evaluation; use only first two bands for training/inference
            y_train_full = y_train.copy()
            y_val_full = y_val.copy()

            # --- NaN handling options: none | drop | temporal ---
            impute_cfg = exp_cfg.get("imputation", {})
            mode = str(impute_cfg.get("mode", "temporal")).lower()
            fill_const = float(impute_cfg.get("fill_constant", 0.0))

            if mode == "temporal":
                print(f"Imputing features with temporal interpolation (fill_constant={fill_const})...")
                X_train = time_interpolate_features(X_train, T=N_TIMESTEPS, C=len(BAND_NAMES), fill_constant=fill_const)
                X_val   = time_interpolate_features(X_val,   T=N_TIMESTEPS, C=len(BAND_NAMES), fill_constant=fill_const)
                print("[DEBUG] After temporal imputation ->",
                      f"X_train NaNs: {np.isnan(X_train).sum()} | X_val NaNs: {np.isnan(X_val).sum()}")

            elif mode == "drop":
                print("Dropping rows with any NaN in features (train & val)...")
                # Train
                mask_tr = ~np.any(np.isnan(X_train), axis=1)
                dropped_tr = int((~mask_tr).sum())
                if dropped_tr > 0:
                    print(f"[INFO] Dropping {dropped_tr} / {X_train.shape[0]} train rows due to NaNs")
                X_train = X_train[mask_tr]
                y_train_full = y_train_full[mask_tr]
                # Val
                mask_va = ~np.any(np.isnan(X_val), axis=1)
                dropped_va = int((~mask_va).sum())
                if dropped_va > 0:
                    print(f"[INFO] Dropping {dropped_va} / {X_val.shape[0]} val rows due to NaNs")
                X_val = X_val[mask_va]
                y_val_full = y_val_full[mask_va]

            elif mode == "none":
                print("NaN handling: none (leaving NaNs as-is). WARNING: most sklearn models cannot handle NaNs.")
            else:
                print(f"[WARN] Unknown imputation mode '{mode}'. Using 'temporal' by default.")
                X_train = time_interpolate_features(X_train, T=N_TIMESTEPS, C=len(BAND_NAMES), fill_constant=fill_const)
                X_val   = time_interpolate_features(X_val,   T=N_TIMESTEPS, C=len(BAND_NAMES), fill_constant=fill_const)

            y_train = y_train_full[:, :2]
            y_val_train_only = y_val_full[:, :2]
            print(f"y_train (first two bands) shape: {y_train.shape}")
            print(f"y_val   (first two bands) shape: {y_val_train_only.shape}")

            model_type = exp_cfg["model"]["type"].lower()
            hyperparams = exp_cfg["model"].get("hyperparameters", {}).get(model_type, {})

            print("Training model...")
            clf = train_model(X_train, y_train, model_type, **hyperparams)
            print("Model training complete.")

            dump(clf, model_path)
            print(f"Model saved to {model_path}")

            print("Running predictions...")
            # Predict in batches with a progress bar to avoid long pauses on large arrays
            inference_cfg = exp_cfg.get("inference", {})
            batch_size = int(inference_cfg.get("batch_size", 250000))  # default large to minimize overhead
            n_rows = X_val.shape[0]
            y_pred_chunks = []
            for start in tqdm(range(0, n_rows, batch_size), desc="Predicting", unit="rows"):
                end = min(start + batch_size, n_rows)
                y_pred_chunks.append(clf.predict(X_val[start:end]))
            y_pred = np.concatenate(y_pred_chunks, axis=0)
            print(f"y_pred shape: {y_pred.shape}")

            metrics = model_metrics(y_pred, y_val_train_only)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print("Metrics:", metrics)


            # --- Save metrics over factors if configured ---
            comp_deatailed_cfg = exp_cfg.get("evaluation", {}).get("compute_detailed_metrics")
            if comp_deatailed_cfg:
                try:
                    # Dimensions and identifiers
                    n_imgs = len(val_dataset)
                    H = val_dataset[0]["image"].shape[2]  # Height
                    W = val_dataset[0]["image"].shape[3]  # Width

                    # Try to get unique IDs from the validation dataset directly
                    raw_ids = list(getattr(val_dataset, "paired_unique_ids", []))
                    if not raw_ids or len(raw_ids) != n_imgs:
                        # fallback to pulling from __getitem__ if needed
                        raw_ids = [val_dataset[i]["id"] for i in range(n_imgs)]
                    ids = np.array([int(str(s).split('_', 1)[0]) for s in raw_ids], dtype=int)

                    # --- Select Band 2 (index 1) for presence (binary) ---
                    target_idx = 1
                    y_pred_band2 = y_pred[:, target_idx].astype(int).reshape(n_imgs, H, W)
                    y_test_band2 = y_val_full[:, target_idx].astype(int).reshape(n_imgs, H, W)

                    # Label metadata from LAST 6 bands of y_val_full: shape (n_imgs, 6, H, W)
                    label_metadata = (
                        y_val_full[:, -6:]             # (pixels, 6)
                        .reshape(n_imgs, H, W, 6)      # (n_imgs, H, W, 6)
                        .transpose(0, 3, 1, 2)         # (n_imgs, 6, H, W)
                        .astype(int)
                    )

                    # Output dirs for detailed metrics
                    detailed_dir = os.path.join(experiment_dir, "detailed_metrics")
                    plots_dir = os.path.join(detailed_dir, "plots")
                    os.makedirs(plots_dir, exist_ok=True)

                    # Compute and save JSON under detailed_dir
                    metrics = metrics_over_factors(
                        y_pred=y_pred_band2,
                        y_test=y_test_band2,
                        multi_class=False,
                        label_metadata=label_metadata,
                        ids=ids,
                        metrics_path=detailed_dir
                    )
                    print(f"Saved detailed metrics JSON to: {os.path.join(detailed_dir, 'metrics.json')}")

                    # Plot and save PNGs under detailed_dir/plots
                    try:
                        plot_metrics_over_factors(metrics_json=metrics, save_dir=plots_dir)
                        print(f"Saved detailed metrics plots to: {plots_dir}")
                    except Exception as plot_e:
                        import traceback as _tb
                        print("Failed to plot detailed metrics:", plot_e)
                        _tb.print_exc()

                except Exception as e:
                    import traceback
                    print("Failed to compute metrics over factors:", e)
                    traceback.print_exc()


            num_samples = exp_cfg["visualization"].get("num_samples", 2)
            print("Generating prediction visualizations...")
            plot_ml_predictions(
                val_dataset, clf,
                num_samples=num_samples, save_path=visualization_path
            )

            save_feat_imp = exp_cfg.get("model", {}).get("save_feature_importance", False)
            if save_feat_imp and hasattr(clf, "estimators_"):
                # --- Create subfolders for CSVs and PNGs ---
                fi_root_dir = os.path.join(experiment_dir, "feature_importance")
                fi_csv_dir = os.path.join(fi_root_dir, "csv")
                fi_plot_dir = os.path.join(fi_root_dir, "plots")
                os.makedirs(fi_csv_dir, exist_ok=True)
                os.makedirs(fi_plot_dir, exist_ok=True)
                # --- Export feature importances to CSV subfolder ---
                export_feature_importances(clf, BAND_NAMES, N_TIMESTEPS, fi_csv_dir)
                # Discover CSVs from the csv/ folder
                fi_csvs = glob.glob(os.path.join(fi_csv_dir, "feature_importance*.csv"))
                print(f"Found feature importance CSV files: {fi_csvs}")
                if not fi_csvs:
                    print(f"Warning: No feature importance files found in {fi_csv_dir}.")
                else:
                    band_csv = os.path.join(fi_csv_dir, "feature_importance_by_band.csv")
                    time_csv = os.path.join(fi_csv_dir, "feature_importance_by_time.csv")
                    # Support either of these names for band-time details
                    band_time_csv = os.path.join(fi_csv_dir, "feature_importance_detailed.csv")

                    model_tag = model_type
                    if os.path.exists(band_csv):
                        png_path = os.path.join(fi_plot_dir, f"band_importance_{model_tag}.png")
                        try:
                            plot_band_importance(band_csv, band_names= BAND_NAMES, save_path=png_path)
                            print(f"Plotted band importance to {png_path}")
                        except Exception as e:
                            print(f"Failed to plot band importance for {band_csv}: {e}")
                    else:
                        print(f"Band importance CSV not found: {band_csv}")

                    if os.path.exists(time_csv):
                        png_path = os.path.join(fi_plot_dir, f"time_importance_{model_tag}_T{N_TIMESTEPS}.png")
                        try:
                            plot_time_importance(time_csv, num_timesteps=N_TIMESTEPS, save_path=png_path)
                            print(f"Plotted time importance to {png_path}")
                        except Exception as e:
                            print(f"Failed to plot time importance for {time_csv}: {e}")
                    else:
                        print(f"Time importance CSV not found: {time_csv}")

                    if band_time_csv and os.path.exists(band_time_csv):
                        png_path = os.path.join(fi_plot_dir, f"band_time_importance_{model_tag}_T{N_TIMESTEPS}.png")
                        try:
                            plot_band_time_importance(
                                importance_df = band_time_csv,
                                band_names= BAND_NAMES,
                                num_timesteps=N_TIMESTEPS,
                                save_path=png_path
                            )
                            print(f"Plotted band-time heatmap to {png_path}")
                        except Exception as e:
                            print(f"Failed to plot band-time heatmap for {band_time_csv}: {e}")
                    else:
                        print(f"Band-time heatmap CSV not found: {band_time_csv}")
                    print(f"Feature importance CSVs saved under: {fi_csv_dir}")
                    print(f"Feature importance plots saved under: {fi_plot_dir}")
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