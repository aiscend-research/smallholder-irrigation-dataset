#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run K-fold cross-validation for irrigation classification with site-aware splits.

What this script does:
1) Builds spatially safe splits (grouped by siteNumeric; optional stratification)
   This also produces a held-out test list, which this script does not evaluate on.
2) Reads train/val file lists and a CV manifest CSV
   (mapping stem -> absolute image_path/label_path) — no file staging/copying.
3) For each fold: loads scenes from disk, flattens to tabular features,
   trains the model, and computes metrics on the fold's validation set.
4) Computes detailed metrics over multiple factors and generates comprehensive 
   visualization plots.
"""

import os
import sys
import json
import yaml
import shutil
import logging
import glob
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import dump

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
ROOT = project_root

from src.modeling.ml_pipeline.ml_model import train_model
from src.modeling.ml_pipeline.evaluation import model_metrics
from src.modeling.ml_pipeline.evaluation import metrics_over_factors, plot_metrics_over_factors
from src.modeling.ml_pipeline.evaluation import export_feature_importances
from src.modeling.ml_pipeline.evaluation import plot_band_time_importance
from src.modeling.ml_pipeline.evaluation import plot_band_importance, plot_time_importance
from src.modeling.ml_pipeline.data_splitting import prepare_and_export_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# Utilities
def resolve_path(path_str: str, base_dir: str | None = None) -> str:
    """Resolve a path relative to the project root if not absolute."""
    if path_str is None:
        return None
    if os.path.isabs(path_str):
        return path_str
    base = base_dir or ROOT
    return os.path.normpath(os.path.join(base, path_str))


def resolve_config_path(config_path: str = "src/modeling/experiment.yaml") -> str:
    """Return an absolute path to the experiment YAML by trying a few likely locations."""
    p = Path(config_path)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / config_path)
        candidates.append(Path(ROOT) / config_path)
        candidates.append(Path(ROOT) / "experiment.yaml")
        candidates.append(Path.cwd() / "experiment.yaml")
        candidates.append(Path(ROOT) / "src/modeling/ml_pipeline/experiment.yaml")

    for c in candidates:
        if c.exists():
            return str(c.resolve())

    tried = "\n  - " + "\n  - ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"experiment.yaml not found. Tried:{tried}")


def load_experiment(config_path: str = "src/modeling/experiment.yaml") -> dict:
    """Load the YAML config and return it as a Python dict."""
    cfg_path = resolve_config_path(config_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_stems(txt_path: str) -> list[str]:
    """Read file stems from a text file (one per line)."""
    p = Path(txt_path)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


# Direct dataset loading from manifest
def load_dataset_from_manifest(
    stems: list[str],
    manifest_df: pd.DataFrame,
    label_bands: list[int],
) -> list:
    """
    Load image/label data directly from absolute paths in CV manifest.
    Returns a list of (image_array, label_array, stem) tuples.
    """
    try:
        import rasterio
    except ImportError:
        raise ImportError("rasterio is required for direct file loading. Install with: pip install rasterio")
    
    manifest_index = manifest_df.set_index("stem")
    dataset = []
    
    logger.info(f"[load] Loading {len(stems)} samples directly from manifest paths...")
    
    missing_in_manifest = 0
    missing_files = 0
    failed_reads = 0
    
    for i, stem in enumerate(stems):
        if stem not in manifest_index.index:
            missing_in_manifest += 1
            logger.warning(f"[load] Stem '{stem}' not found in manifest, skipping")
            continue
            
        row = manifest_index.loc[stem]
        img_path = Path(str(row["image_path"]))
        lab_path = Path(str(row["label_path"]))
        
        if not img_path.exists():
            missing_files += 1
            logger.warning(f"[load] Image missing: {img_path}")
            continue
        if not lab_path.exists():
            missing_files += 1
            logger.warning(f"[load] Label missing: {lab_path}")
            continue
            
        try:
            with rasterio.open(img_path) as src:
                image = src.read()
            
            with rasterio.open(lab_path) as src:
                label = src.read(label_bands)
            
            # Store stem for ID extraction later
            dataset.append((image, label, stem))
            
            if (i + 1) % 50 == 0:
                logger.info(f"[load] Loaded {len(dataset)}/{i + 1} samples (progress: {i+1}/{len(stems)})")
                
        except Exception as e:
            failed_reads += 1
            logger.warning(f"[load] Failed to read {stem}: {e}")
            continue
    
    logger.info(f"[load] Successfully loaded {len(dataset)}/{len(stems)} samples")
    if missing_in_manifest > 0:
        logger.warning(f"[load] {missing_in_manifest} stems not found in manifest")
    if missing_files > 0:
        logger.warning(f"[load] {missing_files} files missing on disk")
    if failed_reads > 0:
        logger.warning(f"[load] {failed_reads} files failed to read")
    
    if len(dataset) == 0:
        raise RuntimeError(
            f"No valid samples were loaded from manifest. "
            f"Missing in manifest: {missing_in_manifest}, "
            f"Missing files: {missing_files}, "
            f"Failed reads: {failed_reads}"
        )
    
    return dataset


def flatten_dataset_from_tuples(dataset: list, pixels_per_image: int = None) -> tuple:
    """
    Flatten dataset from list of (image, label, stem) tuples with optional pixel sampling.
    Converts spatial image data into feature vectors for ML.
    
    Args:
        dataset: List of (image, label, stem) tuples
        pixels_per_image: Number of pixels to randomly sample per image.
                         If None, uses all pixels.
    
    Returns:
        X: Feature matrix (n_pixels, n_bands)
        y: Label matrix (n_pixels, n_label_bands)
        stems: List of stem identifiers for each sample
    """
    X_list = []
    y_list = []
    stems_list = []
    
    sampling_msg = f"sampling {pixels_per_image} pixels per image" if pixels_per_image else "using ALL pixels"
    logger.info(f"[flatten] Flattening {len(dataset)} samples ({sampling_msg})...")
    
    total_pixels_original = 0
    total_pixels_used = 0
    
    for idx, (image, label, stem) in enumerate(dataset):
        n_bands, height, width = image.shape
        total_pixels = height * width
        total_pixels_original += total_pixels
        
        # Flatten to (n_pixels, n_bands)
        X_full = image.reshape(n_bands, -1).T
        y_full = label.reshape(label.shape[0], -1).T
        
        # Sample random pixels if requested and image is larger than desired sample size
        if pixels_per_image and total_pixels > pixels_per_image:
            sample_indices = np.random.choice(total_pixels, pixels_per_image, replace=False)
            X_sample = X_full[sample_indices]
            y_sample = y_full[sample_indices]
            total_pixels_used += pixels_per_image
        else:
            # Use all pixels
            X_sample = X_full
            y_sample = y_full
            total_pixels_used += total_pixels
        
        X_list.append(X_sample)
        y_list.append(y_sample)
        stems_list.append(stem)
        
        if (idx + 1) % 100 == 0:
            logger.info(f"[flatten] Processed {idx + 1}/{len(dataset)} samples")
    
    X = np.vstack(X_list)
    y = np.vstack(y_list)
    
    logger.info(f"[flatten] Final shapes: X={X.shape}, y={y.shape}")
    if pixels_per_image and total_pixels_original > total_pixels_used:
        reduction_pct = 100 * (1 - total_pixels_used / total_pixels_original)
        logger.info(f"[flatten] Total pixels: {total_pixels_used:,} / {total_pixels_original:,} ({reduction_pct:.1f}% reduction)")
    
    return X, y, stems_list


def plot_predictions(dataset: list, model, num_samples: int = 2, save_path: str = None):
    """Visualization function for tuple-based dataset."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import matplotlib.pyplot as plt
        
        sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sample_idx in enumerate(sample_indices):
            image, label, stem = dataset[sample_idx]
            
            n_bands, height, width = image.shape
            X_sample = image.reshape(n_bands, -1).T
            y_pred = model.predict(X_sample)
            y_pred_img = y_pred.reshape(height, width)
            
            if image.shape[0] >= 3:
                rgb = np.stack([image[2], image[1], image[0]], axis=-1)
                rgb = np.clip(rgb / rgb.max() * 255, 0, 255).astype(np.uint8)
                axes[idx, 0].imshow(rgb)
                axes[idx, 0].set_title(f"Sample {sample_idx}: RGB")
            
            axes[idx, 1].imshow(label[0], cmap='viridis')
            axes[idx, 1].set_title("Ground Truth")
            
            axes[idx, 2].imshow(y_pred_img, cmap='viridis')
            axes[idx, 2].set_title("Prediction")
            
            for ax in axes[idx]:
                ax.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"[viz] Saved visualization to {save_path}")
        plt.close()
        
    except Exception as e:
        logger.warning(f"[viz] Visualization failed: {e}")


def train_and_evaluate_fold(train_stems, val_stems, manifest, label_bands, model_type, 
                           hyperparams, fold_name="", pixels_per_image=None, 
                           exp_cfg=None, fold_dir=None):
    """
    Train and evaluate a single fold with comprehensive metrics.
    
    Args:
        pixels_per_image: If None, uses all pixels. Otherwise samples this many per image.
        exp_cfg: Experiment configuration for detailed metrics
        fold_dir: Directory to save fold-specific outputs
    """
    
    logger.info(f"[{fold_name}] Loading training data...")
    train_ds = load_dataset_from_manifest(train_stems, manifest, label_bands)
    
    logger.info(f"[{fold_name}] Loading validation data...")
    val_ds = load_dataset_from_manifest(val_stems, manifest, label_bands)

    logger.info(f"[{fold_name}] Extracting features...")
    X_train, y_train_full, train_stems_list = flatten_dataset_from_tuples(train_ds, pixels_per_image=pixels_per_image)
    X_val, y_val_full, val_stems_list = flatten_dataset_from_tuples(val_ds, pixels_per_image=pixels_per_image)

    # Use only first label band for training (binary classification)
    # Band structure: 0=type, 1=binary, 2-7=uncertainty metadata, 7=certainty
    y_train = y_train_full[:, 1]  # Shape: (n_samples,) - binary irrigation presence
    y_val_for_training = y_val_full[:, 1]

    logger.info(f"[{fold_name}] Training model...")
    clf = train_model(X_train, y_train, model_type, **hyperparams)
    
    logger.info(f"[{fold_name}] Evaluating...")
    y_pred = clf.predict(X_val)
    
    # Basic metrics
    metrics = model_metrics(y_pred, y_val_for_training)
    
    # --- Detailed metrics computation (if configured) ---
    compute_detailed = exp_cfg and exp_cfg.get("evaluation", {}).get("compute_detailed_metrics", False)
    
    if compute_detailed:
        try:
            logger.info(f"[{fold_name}] Computing detailed metrics over factors...")
            
            # Get dimensions from the original dataset
            n_imgs = len(val_ds)
            first_img, first_label, _ = val_ds[0]
            H = first_img.shape[1]  # Height
            W = first_img.shape[2]  # Width
            pixels_per_img = H * W
            
            # Check if we need all 8 bands for detailed metrics
            if y_val_full.shape[1] < 8:
                logger.warning(
                    f"[{fold_name}] Not enough label bands ({y_val_full.shape[1]}) for detailed metrics. "
                    f"Need 8 bands. Skipping detailed metrics."
                )
                return clf, metrics, val_ds, len(train_ds), len(val_ds)
            
            # Handle pixel sampling case
            if pixels_per_image and pixels_per_image < pixels_per_img:
                logger.warning(
                    f"[{fold_name}] Pixel sampling is enabled ({pixels_per_image} < {pixels_per_img}). "
                    f"Detailed metrics require full spatial data. Re-predicting on full images..."
                )
                
                # Re-load validation data without sampling for detailed metrics
                val_ds_full = load_dataset_from_manifest(val_stems, manifest, label_bands)
                X_val_full, y_val_full_complete, _ = flatten_dataset_from_tuples(val_ds_full, pixels_per_image=None)
                
                # Re-predict on full data
                y_pred_full = clf.predict(X_val_full)
                
                # Reshape to spatial format
                y_pred_spatial = y_pred_full.astype(int).reshape(n_imgs, H, W)
                y_test_spatial = y_val_full_complete[:, 1].astype(int).reshape(n_imgs, H, W)
                
                # Extract metadata bands (indices 2-7, which are bands 3-8 in the file)
                label_metadata = (
                    y_val_full_complete[:, 2:8]           # Get bands 2-7 (0-indexed)
                    .reshape(n_imgs, H, W, 6)             # (n_imgs, H, W, 6)
                    .transpose(0, 3, 1, 2)                # (n_imgs, 6, H, W)
                    .astype(int)
                )
            else:
                # No sampling - reshape directly
                y_pred_spatial = y_pred.astype(int).reshape(n_imgs, H, W)
                y_test_spatial = y_val_for_training.astype(int).reshape(n_imgs, H, W)
                
                # Extract metadata bands (indices 2-7 in y_val_full)
                label_metadata = (
                    y_val_full[:, 2:8]                    # Get bands 2-7 (0-indexed)
                    .reshape(n_imgs, H, W, 6)             # (n_imgs, H, W, 6)
                    .transpose(0, 3, 1, 2)                # (n_imgs, 6, H, W)
                    .astype(int)
                )
            
            # Extract unique IDs from stems (format: {uid}_{site}_{date}_image)
            ids = np.array([int(stem.split('_')[0]) for stem in val_stems_list], dtype=int)
            
            if fold_dir:
                # Create detailed metrics directory for this fold
                detailed_dir = os.path.join(fold_dir, "detailed_metrics")
                plots_dir = os.path.join(detailed_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                # Compute metrics over factors
                detailed_metrics = metrics_over_factors(
                    y_pred=y_pred_spatial,
                    y_test=y_test_spatial,
                    multi_class=False,
                    label_metadata=label_metadata,
                    ids=ids,
                    metrics_path=detailed_dir
                )
                logger.info(f"[{fold_name}] Saved detailed metrics JSON to: {os.path.join(detailed_dir, 'metrics.json')}")
                
                # Generate plots
                try:
                    plot_metrics_over_factors(metrics_json=detailed_metrics, save_dir=plots_dir)
                    logger.info(f"[{fold_name}] Saved detailed metrics plots to: {plots_dir}")
                except Exception as plot_e:
                    logger.warning(f"[{fold_name}] Failed to plot detailed metrics: {plot_e}")
                    import traceback
                    traceback.print_exc()
        
        except Exception as e:
            import traceback
            logger.warning(f"[{fold_name}] Failed to compute detailed metrics: {e}")
            traceback.print_exc()

    return clf, metrics, val_ds, len(train_ds), len(val_ds)


def run_cv_experiment(exp_cfg: dict, experiment_dir: str):
    """K-fold CV inside the training pool, with a held-out test list available."""
    data_root = resolve_path(exp_cfg["data"]["data_dir"])
    csv_path = exp_cfg["data"].get("csv_path")
    csv_path = resolve_path(csv_path) if csv_path else None

    grit_images_dir = resolve_path(exp_cfg["data"].get("grit_images_dir")) if exp_cfg["data"].get("grit_images_dir") else None
    grit_masks_dir  = resolve_path(exp_cfg["data"].get("grit_masks_dir"))  if exp_cfg["data"].get("grit_masks_dir")  else None

    logger.info("[splits] Building CV splits...")
    paths = prepare_and_export_splits(
        data_root=data_root,
        csv_path=csv_path,
        y_mode=exp_cfg["data"].get("y_mode", "csv_then_label"),
        n_splits=exp_cfg["data"].get("n_folds", 5),
        test_size=exp_cfg["data"].get("test_size", 0.2),
        val_size=exp_cfg["data"].get("val_size", 0.2),
        min_samples_per_class=exp_cfg["data"].get("min_samples_per_class", 5),
        grit_images_dir=grit_images_dir,
        grit_masks_dir=grit_masks_dir,
    )
    cv_root = Path(paths["cv_dir"])

    # IMPORTANT: Load ALL 8 label bands if detailed metrics are requested
    compute_detailed = exp_cfg.get("evaluation", {}).get("compute_detailed_metrics", False)
    if compute_detailed:
        label_bands = list(range(1, 9))  # Bands 1-8 (1-indexed in rasterio)
        logger.info(f"[cv] Loading all 8 label bands for detailed metrics")
    else:
        label_bands = exp_cfg["data"]["label_bands"]
        logger.info(f"[cv] Loading label bands: {label_bands}")
    
    model_type = exp_cfg["model"]["type"].lower()
    hyperparams = exp_cfg["model"].get("hyperparameters", {}).get(model_type, {})
    
    # Pixel sampling configuration
    pixels_per_image = exp_cfg["data"].get("pixels_per_image", None)
    
    if pixels_per_image:
        logger.info(f"[cv] Using pixel sampling: {pixels_per_image} pixels per image")
    else:
        logger.info(f"[cv] Using ALL pixels (loading directly into RAM)")

    cv_manifest_path = Path(paths["cv_manifest_csv"])
    if not cv_manifest_path.exists():
        raise RuntimeError(f"CV manifest CSV not found: {cv_manifest_path}")
    
    manifest = pd.read_csv(cv_manifest_path)
    logger.info(f"[cv] Loaded manifest with {len(manifest)} entries")

    fold_dirs = sorted((cv_root / "train").glob("fold_*"), key=lambda p: p.name)
    results = []
    
    logger.info(f"[cv] Running {len(fold_dirs)} folds...")
    
    # Get band names for feature importance
    image_bands = exp_cfg["data"].get("image_bands", None)
    if not image_bands:
        # Try to import default band names
        try:
            from src.modeling.custom_dataset import SHORT_BAND_NAMES
            BAND_NAMES = SHORT_BAND_NAMES
        except Exception:
            BAND_NAMES = [f"Band{i+1}" for i in range(14)]  # Default fallback
    else:
        BAND_NAMES = image_bands
    
    for fold_idx, fold_dir in enumerate(fold_dirs, start=1):
        tr_txt = fold_dir / "train_files.txt"
        va_txt = fold_dir / "val_files.txt"
        if not tr_txt.exists() or not va_txt.exists():
            logger.warning(f"[skip] Missing lists in {fold_dir}")
            continue

        train_stems = load_stems(str(tr_txt))
        val_stems = load_stems(str(va_txt))
        logger.info(f"[{fold_dir.name}] train={len(train_stems)}, val={len(val_stems)}")

        # Create fold-specific output directory
        fold_output_dir = os.path.join(experiment_dir, fold_dir.name)
        os.makedirs(fold_output_dir, exist_ok=True)

        clf, metrics, val_ds, train_size, val_size = train_and_evaluate_fold(
            train_stems, val_stems, manifest, label_bands, model_type, hyperparams, 
            fold_name=fold_dir.name, pixels_per_image=pixels_per_image,
            exp_cfg=exp_cfg, fold_dir=fold_output_dir
        )

        # Save fold model
        model_path = os.path.join(fold_output_dir, "model.pkl")
        dump(clf, model_path)
        logger.info(f"[{fold_dir.name}] Model saved to {model_path}")

        results.append({
            "fold": fold_dir.name,
            "metrics": metrics,
            "train_size": train_size,
            "val_size": val_size,
        })
        logger.info(f"[{fold_dir.name}] Metrics:\n{json.dumps(metrics, indent=2)}")
        
        # Save visualization for each fold
        num_samples = exp_cfg.get("visualization", {}).get("num_samples", 2)
        vis_path = os.path.join(fold_output_dir, f"visualization_{fold_dir.name}.png")
        plot_predictions(val_ds, clf, num_samples=num_samples, save_path=vis_path)
        
        # --- Feature importance export for this fold ---
        save_feat_imp = exp_cfg.get("model", {}).get("save_feature_importance", False)
        if save_feat_imp and hasattr(clf, "estimators_") or hasattr(clf, "feature_importances_"):
            try:
                # Infer number of timesteps from first sample
                first_img, _, _ = val_ds[0]
                N_TIMESTEPS = first_img.shape[0] // len(BAND_NAMES)  # Assuming bands × timesteps structure
                
                logger.info(f"[{fold_dir.name}] Exporting feature importances...")
                
                # Create subfolders for CSVs and plots
                fi_root_dir = os.path.join(fold_output_dir, "feature_importance")
                fi_csv_dir = os.path.join(fi_root_dir, "csv")
                fi_plot_dir = os.path.join(fi_root_dir, "plots")
                os.makedirs(fi_csv_dir, exist_ok=True)
                os.makedirs(fi_plot_dir, exist_ok=True)
                
                # Export feature importances to CSV
                export_feature_importances(clf, BAND_NAMES, N_TIMESTEPS, fi_csv_dir)
                
                # Move any legacy CSVs
                legacy_csvs = glob.glob(os.path.join(fold_output_dir, "feature_importance*.csv"))
                for src in legacy_csvs:
                    dst = os.path.join(fi_csv_dir, os.path.basename(src))
                    try:
                        if os.path.abspath(src) != os.path.abspath(dst):
                            shutil.move(src, dst)
                    except Exception as e:
                        logger.warning(f"[{fold_dir.name}] Could not move CSV {src}: {e}")
                
                # Plot feature importances
                band_csv = os.path.join(fi_csv_dir, "feature_importance_by_band.csv")
                time_csv = os.path.join(fi_csv_dir, "feature_importance_by_time.csv")
                band_time_csv = os.path.join(fi_csv_dir, "feature_importance_detailed.csv")
                
                if os.path.exists(band_csv):
                    png_path = os.path.join(fi_plot_dir, f"band_importance_{model_type}.png")
                    try:
                        plot_band_importance(band_csv, band_names=BAND_NAMES, save_path=png_path)
                        logger.info(f"[{fold_dir.name}] Plotted band importance to {png_path}")
                    except Exception as e:
                        logger.warning(f"[{fold_dir.name}] Failed to plot band importance: {e}")
                
                if os.path.exists(time_csv):
                    png_path = os.path.join(fi_plot_dir, f"time_importance_{model_type}_T{N_TIMESTEPS}.png")
                    try:
                        plot_time_importance(time_csv, num_timesteps=N_TIMESTEPS, save_path=png_path)
                        logger.info(f"[{fold_dir.name}] Plotted time importance to {png_path}")
                    except Exception as e:
                        logger.warning(f"[{fold_dir.name}] Failed to plot time importance: {e}")
                
                if os.path.exists(band_time_csv):
                    png_path = os.path.join(fi_plot_dir, f"band_time_importance_{model_type}_T{N_TIMESTEPS}.png")
                    try:
                        plot_band_time_importance(
                            importance_df=band_time_csv,
                            band_names=BAND_NAMES,
                            num_timesteps=N_TIMESTEPS,
                            save_path=png_path
                        )
                        logger.info(f"[{fold_dir.name}] Plotted band-time heatmap to {png_path}")
                    except Exception as e:
                        logger.warning(f"[{fold_dir.name}] Failed to plot band-time heatmap: {e}")
                
                logger.info(f"[{fold_dir.name}] Feature importance CSVs saved under: {fi_csv_dir}")
                logger.info(f"[{fold_dir.name}] Feature importance plots saved under: {fi_plot_dir}")
                
            except Exception as e:
                import traceback
                logger.warning(f"[{fold_dir.name}] Failed to export feature importances: {e}")
                traceback.print_exc()
        elif save_feat_imp:
            logger.warning(f"[{fold_dir.name}] Requested feature importances, but model does not support feature importance extraction")

    # Aggregate CV metrics
    if results:
        logger.info("[cv] Aggregating results...")
        metric_structure = results[0]["metrics"]
        summary = {"n_folds_completed": len(results), "fold_details": results}
        
        for mtype in metric_structure:
            means, stds = {}, {}
            for mname in metric_structure[mtype]:
                vals = [r["metrics"][mtype][mname] for r in results]
                means[mname] = float(np.mean(vals))
                stds[mname] = float(np.std(vals))
            summary[f"{mtype}_mean"] = means
            summary[f"{mtype}_std"] = stds

        out_json = Path(experiment_dir) / "cv_results.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"[cv] Results saved to {out_json}")
        
        logger.info("[cv] Cross-validation summary:")
        for mtype in metric_structure:
            logger.info(f"  {mtype}_mean: {summary[f'{mtype}_mean']}")
            logger.info(f"  {mtype}_std: {summary[f'{mtype}_std']}")
    else:
        logger.error("[cv] No folds completed successfully.")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Irrigation ML runner (CV with site-aware splits + direct loading + comprehensive metrics).")
    ap.add_argument("--config", type=str, default="src/modeling/experiment.yaml",
                    help="Path to the experiment YAML config file")
    args = ap.parse_args()

    exp_cfg = load_experiment(args.config)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{exp_cfg['name']}_{timestamp}"
    out_root = resolve_path(exp_cfg["output"]["base_dir"])
    run_dir = os.path.join(out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    cfg_path = resolve_config_path(args.config)
    shutil.copyfile(cfg_path, os.path.join(run_dir, "experiment.yaml"))
    
    fh = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    logger.info("="*80)
    logger.info(f"Starting experiment: {exp_cfg['name']}")
    logger.info(f"Output directory: {run_dir}")
    logger.info(f"Configuration: {cfg_path}")
    logger.info("="*80)

    try:
        run_cv_experiment(exp_cfg, run_dir)
        
        logger.info("="*80)
        logger.info("[SUCCESS] Experiment completed successfully")
        logger.info(f"Results saved to: {run_dir}")
        logger.info("="*80)
    except Exception as e:
        logger.error("="*80)
        logger.error(f"[FAILED] Experiment failed with error: {e}")
        logger.error("="*80)
        raise
    finally:
        logger.removeHandler(fh)
        fh.close()


if __name__ == "__main__":
    main()