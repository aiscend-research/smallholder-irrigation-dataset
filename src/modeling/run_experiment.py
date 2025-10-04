#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script:
1. Builds spatially safe splits (siteNumeric grouping; optional stratification)
   by calling prepare_and_export_splits() from data_splitting.py.
2. Consumes the produced *.txt lists and manifest.csv (no file staging needed).
3. Runs K-fold CV over the training pool with a held-out test list.

Loads data directly into RAM - designed for systems with sufficient memory (64GB+).
"""

import os
import sys
import json
import yaml
import shutil
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
ROOT = project_root

from src.modeling.ml_pipeline.ml_model import train_model
from src.modeling.ml_pipeline.evaluation import model_metrics
from src.modeling.ml_pipeline.evaluation import export_feature_importances
from src.modeling.ml_pipeline.evaluation import plot_band_time_importance
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
    Load image/label data directly from absolute paths in manifest.
    Returns a list of (image_array, label_array) tuples.
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
            
            dataset.append((image, label))
            
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
    Flatten dataset from list of (image, label) tuples with optional pixel sampling.
    Converts spatial image data into feature vectors for ML.
    
    Args:
        dataset: List of (image, label) tuples
        pixels_per_image: Number of pixels to randomly sample per image.
                         If None, uses all pixels.
    """
    X_list = []
    y_list = []
    
    sampling_msg = f"sampling {pixels_per_image} pixels per image" if pixels_per_image else "using ALL pixels"
    logger.info(f"[flatten] Flattening {len(dataset)} samples ({sampling_msg})...")
    
    total_pixels_original = 0
    total_pixels_used = 0
    
    for idx, (image, label) in enumerate(dataset):
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
        
        if (idx + 1) % 100 == 0:
            logger.info(f"[flatten] Processed {idx + 1}/{len(dataset)} samples")
    
    X = np.vstack(X_list)
    y = np.vstack(y_list)
    
    logger.info(f"[flatten] Final shapes: X={X.shape}, y={y.shape}")
    if pixels_per_image and total_pixels_original > total_pixels_used:
        reduction_pct = 100 * (1 - total_pixels_used / total_pixels_original)
        logger.info(f"[flatten] Total pixels: {total_pixels_used:,} / {total_pixels_original:,} ({reduction_pct:.1f}% reduction)")
    
    return X, y


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
            image, label = dataset[sample_idx]
            
            n_bands, height, width = image.shape
            X_sample = image.reshape(n_bands, -1).T
            y_pred = model.predict(X_sample)
            y_pred_img = y_pred.reshape(height, width, -1)
            
            if image.shape[0] >= 3:
                rgb = np.stack([image[2], image[1], image[0]], axis=-1)
                rgb = np.clip(rgb / rgb.max() * 255, 0, 255).astype(np.uint8)
                axes[idx, 0].imshow(rgb)
                axes[idx, 0].set_title(f"Sample {sample_idx}: RGB")
            
            axes[idx, 1].imshow(label[0], cmap='viridis')
            axes[idx, 1].set_title("Ground Truth")
            
            axes[idx, 2].imshow(y_pred_img[:, :, 0], cmap='viridis')
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
                           hyperparams, fold_name="", pixels_per_image=None):
    """
    Train and evaluate a single fold.
    
    Args:
        pixels_per_image: If None, uses all pixels. Otherwise samples this many per image.
    """
    
    logger.info(f"[{fold_name}] Loading training data...")
    train_ds = load_dataset_from_manifest(train_stems, manifest, label_bands)
    
    logger.info(f"[{fold_name}] Loading validation data...")
    val_ds = load_dataset_from_manifest(val_stems, manifest, label_bands)

    logger.info(f"[{fold_name}] Extracting features...")
    X_train, y_train = flatten_dataset_from_tuples(train_ds, pixels_per_image=pixels_per_image)
    X_val, y_val = flatten_dataset_from_tuples(val_ds, pixels_per_image=pixels_per_image)

    # Use only first label band and flatten to 1D (scikit-learn expects 1D targets)
    y_train = y_train[:, 0]  # Shape: (n_samples,)
    y_val = y_val[:, 0]

    logger.info(f"[{fold_name}] Training model...")
    clf = train_model(X_train, y_train, model_type, **hyperparams)
    
    logger.info(f"[{fold_name}] Evaluating...")
    y_pred = clf.predict(X_val)
    metrics = model_metrics(y_pred, y_val)

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
        val_size=0.0,
        min_samples_per_class=exp_cfg["data"].get("min_samples_per_class", 5),
        grit_images_dir=grit_images_dir,
        grit_masks_dir=grit_masks_dir,
    )
    cv_root = Path(paths["cv_dir"])

    label_bands = exp_cfg["data"]["label_bands"]
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
    
    for fold_dir in fold_dirs:
        tr_txt = fold_dir / "train_files.txt"
        va_txt = fold_dir / "val_files.txt"
        if not tr_txt.exists() or not va_txt.exists():
            logger.warning(f"[skip] Missing lists in {fold_dir}")
            continue

        train_stems = load_stems(str(tr_txt))
        val_stems = load_stems(str(va_txt))
        logger.info(f"[{fold_dir.name}] train={len(train_stems)}, val={len(val_stems)}")

        clf, metrics, val_ds, train_size, val_size = train_and_evaluate_fold(
            train_stems, val_stems, manifest, label_bands, model_type, hyperparams, 
            fold_name=fold_dir.name, pixels_per_image=pixels_per_image
        )

        results.append({
            "fold": fold_dir.name,
            "metrics": metrics,
            "train_size": train_size,
            "val_size": val_size,
        })
        logger.info(f"[{fold_dir.name}] Metrics:\n{json.dumps(metrics, indent=2)}")
        
        # Save visualization for first fold only
        if len(results) == 1:
            num_samples = exp_cfg.get("visualization", {}).get("num_samples", 2)
            vis_path = os.path.join(experiment_dir, f"visualization_{fold_dir.name}.png")
            plot_predictions(val_ds, clf, num_samples=num_samples, save_path=vis_path)

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

    ap = argparse.ArgumentParser(description="Irrigation ML runner (CV with site-aware splits + direct loading).")
    ap.add_argument("--config", type=str, default="src/modeling/experiment.yaml",
                    help="Path to experiment configuration YAML file")
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