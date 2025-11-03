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
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from joblib import dump

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
ROOT = project_root

from src.modeling.custom_dataset import load_dataset_from_manifest, flatten_dataset_from_tuples, plot_predictions
from src.modeling.ml_pipeline.ml_model import train_and_evaluate_fold
from src.modeling.ml_pipeline.evaluation import (
    model_metrics,
    metrics_over_factors,
    plot_metrics_over_factors,
    export_feature_importances,
    plot_band_time_importance,
    plot_band_importance,
    plot_time_importance,
)
from src.modeling.ml_pipeline.data_splitting import prepare_and_export_splits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# Main cross-validation experiment
# -------------------------------------------------------------------------
def run_cv_experiment(exp_cfg: dict, experiment_dir: str):
    """
    Run full cross-validation or single train/validation split depending on YAML.
    """

    data_root = resolve_path(exp_cfg["data"]["data_dir"])
    csv_path = resolve_path(exp_cfg["data"].get("csv_path"))
    grit_images_dir = resolve_path(exp_cfg["data"].get("grit_images_dir"))
    grit_masks_dir = resolve_path(exp_cfg["data"].get("grit_masks_dir"))

    # ----------------------------------------------------------------------
    # Cross-validation or single-split logic
    # ----------------------------------------------------------------------
    use_cv = exp_cfg["data"].get("use_cross_validation", True)
    if use_cv:
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
    else:
        logger.info("[split] Building single train/validation split...")
        paths = prepare_and_export_splits(
            data_root=data_root,
            csv_path=csv_path,
            y_mode=exp_cfg["data"].get("y_mode", "csv_then_label"),
            n_splits=1,
            test_size=exp_cfg["data"].get("test_size", 0.2),
            val_size=exp_cfg["data"].get("val_size", 0.2),
            min_samples_per_class=exp_cfg["data"].get("min_samples_per_class", 5),
            grit_images_dir=grit_images_dir,
            grit_masks_dir=grit_masks_dir,
        )

    # ----------------------------------------------------------------------
    # Load manifest and other experiment configuration
    # ----------------------------------------------------------------------
    cv_root = Path(paths["cv_dir"])
    compute_detailed = exp_cfg.get("evaluation", {}).get("compute_detailed_metrics", False)
    label_bands = list(range(1, 9)) if compute_detailed else exp_cfg["data"]["label_bands"]
    pixels_per_image = exp_cfg["data"].get("pixels_per_image", None)
    manifest = pd.read_csv(Path(paths["cv_manifest_csv"]))

    # If only one split, normalize fold handling
    fold_dirs = sorted((cv_root / "train").glob("fold_*"), key=lambda p: p.name)
    if not use_cv:
        fold_dirs = fold_dirs[:1]  # only one train/val pair for tuning

    results = []
    image_bands = exp_cfg["data"].get("image_bands", None)
    try:
        from src.modeling.custom_dataset import SHORT_BAND_NAMES
        BAND_NAMES = SHORT_BAND_NAMES
    except Exception:
        BAND_NAMES = image_bands or [f"Band{i+1}" for i in range(14)]

    # ----------------------------------------------------------------------
    # Fold Loop
    # ----------------------------------------------------------------------
    for fold_dir in fold_dirs:
        tr_txt = fold_dir / "train_files.txt"
        va_txt = fold_dir / "val_files.txt"
        if not tr_txt.exists() or not va_txt.exists():
            logger.warning(f"[skip] Missing lists in {fold_dir}")
            continue

        train_stems = load_stems(str(tr_txt))
        val_stems = load_stems(str(va_txt))
        fold_output_dir = os.path.join(experiment_dir, fold_dir.name)
        os.makedirs(fold_output_dir, exist_ok=True)

        clf, metrics, val_ds, train_size, val_size = train_and_evaluate_fold(
            train_stems,
            val_stems,
            manifest,
            label_bands,
            exp_cfg["model"],
            fold_name=fold_dir.name,
            pixels_per_image=pixels_per_image,
            exp_cfg=exp_cfg,
            fold_dir=fold_output_dir,
        )

        # Save model
        model_path = os.path.join(fold_output_dir, "model.pkl")
        dump(clf, model_path)
        logger.info(f"[{fold_dir.name}] Model saved to {model_path}")

        results.append(
            {"fold": fold_dir.name, "metrics": metrics, "train_size": train_size, "val_size": val_size}
        )

        # Visualization
        num_samples = exp_cfg.get("visualization", {}).get("num_samples", 2)
        vis_path = os.path.join(fold_output_dir, f"visualization_{fold_dir.name}.png")
        plot_predictions(val_ds, clf, num_samples=num_samples, save_path=vis_path)

        # ------------------------------------------------------------------
        # Feature importance export
        # ------------------------------------------------------------------
        save_feat_imp = exp_cfg.get("model", {}).get("save_feature_importance", False)
        if save_feat_imp and hasattr(clf, "feature_importances_"):
            try:
                first_img, _, _ = val_ds[0]
                N_TIMESTEPS = first_img.shape[0] // len(BAND_NAMES)
                fi_root_dir = os.path.join(fold_output_dir, "feature_importance")
                fi_csv_dir = os.path.join(fi_root_dir, "csv")
                fi_plot_dir = os.path.join(fi_root_dir, "plots")
                os.makedirs(fi_csv_dir, exist_ok=True)
                os.makedirs(fi_plot_dir, exist_ok=True)
                export_feature_importances(clf, BAND_NAMES, N_TIMESTEPS, fi_csv_dir)

                band_csv = os.path.join(fi_csv_dir, "feature_importance_by_band.csv")
                time_csv = os.path.join(fi_csv_dir, "feature_importance_by_time.csv")
                band_time_csv = os.path.join(fi_csv_dir, "feature_importance_detailed.csv")

                if os.path.exists(band_csv):
                    plot_band_importance(
                        band_csv, band_names=BAND_NAMES,
                        save_path=os.path.join(fi_plot_dir, "band_importance.png")
                    )
                if os.path.exists(time_csv):
                    plot_time_importance(
                        time_csv, num_timesteps=N_TIMESTEPS,
                        save_path=os.path.join(fi_plot_dir, "time_importance.png")
                    )
                if os.path.exists(band_time_csv):
                    plot_band_time_importance(
                        importance_df=band_time_csv,
                        band_names=BAND_NAMES,
                        num_timesteps=N_TIMESTEPS,
                        save_path=os.path.join(fi_plot_dir, "band_time_heatmap.png"),
                    )
            except Exception as e:
                logger.warning(f"[{fold_dir.name}] Failed feature importance export: {e}")

        # ------------------------------------------------------------------
        # Detailed metrics (optional)
        # ------------------------------------------------------------------
        if compute_detailed:
            try:
                detailed_dir = os.path.join(fold_output_dir, "detailed_metrics")
                os.makedirs(detailed_dir, exist_ok=True)

                y_pred = np.array([pred for _, pred, _ in val_ds.predictions])
                y_test = np.array([truth for _, _, truth in val_ds.labels])
                label_metadata = np.array([meta for meta in val_ds.metadata])
                ids = np.array([stem for _, _, stem in val_ds.samples])

                metrics_json = metrics_over_factors(
                    y_pred=y_pred,
                    y_test=y_test,
                    multi_class=exp_cfg["model"].get("multi_class", False),
                    label_metadata=label_metadata,
                    ids=ids,
                    metrics_path=detailed_dir,
                )

                plots_dir = os.path.join(detailed_dir, "plots")
                plot_metrics_over_factors(metrics_json, save_dir=plots_dir)
                logger.info(f"[{fold_dir.name}] Detailed metrics and plots saved to {plots_dir}")
            except Exception as e:
                logger.warning(f"[{fold_dir.name}] Failed detailed metrics: {e}")

    # ----------------------------------------------------------------------
    # Aggregate results across folds
    # ----------------------------------------------------------------------
    if results:
        metric_structure = results[0]["metrics"]
        summary = {"n_folds_completed": len(results), "fold_details": results}
        for mtype in metric_structure:
            means, stds = {}, {}
            for mname in metric_structure[mtype]:
                vals = [
                    r["metrics"][mtype][mname]
                    for r in results
                    if not isinstance(r["metrics"][mtype][mname], dict)
                ]
                if vals:
                    means[mname] = float(np.mean(vals))
                    stds[mname] = float(np.std(vals))
            summary[f"{mtype}_mean"] = means
            summary[f"{mtype}_std"] = stds

        out_json = Path(experiment_dir) / "cv_results.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info(f"[cv] Results saved to {out_json}")
    else:
        logger.error("[cv] No folds completed successfully.")


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Irrigation ML runner with SMOTE+RF baseline and full metrics.")
    ap.add_argument(
        "--config",
        type=str,
        default="src/modeling/experiment.yaml",
        help="Path to the experiment YAML config file",
    )
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

    logger.info("=" * 80)
    logger.info(f"Starting experiment: {exp_cfg['name']}")
    logger.info(f"Output directory: {run_dir}")
    logger.info(f"Configuration: {cfg_path}")
    logger.info("=" * 80)

    try:
        run_cv_experiment(exp_cfg, run_dir)
        logger.info("=" * 80)
        logger.info("[SUCCESS] Experiment completed successfully")
        logger.info(f"Results saved to: {run_dir}")
        logger.info("=" * 80)
    except Exception as e:
        logger.error("=" * 80)
        logger.error(f"[FAILED] Experiment failed with error: {e}")
        logger.error("=" * 80)
        raise
    finally:
        logger.removeHandler(fh)
        fh.close()


if __name__ == "__main__":
    main()