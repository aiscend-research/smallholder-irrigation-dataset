#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script:
1. Builds spatially safe splits (siteNumeric grouping; optional stratification)
   by calling prepare_and_export_splits() from data_splitting.py.
2. Consumes the produced *.txt lists (no file movement of the whole dataset).
3. Runs either a one-shot train/val experiment or K-fold CV over the training
   pool, with a held-out test list also produced for CV.
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)
ROOT = project_root

from src.modeling.ml_pipeline.ml_model import train_model
from src.modeling.ml_pipeline.evaluation import model_metrics
from src.modeling.ml_pipeline.evaluation import export_feature_importances
from src.modeling.ml_pipeline.evaluation import plot_band_time_importance
from src.modeling.ml_pipeline.visualization import plot_ml_predictions
from src.modeling.ml_pipeline.build_features import flatten_dataset
from src.modeling.custom_dataset import MultiTemporalCropDataset
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
    with open(cfg_path, "r", encoding="utf-8") as f:   # ← use resolved absolute path
        return yaml.safe_load(f)


def load_stems(txt_path: str) -> list[str]:
    """Read file stems from a text file (one per line)."""
    p = Path(txt_path)
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


# Dataset helper
def create_filtered_dataset(source_dir: str, stems: list[str], label_bands: list[int], manifest_df: pd.DataFrame | None = None):
    """
    Stage only the requested files into a temporary directory so that
    MultiTemporalCropDataset can point to a single folder. This copies a small
    subset instead of moving the whole dataset.

    If you later adapt MultiTemporalCropDataset to accept explicit file paths,
    you can remove this staging and read directly from manifest.csv.
    """
    import tempfile

    src = Path(source_dir)
    temp_dir = Path(tempfile.mkdtemp(prefix="filtered_data_"))
    logger.info(f"[staging] {len(stems)} stems -> {temp_dir}")

    # If a manifest is provided (from data_splitting), prefer its absolute paths (supports GRIT)
    manifest_index = None
    if manifest_df is not None and not manifest_df.empty:
        # --- NEW: normalize keys to plain strings without whitespace
        mf = manifest_df.copy()
        mf["stem"] = mf["stem"].astype(str).str.strip()
        manifest_index = mf.set_index("stem")
    else:
        logger.info("[staging] No manifest provided or it is empty; will try local organized/* fallback.")

    copied = 0
    manifest_hits = 0
    manifest_misses = []

    for s in stems:
        used_manifest = False
        if manifest_index is not None and s in manifest_index.index:
            used_manifest = True
            row = manifest_index.loc[s]
            img_path = Path(str(row["image_path"])).expanduser()
            lab_path = Path(str(row["label_path"])).expanduser()
            jsn_path = Path(str(row["json_path"])).expanduser() if "json_path" in row and pd.notna(row["json_path"]) else None

            # --- IMPORTANT: JSON is optional. We only require image+label to exist.
            missing = []
            if not img_path.exists(): missing.append(str(img_path))
            if not lab_path.exists(): missing.append(str(lab_path))

            if missing:
                manifest_misses.append((s, missing))
            else:
                try:
                    shutil.copy2(img_path, temp_dir / img_path.name)
                    shutil.copy2(lab_path, temp_dir / lab_path.name)
                    if jsn_path is not None and jsn_path.exists():
                        shutil.copy2(jsn_path, temp_dir / jsn_path.name)
                except Exception as e:
                    logger.warning(f"[staging] copy failed for stem '{s}': {e}")
                else:
                    copied += 1
                    manifest_hits += 1
                    continue  # done with this stem

        # Local organized fallback (unchanged behavior)
        # Only reached if: (a) no manifest, (b) stem not present in manifest, or (c) manifest entry missing files.
        img = src / "organized" / "images" / f"{s}.tif"
        lab = src / "organized" / "labels" / f"{s.replace('_image', '_label')}.tif"
        jsn_label = src / "organized" / "metadata" / f"{s.replace('_image', '_label')}.json"
        jsn_image = src / "organized" / "metadata" / f"{s}.json"
        jsn = jsn_label if jsn_label.exists() else jsn_image

        if img.exists() and lab.exists():
            try:
                shutil.copy2(img, temp_dir / img.name)
                shutil.copy2(lab, temp_dir / lab.name)
                if jsn is not None and Path(jsn).exists():
                    shutil.copy2(jsn, temp_dir / jsn.name)
            except Exception as e:
                logger.warning(f"[staging] local fallback copy failed for '{s}': {e}")
            else:
                copied += 1
        else:
            # Only log detailed misses sparsely to avoid spam
            if used_manifest:
                # We already recorded the miss above with reasons
                pass
            else:
                manifest_misses.append((s, [str(img), str(lab)]))

    # --- NEW: diagnostics
    if manifest_index is not None:
        logger.info(f"[staging] manifest matches copied: {manifest_hits} / requested: {len(stems)}")
    if manifest_misses:
        sample = "\n  ".join([f"{stem} :: missing {paths}" for stem, paths in manifest_misses[:10]])
        logger.warning(f"[staging] {len(manifest_misses)} stems could not be staged. First few:\n  {sample}")

    if copied == 0:
        # Fail fast with an actionable message (avoid cryptic error later in flatten_dataset)
        raise RuntimeError(
            "No valid (image,label) pairs were staged for the requested stems.\n"
            "Common causes:\n"
            "  • CV manifest paths are not visible on this node/container (check mounts/permissions),\n"
            "  • Stems in the fold do not appear in manifest.csv (mismatch),\n"
            "  • Files exist but only JSON is missing (should be OK) or filenames changed.\n"
            f"Temp dir: {temp_dir}\n"
            "Hints:\n"
            "  - Open the CV manifest listed in the logs and verify a stem from this fold has image_path/label_path that exist.\n"
            "  - Ensure the GRIT directories are mounted inside this Singularity session."
        )

    ds = MultiTemporalCropDataset(
        image_dir=str(temp_dir),
        label_dir=str(temp_dir),
        label_bands=label_bands,
    )
    return ds, str(temp_dir)


# Core runners
def run_single_experiment(exp_cfg: dict, experiment_dir: str):
    """One-shot train/val using the lists written by data_splitting."""
    data_root = resolve_path(exp_cfg["data"]["data_dir"])
    csv_path = exp_cfg["data"].get("csv_path")
    csv_path = resolve_path(csv_path) if csv_path else None

    # GRIT paths (optional). If provided, data_splitting will scan cloud folders and write cloud-absolute paths in manifest.csv
    grit_images_dir = resolve_path(exp_cfg["data"].get("grit_images_dir")) if exp_cfg["data"].get("grit_images_dir") else None
    grit_masks_dir  = resolve_path(exp_cfg["data"].get("grit_masks_dir"))  if exp_cfg["data"].get("grit_masks_dir")  else None

    # Build splits and get paths
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

    # Read lists
    train_stems = load_stems(paths["train_list"])
    val_stems = load_stems(paths["val_list"])
    manifest = pd.read_csv(paths["manifest_csv"]) if Path(paths["manifest_csv"]).exists() else None

    logger.info(f"[splits] train={len(train_stems)}, val={len(val_stems)}")

    # Build datasets
    label_bands = exp_cfg["data"]["label_bands"]
    train_ds, tmp_train = create_filtered_dataset(data_root, train_stems, label_bands, manifest_df=manifest)
    val_ds, tmp_val = create_filtered_dataset(data_root, val_stems, label_bands, manifest_df=manifest) if val_stems else (None, None)

    try:
        # --- NEW: guard against empty dataset
        if len(train_ds) == 0:
            raise RuntimeError("Training dataset has 0 items after staging; cannot proceed.")
        X_train, y_train = flatten_dataset(train_ds)

        if val_ds is not None and len(val_ds) > 0:
            X_val, y_val = flatten_dataset(val_ds)
        else:
            X_val, y_val = None, None

        y_train = y_train[:, :2]
        if y_val is not None:
            y_val = y_val[:, :2]

        # Train
        model_type = exp_cfg["model"]["type"].lower()
        hyperparams = exp_cfg["model"].get("hyperparameters", {}).get(model_type, {})
        logger.info(f"[train] model={model_type} hyperparams={hyperparams}")
        clf = train_model(X_train, y_train, model_type, **hyperparams)

        from joblib import dump
        model_path = os.path.join(experiment_dir, "model.pkl")
        dump(clf, model_path)
        logger.info(f"[save] model -> {model_path}")

        # Evaluate & visualize
        if X_val is not None:
            y_pred = clf.predict(X_val)
            metrics = model_metrics(y_pred, y_val)
            with open(os.path.join(experiment_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"[metrics]\n{json.dumps(metrics, indent=2)}")

            num_samples = exp_cfg.get("visualization", {}).get("num_samples", 2)
            vis_path = os.path.join(experiment_dir, "visualization.png")
            plot_ml_predictions(val_ds, clf, num_samples=num_samples, save_path=vis_path)

            save_fi = exp_cfg.get("model", {}).get("save_feature_importance", False)
            if save_fi and hasattr(clf, "estimators_"):
                BAND_NAMES = ["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12","NDVI","EVI","NDWI","SCL"]
                N_TIMESTEPS = 37
                fi_csv = os.path.join(experiment_dir, "feature_importance.csv")
                export_feature_importances(clf, BAND_NAMES, N_TIMESTEPS, fi_csv)
                fi_png = os.path.join(experiment_dir, "band_time_importance.png")
                plot_band_time_importance(fi_csv, band_names=BAND_NAMES, n_timesteps=N_TIMESTEPS, save_path=fi_png)
    finally:
        # Clean staging dirs
        for d in [tmp_train, tmp_val]:
            if d:
                shutil.rmtree(d, ignore_errors=True)


def run_cv_experiment(exp_cfg: dict, experiment_dir: str):
    """K-fold CV inside the training pool, with a held-out test list available."""
    data_root = resolve_path(exp_cfg["data"]["data_dir"])
    csv_path = exp_cfg["data"].get("csv_path")
    csv_path = resolve_path(csv_path) if csv_path else None

    # GRIT paths (optional)
    grit_images_dir = resolve_path(exp_cfg["data"].get("grit_images_dir")) if exp_cfg["data"].get("grit_images_dir") else None
    grit_masks_dir  = resolve_path(exp_cfg["data"].get("grit_masks_dir"))  if exp_cfg["data"].get("grit_masks_dir")  else None

    # Build lists
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

    # Load CV manifest (contains cloud-absolute paths in GRIT mode)
    cv_manifest = Path(paths["cv_manifest_csv"])
    manifest = pd.read_csv(cv_manifest) if cv_manifest.exists() else None
    if manifest is not None:
        logger.info(f"[cv] loaded manifest: {cv_manifest} with {len(manifest)} rows")

    # Iterate folds
    fold_dirs = sorted((cv_root / "train").glob("fold_*"), key=lambda p: p.name)
    results = []
    for fold_dir in fold_dirs:
        tr_txt = fold_dir / "train_files.txt"
        va_txt = fold_dir / "val_files.txt"
        if not tr_txt.exists() or not va_txt.exists():
            logger.warning(f"[skip] missing lists in {fold_dir}")
            continue

        train_stems = load_stems(str(tr_txt))
        val_stems = load_stems(str(va_txt))
        logger.info(f"[fold {fold_dir.name}] train={len(train_stems)} val={len(val_stems)}")

        train_ds, tmp_train = create_filtered_dataset(data_root, train_stems, label_bands, manifest_df=manifest)
        val_ds, tmp_val = create_filtered_dataset(data_root, val_stems, label_bands, manifest_df=manifest)

        try:
            # --- NEW: guard before flattening
            if len(train_ds) == 0:
                raise RuntimeError(f"{fold_dir.name}: training dataset has 0 items after staging; cannot proceed.")
            if len(val_ds) == 0:
                raise RuntimeError(f"{fold_dir.name}: validation dataset has 0 items after staging; cannot proceed.")

            X_train, y_train = flatten_dataset(train_ds)
            X_val, y_val = flatten_dataset(val_ds)

            y_train = y_train[:, :2]
            y_val = y_val[:, :2]

            clf = train_model(X_train, y_train, model_type, **hyperparams)
            y_pred = clf.predict(X_val)
            metrics = model_metrics(y_pred, y_val)

            results.append({
                "fold": fold_dir.name,
                "metrics": metrics,
                "train_size": len(train_ds),
                "val_size": len(val_ds),
            })
            logger.info(f"[{fold_dir.name}] metrics:\n{json.dumps(metrics, indent=2)}")
        finally:
            for d in [tmp_train, tmp_val]:
                if d:
                    shutil.rmtree(d, ignore_errors=True)

    # Aggregate CV metrics
    if results:
        metric_structure = results[0]["metrics"]
        summary = {"n_folds_completed": len(results), "fold_details": results}
        import numpy as np
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
        logger.info(f"[cv] results -> {out_json}")
    else:
        logger.error("[cv] no folds completed.")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Irrigation ML runner (site-aware splits + lists).")
    ap.add_argument("--config", type=str, default="src/modeling/experiment.yaml")
    args = ap.parse_args()

    exp_cfg = load_experiment(args.config)

    # Prepare run folder
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{exp_cfg['name']}_{timestamp}"
    out_root = resolve_path(exp_cfg["output"]["base_dir"])
    run_dir = os.path.join(out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Copy the config for reproducibility
    cfg_path = resolve_config_path(args.config)
    shutil.copyfile(cfg_path, os.path.join(run_dir, "experiment.yaml"))
    fh = logging.FileHandler(os.path.join(run_dir, "run.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

    try:
        if exp_cfg["data"].get("use_cross_validation", False):
            logger.info("[mode] CV")
            run_cv_experiment(exp_cfg, run_dir)
        else:
            logger.info("[mode] one-shot")
            run_single_experiment(exp_cfg, run_dir)
        logger.info("[done]")
    finally:
        logger.removeHandler(fh)
        fh.close()


if __name__ == "__main__":
    main()