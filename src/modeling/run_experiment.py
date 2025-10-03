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
import uuid

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


def get_staging_root(exp_cfg: dict, data_root: str, experiment_dir: str) -> Path:
    """
    Decide where to stage files (for datasets that need a single folder).
    We avoid /tmp which can be small on the GRIT host.

    Priority:
      1) env STAGE_ROOT (if set),
      2) exp_cfg['data']['staging_root'] (optional),
      3) <data_root>/organized/.stage  (default; on big storage)
    """
    env_root = os.environ.get("STAGE_ROOT", "").strip()
    if env_root:
        root = Path(env_root)
    else:
        cfg_root = (exp_cfg.get("data", {}) or {}).get("staging_root")
        if cfg_root:
            root = Path(resolve_path(cfg_root))
        else:
            root = Path(data_root) / "organized" / ".stage"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _link_or_copy(src: Path, dst: Path) -> str:
    """
    Try to hard-link first (0 extra space); if it fails (e.g., cross-device),
    fall back to a regular copy. Returns 'link' or 'copy'.
    """
    try:
        # Remove any existing broken/old path first
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.link(src, dst)
        return "link"
    except Exception:
        try:
            shutil.copy2(src, dst)
            return "copy"
        except Exception as e:
            raise e


# Dataset helper
def create_filtered_dataset(
    source_dir: str,
    stems: list[str],
    label_bands: list[int],
    manifest_df: pd.DataFrame | None = None,
    staging_root: Path | None = None,
):
    """
    Stage only the requested files into a temporary directory so that
    MultiTemporalCropDataset can point to a single folder. This now uses
    HARD-LINKS when possible (no extra space). We also avoid /tmp by default.

    If you later adapt MultiTemporalCropDataset to accept explicit file paths,
    you can remove this staging and read directly from manifest.csv.
    """
    src = Path(source_dir)
    if staging_root is None:
        staging_root = Path(source_dir)  # fallback, but we normally pass a real root
    # Unique subdir per call to avoid collisions when running CV
    temp_dir = staging_root / f"filtered_data_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[staging] {len(stems)} stems -> {temp_dir}")

    # If a manifest is provided (from data_splitting), prefer its absolute paths (supports GRIT)
    manifest_index = None
    if manifest_df is not None and not manifest_df.empty:
        manifest_index = manifest_df.set_index("stem")

    copied = 0
    linked = 0
    requested = 0
    matched_pairs = 0

    for s in stems:
        requested += 1
        img_path = None
        lab_path = None
        jsn_path = None

        if manifest_index is not None and s in manifest_index.index:
            row = manifest_index.loc[s]
            # JSON can be empty (""), so guard it
            jp = str(row.get("json_path", "")).strip() if "json_path" in row else ""
            img_path = Path(str(row["image_path"]))
            lab_path = Path(str(row["label_path"]))
            jsn_path = Path(jp) if jp else None
        else:
            # Local organized fallback (unchanged behavior)
            img_path = src / "organized" / "images" / f"{s}.tif"
            lab_path = src / "organized" / "labels" / f"{s.replace('_image', '_label')}.tif"
            jsn_label = src / "organized" / "metadata" / f"{s.replace('_image', '_label')}.json"
            jsn_image = src / "organized" / "metadata" / f"{s}.json"
            jsn_path = jsn_label if jsn_label.exists() else (jsn_image if jsn_image.exists() else None)

        missing = []
        if not (img_path and img_path.exists()):
            missing.append(f"{img_path}")
        if not (lab_path and lab_path.exists()):
            missing.append(f"{lab_path}")
        # JSON is optional → just warn, do not drop the sample
        if jsn_path is not None and not jsn_path.exists():
            logger.warning(f"[staging] json missing for {s}: {jsn_path}")
            jsn_path = None

        if missing:
            logger.warning(f"[staging] missing files for stem '{s}': {missing}")
            continue

        # Try to hard-link first; copy on failure
        for src_path in (img_path, lab_path):
            dst = temp_dir / src_path.name
            try:
                how = _link_or_copy(src_path, dst)
                if how == "link":
                    linked += 1
                else:
                    copied += 1
            except Exception as e:
                logger.warning(f"[staging] copy/link failed for stem '{s}', file '{src_path.name}': {e}")
                break
        else:
            # Only try JSON if present; ignore failures
            if jsn_path is not None:
                dst_json = temp_dir / jsn_path.name
                try:
                    how = _link_or_copy(jsn_path, dst_json)
                    if how == "link":
                        linked += 1
                    else:
                        copied += 1
                except Exception as e:
                    logger.warning(f"[staging] json copy/link failed for stem '{s}': {e}")
            matched_pairs += 1

    logger.info(
        f"[staging] manifest matches staged: {matched_pairs} / requested: {requested}"
    )
    logger.info(f"[staging] summary: linked={linked}, copied={copied}")

    if matched_pairs == 0:
        # Clean dir to avoid clutter
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(
            "No valid (image,label) pairs were staged for the requested stems. "
            "Check that manifest paths exist and that your staging_root is on a filesystem with space."
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
    stage_root = get_staging_root(exp_cfg, data_root, experiment_dir)

    train_ds, tmp_train = create_filtered_dataset(
        data_root, train_stems, label_bands, manifest_df=manifest, staging_root=stage_root
    )
    val_ds, tmp_val = (
        create_filtered_dataset(data_root, val_stems, label_bands, manifest_df=manifest, staging_root=stage_root)
        if val_stems else (None, None)
    )

    try:
        # Feature extraction
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

    # staging root (large disk; hard-link where possible)
    stage_root = get_staging_root(exp_cfg, data_root, experiment_dir)

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

        train_ds, tmp_train = create_filtered_dataset(
            data_root, train_stems, label_bands, manifest_df=manifest, staging_root=stage_root
        )
        val_ds, tmp_val = create_filtered_dataset(
            data_root, val_stems, label_bands, manifest_df=manifest, staging_root=stage_root
        )

        try:
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