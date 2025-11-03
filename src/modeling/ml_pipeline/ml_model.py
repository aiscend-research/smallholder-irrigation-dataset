#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model training and evaluation utilities for irrigation classification.
Includes RandomForest, GradientBoosting, and K-fold evaluation.
"""

import os
import numpy as np
import logging
from pathlib import Path

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)
from imblearn.over_sampling import SMOTE

from src.modeling.ml_pipeline.evaluation import (
    model_metrics,
    metrics_over_factors,
    plot_metrics_over_factors,
)
from src.modeling.ml_pipeline.sampling import downsample_majority_class
from src.modeling.custom_dataset import (
    load_dataset_from_manifest,
    flatten_dataset_from_tuples,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

# ----------------------------------------------------------------------
# Model wrappers
# ----------------------------------------------------------------------
def _filter_params(estimator_cls, params: dict) -> dict:
    """Keep only kwargs that the estimator actually supports."""
    valid = estimator_cls().get_params().keys()
    return {k: v for k, v in params.items() if k in valid}


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42, **kwargs):
    defaults = dict(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf_params = {**defaults, **kwargs}
    rf_params = _filter_params(RandomForestClassifier, rf_params)
    clf = RandomForestClassifier(**rf_params)
    clf.fit(X_train, y_train)
    return clf


def train_gradient_boosting(
    X_train,
    y_train,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0,
    random_state=42,
    **kwargs,
):
    defaults = dict(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=random_state,
    )
    gb_params = {**defaults, **kwargs}
    gb_params = _filter_params(GradientBoostingClassifier, gb_params)
    clf = GradientBoostingClassifier(**gb_params)
    clf.fit(X_train, y_train)
    return clf


def train_model(X_train, y_train, model_type="random_forest", **hyperparams):
    """General model selector."""
    if model_type == "random_forest":
        clf = train_random_forest(X_train, y_train, **hyperparams)
    elif model_type == "gradient_boosting":
        clf = train_gradient_boosting(X_train, y_train, **hyperparams)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return clf

def tune_random_forest(X_train, y_train, param_grid, metric="f1", search="grid", n_iter=10, random_state=42):
    base_model = RandomForestClassifier(
        class_weight="balanced", n_jobs=-1, random_state=random_state
    )
    if search == "grid":
        searcher = GridSearchCV(
            base_model,
            param_grid=param_grid,
            scoring=metric,
            cv=1,
            verbose=2,
            n_jobs=-1,
        )
    else:
        searcher = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            scoring=metric,
            cv=1,
            n_iter=n_iter,
            verbose=2,
            n_jobs=-1,
            random_state=random_state,
        )
    searcher.fit(X_train, y_train)
    best_model = searcher.best_estimator_
    logger.info(f"Best RF params ({metric}): {searcher.best_params_}")
    return best_model

# ----------------------------------------------------------------------
# Fold Training + Evaluation
# ----------------------------------------------------------------------
def train_and_evaluate_fold(
    train_stems,
    val_stems,
    manifest,
    label_bands,
    model_config,
    fold_name="",
    pixels_per_image=None,
    exp_cfg=None,
    fold_dir=None,
):
    """
    Train and evaluate a single fold with comprehensive metrics.
    """

    logger.info(f"[{fold_name}] Loading training data...")
    train_ds = load_dataset_from_manifest(train_stems, manifest, label_bands)
    logger.info(f"[{fold_name}] Loading validation data...")
    val_ds = load_dataset_from_manifest(val_stems, manifest, label_bands)

    logger.info(f"[{fold_name}] Flattening data...")
    X_train, y_train_full, _ = flatten_dataset_from_tuples(
        train_ds, pixels_per_image=pixels_per_image
    )
    X_val, y_val_full, _ = flatten_dataset_from_tuples(
        val_ds, pixels_per_image=pixels_per_image
    )

    # Assume irrigation mask in 2nd band
    y_train = y_train_full[:, 1]
    y_val_for_training = y_val_full[:, 1]
    irrigation_ratio = float((y_train == 1).sum() / len(y_train))
    logger.info(
        f"[{fold_name}] Class distribution: {(y_train == 0).sum()} normal, {(y_train == 1).sum()} irrigation"
    )
    logger.info(f"[{fold_name}] Irrigation ratio: {irrigation_ratio:.4f}")

    # ----------------------------------------------------------------------
    # Handle imbalance: Downsample + SMOTE
    # ----------------------------------------------------------------------
    sampling_cfg = model_config.get("sampling", {})
    smote_cfg = model_config.get("smote", {})

    X_down, y_down = downsample_majority_class(
        X_train,
        y_train,
        target_ratio=sampling_cfg.get("target_ratio", 3.0),
        random_state=smote_cfg.get("random_state", 42),
    )

    if sampling_cfg.get("use_smote", True):
        smote = SMOTE(
            sampling_strategy=smote_cfg.get("sampling_strategy", "auto"),
            k_neighbors=smote_cfg.get("k_neighbors", 5),
            random_state=smote_cfg.get("random_state", 42),
        )
        X_res, y_res = smote.fit_resample(X_down, y_down)
    else:
        X_res, y_res = X_down, y_down

    logger.info(f"[{fold_name}] After sampling: {np.bincount(y_res.astype(int))}")

    # ----------------------------------------------------------------------
    # StandardScaler normalization
    # ----------------------------------------------------------------------
    logger.info(f"[{fold_name}] Applying StandardScaler normalization...")
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_res)
    X_val = scaler.transform(X_val)

    # ----------------------------------------------------------------------
    # Hyperparameter tuning or standard training
    # ----------------------------------------------------------------------
    tuning_cfg = model_config.get("tuning", {})
    if tuning_cfg:
        logger.info(
            f"[{fold_name}] Running hyperparameter tuning for best {tuning_cfg.get('metric', 'f1')}..."
        )
        clf = tune_random_forest(
            X_res, y_res,
            param_grid=tuning_cfg.get("param_grid", {}),
            metric=tuning_cfg.get("metric", "f1"),
            search=tuning_cfg.get("search", "grid"),
            n_iter=tuning_cfg.get("n_iter", 10),
        )
    else:
        clf = train_model(
            X_res, y_res, model_type=model_config.get("type", "random_forest")
        )

    # ----------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------
    logger.info(f"[{fold_name}] Evaluating...")
    y_scores = clf.predict_proba(X_val)[:, 1]

    # Find best threshold by F1
    prec, rec, thr = precision_recall_curve(y_val_for_training, y_scores)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_idx = np.argmax(f1)
    best_thr = thr[best_idx]
    logger.info(f"[{fold_name}] Best F1 threshold: {best_thr:.3f}")

    y_pred = (y_scores > best_thr).astype(int)

    # ----------------------------------------------------------------------
    # Compute metrics
    # ----------------------------------------------------------------------
    metrics = model_metrics(y_pred, y_val_for_training)
    pr_auc = float(average_precision_score(y_val_for_training, y_scores))
    roc_auc = float(roc_auc_score(y_val_for_training, y_scores))
    mcc = float(matthews_corrcoef(y_val_for_training, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_val_for_training, y_pred))
    cm = confusion_matrix(y_val_for_training, y_pred, labels=[0, 1]).tolist()

    for task_key, vals in metrics.items():
        vals.update(
            {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "mcc": mcc,
                "balanced_accuracy": balanced_acc,
                "irrigation_ratio": irrigation_ratio,
                "confusion_matrix": {"labels": [0, 1], "matrix": cm},
            }
        )

    # ----------------------------------------------------------------------
    # Detailed metrics (optional)
    # ----------------------------------------------------------------------
    compute_detailed = (
        exp_cfg and exp_cfg.get("evaluation", {}).get("compute_detailed_metrics", False)
    )
    if compute_detailed:
        try:
            logger.info(f"[{fold_name}] Computing detailed metrics...")
            n_imgs = len(val_ds)
            first_img, first_label, _ = val_ds[0]
            H, W = first_img.shape[1], first_img.shape[2]

            if y_val_full.shape[1] < 8:
                logger.warning(
                    f"[{fold_name}] Not enough label bands ({y_val_full.shape[1]}) for detailed metrics."
                )
                return clf, metrics, val_ds, len(train_ds), len(val_ds)

            y_pred_spatial = y_pred.reshape(n_imgs, H, W)
            y_test_spatial = y_val_for_training.reshape(n_imgs, H, W)
            label_metadata = (
                y_val_full[:, 2:8]
                .reshape(n_imgs, H, W, 6)
                .transpose(0, 3, 1, 2)
                .astype(int)
            )
            ids = np.array([int(stem.split("_")[0]) for stem in val_stems], dtype=int)

            if fold_dir:
                detailed_dir = os.path.join(fold_dir, "detailed_metrics")
                plots_dir = os.path.join(detailed_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                detailed_metrics = metrics_over_factors(
                    y_pred=y_pred_spatial,
                    y_test=y_test_spatial,
                    multi_class=False,
                    label_metadata=label_metadata,
                    ids=ids,
                    metrics_path=detailed_dir,
                )
                plot_metrics_over_factors(
                    metrics_json=detailed_metrics, save_dir=plots_dir
                )
                logger.info(
                    f"[{fold_name}] Saved detailed metrics plots to: {plots_dir}"
                )
        except Exception as e:
            logger.warning(f"[{fold_name}] Failed detailed metrics: {e}")

    return clf, metrics, val_ds, len(train_ds), len(val_ds)