#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation utilities for irrigation classification models.
Computes metrics, exports feature importances, and generates plots.
"""

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, mean_squared_error, mean_absolute_error
)
from sklearn.metrics import root_mean_squared_error
import calendar
import json
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
LABEL_CSV = "/home/madhav/smallholder-irrigation-dataset/data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv"
MULTI_CLASSES = ['Not irrigated', 'Small-scale', 'Tree crop', 'Industrial', 'Lawn', 'Covered']
BINARY_CLASSES = ['Not irrigated', 'Irrigated']
UNCERTAINTY_EXPLANATIONS = [
    'Unclear signs of agriculture',
    'Only slightly green',
    'Uneven',
    'May naturally be green',
    'May be a fishpond'
]

# ---------------------------------------------------------------------
# Basic model metrics
# ---------------------------------------------------------------------
def model_metrics(y_pred, y_test):
    """
    Computes accuracy, precision, recall, and F1 score for binary classification.
    """
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    if y_test.ndim > 1:
        y_test = y_test.ravel()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
    rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)

    return {
        "irrigation_presence": {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        }
    }

# ---------------------------------------------------------------------
# Feature importance export and visualization
# ---------------------------------------------------------------------
def export_feature_importances(
    clf,
    band_names=None,
    num_timesteps=None,
    out_dir="./",
    prefix="",
    num_bands=None,
):
    """
    Exports feature importances and saves three CSVs:
      - Detailed (band × timestep)
      - Aggregated by band
      - Aggregated by time_step
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Extract importances
    if hasattr(clf, "estimators_"):
        all_imp = np.array([est.feature_importances_ for est in clf.estimators_])
        importances = all_imp.mean(axis=0)
    else:
        importances = np.asarray(getattr(clf, "feature_importances_", None))
        if importances is None:
            raise ValueError("Provided model does not expose feature_importances_.")

    # --- Resolve band/time structure
    if num_bands is None and band_names is not None:
        num_bands = len(band_names)
    if num_timesteps is None and num_bands and len(importances) % num_bands == 0:
        num_timesteps = len(importances) // num_bands
    if band_names is None and num_bands:
        band_names = [f"B{i+1}" for i in range(num_bands)]

    expected_len = num_bands * num_timesteps
    if len(importances) < expected_len:
        importances = np.concatenate([importances, np.zeros(expected_len - len(importances))])
    elif len(importances) > expected_len:
        importances = importances[:expected_len]

    feature_names = [f"{band_names[b]}_t{t+1}" for t in range(num_timesteps) for b in range(num_bands)]
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    df["band"] = df["feature"].str.extract(r"^(.*?)_t")[0]
    df["time_step"] = df["feature"].str.extract(r"_t(\d+)$")[0].astype(int)

    # --- Aggregations
    agg_band = df.groupby("band", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
    agg_time = df.groupby("time_step", as_index=False)["importance"].sum().sort_values("time_step")

    # --- Save CSVs
    df.to_csv(os.path.join(out_dir, f"{prefix}feature_importance_detailed.csv"), index=False)
    agg_band.to_csv(os.path.join(out_dir, f"{prefix}feature_importance_by_band.csv"), index=False)
    agg_time.to_csv(os.path.join(out_dir, f"{prefix}feature_importance_by_time.csv"), index=False)
    print(f"Saved feature importances to {out_dir}")

    return df, agg_band, agg_time


def plot_band_time_importance(
    importance_df,
    band_names=None,
    num_timesteps=None,
    figsize=(16, 6),
    title="Feature Importance by Band and Time Step",
    save_path=None
):
    """2D heatmap of feature importances by band/time step."""
    if isinstance(importance_df, str):
        importance_df = pd.read_csv(importance_df)

    bands = band_names or sorted(importance_df['band'].unique(), key=str)
    timesteps = sorted(importance_df['time_step'].unique())
    if num_timesteps:
        timesteps = list(range(1, num_timesteps + 1))

    importance_matrix = np.zeros((len(bands), len(timesteps)))
    for i, b in enumerate(bands):
        for j, t in enumerate(timesteps):
            val = importance_df[
                (importance_df["band"] == b) & (importance_df["time_step"] == t)
            ]["importance"]
            importance_matrix[i, j] = val.values[0] if not val.empty else 0

    plt.figure(figsize=figsize)
    im = plt.imshow(importance_matrix, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, label='Importance')
    plt.yticks(range(len(bands)), bands)
    plt.xticks(range(len(timesteps)), [f"t{t}" for t in timesteps], rotation=90)
    plt.xlabel("Time Step")
    plt.ylabel("Band")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved feature importance heatmap to {save_path}")
    else:
        plt.show()


def plot_band_importance(df, band_names=None, title="Feature Importance by Band", save_path=None):
    """Bar chart of feature importances aggregated by band."""
    if isinstance(df, str):
        df = pd.read_csv(df)
    if 'band' not in df.columns or 'importance' not in df.columns:
        raise ValueError("DataFrame must contain 'band' and 'importance' columns.")

    if band_names is not None:
        bands_in_df = set(df['band'])
        missing = set(band_names) - bands_in_df
        if missing:
            filler = pd.DataFrame({'band': list(missing), 'importance': [0] * len(missing)})
            df = pd.concat([df, filler], ignore_index=True)
        df['band'] = pd.Categorical(df['band'], categories=band_names, ordered=True)
        df = df.sort_values('band')
    elif df['band'].duplicated().any():
        df = df.groupby('band', as_index=False)['importance'].sum()

    plt.figure(figsize=(10, 6))
    plt.bar(df['band'], df['importance'], color='skyblue')
    plt.xlabel('Band')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved band importance plot to {save_path}")
    else:
        plt.show()


def plot_time_importance(df, num_timesteps=None, title="Feature Importance by Time Step", save_path=None):
    """Bar chart of feature importances aggregated by time step."""
    if isinstance(df, str):
        df = pd.read_csv(df)
    if 'time_step' not in df.columns or 'importance' not in df.columns:
        raise ValueError("DataFrame must contain 'time_step' and 'importance' columns.")

    if num_timesteps is not None:
        all_steps = pd.DataFrame({'time_step': list(range(1, num_timesteps + 1))})
        df = all_steps.merge(df, on='time_step', how='left').fillna(0)
    elif df['time_step'].duplicated().any():
        df = df.groupby('time_step', as_index=False)['importance'].sum()

    plt.figure(figsize=(12, 6))
    plt.bar(df['time_step'], df['importance'], color='coral')
    plt.xlabel('Time Step')
    plt.ylabel('Importance')
    plt.title(title)
    plt.xticks(df['time_step'])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved time importance plot to {save_path}")
    else:
        plt.show()

# Metrics over metadata factors
def get_image_metadata(ids):
    """Retrieve month/year/water_source for each image ID."""
    irrigation_table = pd.read_csv(LABEL_CSV)
    months = [irrigation_table.loc[irrigation_table['unique_id'] == i, 'month'].values[0] for i in ids]
    years = [irrigation_table.loc[irrigation_table['unique_id'] == i, 'year'].values[0] for i in ids]
    water_sources = [irrigation_table.loc[irrigation_table['unique_id'] == i, 'water_source'].values[0] for i in ids]
    return np.array(months), np.array(years), np.array(water_sources)


def get_metrics(truth, pred, target_names):
    """Compute per-class precision/recall/F1."""
    y_true, y_pred = truth.flatten(), pred.flatten()
    labels = list(range(len(target_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True, zero_division=0)
    results = {}
    for class_name in target_names:
        class_metrics = report[class_name]
        results[class_name.lower().replace(" ", "_").replace("-", "_")] = {
            'f1-score': class_metrics['f1-score'],
            'precision': class_metrics['precision'],
            'recall': class_metrics['recall'],
            'support': class_metrics['support']
        }
    return results


def get_uncertainty_explanation_metrics(label_metadata, y_pred, y_test, target_names):
    """Metrics by uncertainty explanation."""
    metrics = {}
    for i in range(5):
        mask = np.where(label_metadata[:, i, :, :] == 1)
        category_name = UNCERTAINTY_EXPLANATIONS[i].lower().replace(" ", "_").replace("-", "_")
        metrics[category_name] = get_metrics(y_test[mask], y_pred[mask], target_names)
    return metrics


def get_certainty_score_metrics(label_metadata, y_pred, y_test, target_names):
    """Metrics by certainty score bins."""
    metrics = {}
    scores = label_metadata[:, 5, :, :]
    low_mask = np.where(scores <= 3)
    high_mask = np.where(scores > 3)
    metrics['low_certainty'] = get_metrics(y_test[low_mask], y_pred[low_mask], target_names)
    metrics['high_certainty'] = get_metrics(y_test[high_mask], y_pred[high_mask], target_names)
    return metrics


def get_month_metrics(months, y_pred, y_test, target_names):
    """Metrics by month (June–October)."""
    metrics = {}
    for i in range(6, 11):
        mask = np.where(months == i)
        metrics[calendar.month_name[i].lower()] = get_metrics(y_test[mask], y_pred[mask], target_names)
    return metrics


def get_year_metrics(years, y_pred, y_test, target_names):
    """Metrics by year (2016–2025)."""
    metrics = {}
    for year in range(2016, 2026):
        mask = np.where(years == year)
        metrics[str(year)] = get_metrics(y_test[mask], y_pred[mask], target_names)
    return metrics


def get_water_source_metrics(water_sources, y_pred, y_test, target_names):
    """Metrics by water source presence."""
    metrics = {}
    presence_mask = np.where(water_sources == True)
    absence_mask = np.where(water_sources == False)
    metrics['water_source_present'] = get_metrics(y_test[presence_mask], y_pred[presence_mask], target_names)
    metrics['water_source_absent'] = get_metrics(y_test[absence_mask], y_pred[absence_mask], target_names)
    return metrics


def get_class_presence(mask, num_classes=2, presence_thresh=1):
    """Binary class presence vector per image."""
    return np.array([
        int(np.sum(mask == c) >= presence_thresh)
        for c in range(num_classes)
    ])


def compute_presence_metrics(preds, gts, target_names, presence_thresh=1):
    """Multi-label class presence metrics per image."""
    num_classes = len(target_names)
    Y_pred = np.array([get_class_presence(p, num_classes, presence_thresh) for p in preds])
    Y_true = np.array([get_class_presence(g, num_classes, presence_thresh) for g in gts])
    per_class_metrics = {}
    for idx in range(num_classes):
        pred, true = Y_pred[:, idx], Y_true[:, idx]
        category = target_names[idx].lower().replace(" ", "_").replace("-", "_")
        per_class_metrics[category] = {
            "precision": precision_score(true, pred, zero_division=0),
            "recall": recall_score(true, pred, zero_division=0),
            "f1-score": f1_score(true, pred, zero_division=0),
        }
    return per_class_metrics


def metrics_over_factors(y_pred, y_test, multi_class, label_metadata, ids, metrics_path):
    """Compute metrics grouped by uncertainty, month, year, and water source."""
    assert y_pred.shape == y_test.shape
    assert y_pred.shape[0] == label_metadata.shape[0] == len(ids)

    pixel_metrics, image_metrics = {}, {}
    months, years, water_sources = get_image_metadata(ids)
    target_names = MULTI_CLASSES if multi_class else BINARY_CLASSES

    pixel_metrics['overall'] = get_metrics(y_test, y_pred, target_names)
    pixel_metrics['per_uncertainty_explanation'] = get_uncertainty_explanation_metrics(label_metadata, y_pred, y_test, target_names)
    pixel_metrics['per_uncertainty_score'] = get_certainty_score_metrics(label_metadata, y_pred, y_test, target_names)
    pixel_metrics['per_month'] = get_month_metrics(months, y_pred, y_test, target_names)
    pixel_metrics['per_year'] = get_year_metrics(years, y_pred, y_test, target_names)
    pixel_metrics['water_source'] = get_water_source_metrics(water_sources, y_pred, y_test, target_names)

    image_metrics['image_level_class_presence'] = compute_presence_metrics(y_pred, y_test, target_names)

    pred_frac = [np.mean(p > 0) for p in y_pred]
    true_frac = [np.mean(t > 0) for t in y_test]
    image_metrics['image_level_fraction_irrigated'] = {
        'mae': mean_absolute_error(true_frac, pred_frac),
        'rmse': mean_squared_error(true_frac, pred_frac, squared=False),
        'mse': mean_squared_error(true_frac, pred_frac),
    }

    metrics = {'pixel_metrics': pixel_metrics, 'image_metrics': image_metrics}
    os.makedirs(metrics_path, exist_ok=True)
    with open(os.path.join(metrics_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    return metrics


# ---------------------------------------------------------------------
# Plot metrics over factors (clean final version)
# ---------------------------------------------------------------------
def plot_metrics_over_factors(metrics_json, save_dir="plots"):
    """Plots metrics (Precision/Recall/F1) grouped by uncertainty, time, year, and water source."""
    os.makedirs(save_dir, exist_ok=True)
    pixel_metrics = metrics_json["pixel_metrics"]
    image_metrics = metrics_json["image_metrics"]

    irrigation_classes = BINARY_CLASSES
    if len(metrics_json['pixel_metrics']['overall'].keys()) == 6:
        irrigation_classes = MULTI_CLASSES

    def extract_data(section_key, image_level):
        section_data = image_metrics[section_key] if image_level else pixel_metrics[section_key]
        data = []
        if section_key == 'overall' or image_level:
            for irrigation_class in irrigation_classes:
                key = irrigation_class.lower().replace(" ", "_").replace("-", "_")
                metrics = section_data.get(key, {})
                data.append({
                    "category": "",
                    "class": irrigation_class,
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1-score": metrics.get("f1-score", 0.0),
                })
        else:
            for category, category_data in section_data.items():
                for irrigation_class in irrigation_classes:
                    key = irrigation_class.lower().replace(" ", "_").replace("-", "_")
                    metrics = category_data.get(key, {})
                    data.append({
                        "category": category.replace("_", " ").capitalize(),
                        "class": irrigation_class,
                        "precision": metrics.get("precision", 0.0),
                        "recall": metrics.get("recall", 0.0),
                        "f1-score": metrics.get("f1-score", 0.0),
                        "support": metrics.get("support", 0.0)
                    })
        return pd.DataFrame(data)

    def make_plot(df, metric, title, filename):
        if "support" in df.columns:
            df = df[df["support"] > 0]
            if df.empty:
                print(f"Skipping {title} ({metric}) – zero support.")
                return
        plt.figure(figsize=(12, 5))
        ax = plt.subplot()
        pivot_df = df.pivot(index='category', columns='class', values=metric.lower())
        pivot_df.plot(kind='bar', ax=ax, width=0.85)
        plt.title(f"{metric} per {title}")
        plt.ylabel(metric)
        plt.xlabel('Category' if title != 'Overall' else '')
        plt.xticks(rotation=15)
        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{filename}_{metric.lower()}.png"))
        plt.close()

    sections_to_plot = {
        "overall": "Overall",
        "per_uncertainty_explanation": "Uncertainty Explanation",
        "per_uncertainty_score": "Uncertainty Score",
        "per_month": "Month",
        "per_year": "Year",
        "water_source": "Water Source",
        "image_level_class_presence": "Image-Level Class Presence"
    }

    for section_key, title in sections_to_plot.items():
        image_level = section_key == 'image_level_class_presence'
        df = extract_data(section_key, image_level)
        for metric in ['Precision', 'Recall', 'F1-Score']:
            make_plot(df, metric, title, section_key)