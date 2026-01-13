"""
Inter-rater comparison tools for irrigation labels.

This module provides reusable functions to compare a ground truth labeler
against one or more comparison labelers, including:
- Image matching across labelers (with date tolerance)
- Visualization of polygon overlays
- Image-level irrigation detection metrics (TP, FP, FN, TN)
- Area overlap metrics (IoU, precision, recall, F1)
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings


# ============================================================================
# DATA LOADING AND FILTERING
# ============================================================================

def load_comparison_data(
    irrigation_table_path: str,
    polygons_path: str,
    image_boundaries_path: str
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load irrigation data from standard file paths.

    Parameters
    ----------
    irrigation_table_path : str
        Path to latest_irrigation_table.csv
    polygons_path : str
        Path to latest_polygons.geojson
    image_boundaries_path : str
        Path to latest_irrigation_data.geojson

    Returns
    -------
    df : pd.DataFrame
        Irrigation table with all image labels
    gdf_polygons : gpd.GeoDataFrame
        All labeled polygons
    gdf_images : gpd.GeoDataFrame
        Image boundary rectangles
    """
    df = pd.read_csv(irrigation_table_path)
    gdf_polygons = gpd.read_file(polygons_path)
    gdf_images = gpd.read_file(image_boundaries_path)

    return df, gdf_polygons, gdf_images


def filter_by_operators(
    df: pd.DataFrame,
    gdf_polygons: gpd.GeoDataFrame,
    gdf_images: gpd.GeoDataFrame,
    operators: List[str]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, gpd.GeoDataFrame], Dict[str, gpd.GeoDataFrame]]:
    """
    Filter data by operator initials.

    Parameters
    ----------
    df : pd.DataFrame
        Full irrigation table
    gdf_polygons : gpd.GeoDataFrame
        All polygons
    gdf_images : gpd.GeoDataFrame
        All image boundaries
    operators : List[str]
        List of operator initials (e.g., ['AB', 'PS', 'JL'])

    Returns
    -------
    df_dict : Dict[str, pd.DataFrame]
        Dictionary mapping operator -> filtered dataframe
    gdf_poly_dict : Dict[str, gpd.GeoDataFrame]
        Dictionary mapping operator -> filtered polygon geodataframe
    gdf_img_dict : Dict[str, gpd.GeoDataFrame]
        Dictionary mapping operator -> filtered image boundary geodataframe
    """
    df_dict = {}
    gdf_poly_dict = {}
    gdf_img_dict = {}

    for op in operators:
        df_dict[op] = df[df['operator_initials'] == op].copy()
        gdf_poly_dict[op] = gdf_polygons[gdf_polygons['operator_initials'] == op].copy()
        gdf_img_dict[op] = gdf_images[gdf_images['operator_initials'] == op].copy()

    return df_dict, gdf_poly_dict, gdf_img_dict


def filter_by_certainty(
    gdf_polygons: gpd.GeoDataFrame,
    min_certainty: int = 3
) -> gpd.GeoDataFrame:
    """
    Filter polygons by minimum certainty level.

    Parameters
    ----------
    gdf_polygons : gpd.GeoDataFrame
        Polygon geodataframe with 'certainty' column
    min_certainty : int, default=3
        Minimum certainty level to include

    Returns
    -------
    gpd.GeoDataFrame
        Filtered polygons
    """
    if 'certainty' not in gdf_polygons.columns:
        warnings.warn("No 'certainty' column found in polygons. Returning all polygons.")
        return gdf_polygons

    return gdf_polygons[gdf_polygons['certainty'] >= min_certainty].copy()


# ============================================================================
# IMAGE MATCHING
# ============================================================================

def match_images_across_operators(
    df_dict: Dict[str, pd.DataFrame],
    gt_operator: str,
    comparison_operators: List[str],
    date_tolerance_days: int = 1
) -> Dict[str, pd.DataFrame]:
    """
    Match images between GT operator and each comparison operator separately.

    Parameters
    ----------
    df_dict : Dict[str, pd.DataFrame]
        Dictionary mapping operator -> dataframe
    gt_operator : str
        Ground truth operator initials
    comparison_operators : List[str]
        List of comparison operator initials
    date_tolerance_days : int, default=1
        Allow matching images within ±N days

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping comparison operator -> matched images DataFrame
        Each DataFrame has columns:
        - site_id
        - gt_date
        - comp_date
        - date_diff
    """
    df_gt = df_dict[gt_operator].copy()
    df_gt['date'] = pd.to_datetime(df_gt[['year', 'month', 'day']])
    gt_images = df_gt[['site_id', 'date']].drop_duplicates()

    matches_dict = {}

    # Match GT against each comparison operator separately
    for comp_op in comparison_operators:
        matches = []
        df_comp = df_dict[comp_op].copy()
        df_comp['date'] = pd.to_datetime(df_comp[['year', 'month', 'day']])
        comp_images = df_comp[['site_id', 'date']].drop_duplicates()

        for _, gt_row in gt_images.iterrows():
            site = gt_row['site_id']
            gt_date = gt_row['date']

            # Find best match within tolerance at this site
            comp_at_site = comp_images[comp_images['site_id'] == site]
            best_match = None
            best_diff = float('inf')

            for _, comp_row in comp_at_site.iterrows():
                comp_date = comp_row['date']
                date_diff = abs((gt_date - comp_date).days)

                if date_diff <= date_tolerance_days and date_diff < best_diff:
                    best_match = comp_date
                    best_diff = date_diff

            if best_match is not None:
                matches.append({
                    'site_id': site,
                    'gt_date': gt_date,
                    'comp_date': best_match,
                    'date_diff': best_diff
                })

        matches_dict[comp_op] = pd.DataFrame(matches)

    return matches_dict


# ============================================================================
# POLYGON EXTRACTION HELPERS
# ============================================================================

def get_polygons_for_image(
    gdf: gpd.GeoDataFrame,
    site_id: str,
    date: datetime
) -> gpd.GeoDataFrame:
    """Extract polygons for a specific site_id and date."""
    mask = (
        (gdf['site_id'] == site_id) &
        (gdf['year'] == date.year) &
        (gdf['month'] == date.month) &
        (gdf['day'] == date.day)
    )
    return gdf[mask].copy()


def get_image_boundary(
    gdf_images: gpd.GeoDataFrame,
    site_id: str,
    date: datetime
):
    """Get the image boundary rectangle for a specific site_id and date."""
    mask = (
        (gdf_images['site_id'] == site_id) &
        (gdf_images['year'] == date.year) &
        (gdf_images['month'] == date.month) &
        (gdf_images['day'] == date.day)
    )
    result = gdf_images[mask]
    if len(result) > 0:
        return result.iloc[0].geometry
    return None


def get_internal_id(
    gdf_images: gpd.GeoDataFrame,
    site_id: str,
    date: datetime
) -> Optional[str]:
    """Get the internal_id for a specific site_id and date."""
    mask = (
        (gdf_images['site_id'] == site_id) &
        (gdf_images['year'] == date.year) &
        (gdf_images['month'] == date.month) &
        (gdf_images['day'] == date.day)
    )
    result = gdf_images[mask]
    if len(result) > 0:
        return result.iloc[0]['internal_id']
    return None


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comparison(
    site_id: str,
    gt_date: datetime,
    comparison_dates: Dict[str, datetime],
    gt_operator: str,
    gdf_poly_dict: Dict[str, gpd.GeoDataFrame],
    gdf_img_dict: Dict[str, gpd.GeoDataFrame],
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot ground truth vs comparison labeler(s) for a single image.

    Parameters
    ----------
    site_id : str
        Site ID
    gt_date : datetime
        Ground truth date
    comparison_dates : Dict[str, datetime]
        Dictionary mapping comparison operator -> date
    gt_operator : str
        Ground truth operator initials
    gdf_poly_dict : Dict[str, gpd.GeoDataFrame]
        Dictionary mapping operator -> polygon geodataframe
    gdf_img_dict : Dict[str, gpd.GeoDataFrame]
        Dictionary mapping operator -> image boundary geodataframe
    figsize : Tuple[int, int], default=(15, 5)
        Figure size
    """
    # Get GT polygons and boundary
    gt_polys = get_polygons_for_image(gdf_poly_dict[gt_operator], site_id, gt_date)
    gt_boundary = get_image_boundary(gdf_img_dict[gt_operator], site_id, gt_date)
    gt_internal_id = get_internal_id(gdf_img_dict[gt_operator], site_id, gt_date)

    # Get comparison polygons
    comp_poly_dict = {}
    comp_boundary_dict = {}
    comp_id_dict = {}

    for comp_op, comp_date in comparison_dates.items():
        comp_poly_dict[comp_op] = get_polygons_for_image(
            gdf_poly_dict[comp_op], site_id, comp_date
        )
        comp_boundary_dict[comp_op] = get_image_boundary(
            gdf_img_dict[comp_op], site_id, comp_date
        )
        comp_id_dict[comp_op] = get_internal_id(
            gdf_img_dict[comp_op], site_id, comp_date
        )

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Panel 1: Ground truth
    if gt_boundary is not None:
        gpd.GeoSeries([gt_boundary]).plot(
            ax=ax1, facecolor='lightgray', edgecolor='black', alpha=0.3
        )
    if len(gt_polys) > 0:
        gt_polys.plot(
            ax=ax1, facecolor='blue', edgecolor='darkblue', alpha=0.6, linewidth=1.5
        )
    gt_title = f'{gt_operator} ({gt_date.date()})'
    if gt_internal_id is not None:
        gt_title += f' [ID: {gt_internal_id}]'
    gt_title += f'\n{len(gt_polys)} polygons'
    ax1.set_title(gt_title, fontsize=11)
    ax1.set_aspect('equal')

    # Panel 2: Comparison operator(s) - use different colors for multiple
    comparison_operators = list(comparison_dates.keys())
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']

    if len(comparison_operators) == 1:
        # Single comparison - use red like original
        comp_op = comparison_operators[0]
        comp_date = comparison_dates[comp_op]
        comp_polys = comp_poly_dict[comp_op]
        comp_boundary = comp_boundary_dict[comp_op]
        comp_id = comp_id_dict[comp_op]

        if comp_boundary is not None:
            gpd.GeoSeries([comp_boundary]).plot(
                ax=ax2, facecolor='lightgray', edgecolor='black', alpha=0.3
            )
        if len(comp_polys) > 0:
            comp_polys.plot(
                ax=ax2, facecolor='red', edgecolor='darkred', alpha=0.6, linewidth=1.5
            )
        comp_title = f'{comp_op} ({comp_date.date()})'
        if comp_id is not None:
            comp_title += f' [ID: {comp_id}]'
        comp_title += f'\n{len(comp_polys)} polygons'
        ax2.set_title(comp_title, fontsize=11)
    else:
        # Multiple comparisons - plot all with different colors
        # Use first comparison's boundary
        first_op = comparison_operators[0]
        boundary = comp_boundary_dict[first_op]
        if boundary is not None:
            gpd.GeoSeries([boundary]).plot(
                ax=ax2, facecolor='lightgray', edgecolor='black', alpha=0.3
            )

        total_polys = 0
        legend_handles = []
        for i, comp_op in enumerate(comparison_operators):
            comp_polys = comp_poly_dict[comp_op]
            if len(comp_polys) > 0:
                color = colors[i % len(colors)]
                comp_polys.plot(
                    ax=ax2,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.6,
                    linewidth=1.5
                )
                # Create manual legend handle
                legend_handles.append(Patch(facecolor=color, edgecolor=color, label=comp_op))
            total_polys += len(comp_polys)

        ax2.set_title(f'Comparison Operators\n{total_polys} total polygons', fontsize=11)
        if legend_handles:
            ax2.legend(handles=legend_handles, loc='upper right')

    ax2.set_aspect('equal')

    # Panel 3: Overlay
    boundary = gt_boundary if gt_boundary is not None else comp_boundary_dict.get(comparison_operators[0])
    if boundary is not None:
        gpd.GeoSeries([boundary]).plot(
            ax=ax3, facecolor='lightgray', edgecolor='black', alpha=0.3
        )

    # Plot GT in blue and build legend
    legend_handles_overlay = []
    if len(gt_polys) > 0:
        gt_polys.plot(
            ax=ax3, facecolor='blue', edgecolor='darkblue',
            alpha=0.4, linewidth=1.2
        )
        legend_handles_overlay.append(Patch(facecolor='blue', edgecolor='darkblue', label=gt_operator))

    # Plot comparison operators in their respective colors
    if len(comparison_operators) == 1:
        comp_op = comparison_operators[0]
        comp_polys = comp_poly_dict[comp_op]
        if len(comp_polys) > 0:
            comp_polys.plot(
                ax=ax3, facecolor='red', edgecolor='darkred',
                alpha=0.4, linewidth=1.2
            )
            legend_handles_overlay.append(Patch(facecolor='red', edgecolor='darkred', label=comp_op))
    else:
        for i, comp_op in enumerate(comparison_operators):
            comp_polys = comp_poly_dict[comp_op]
            if len(comp_polys) > 0:
                color = colors[i % len(colors)]
                comp_polys.plot(
                    ax=ax3,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.4,
                    linewidth=1.2
                )
                legend_handles_overlay.append(Patch(facecolor=color, edgecolor=color, label=comp_op))

    ax3.set_title('Overlay', fontsize=11)
    ax3.set_aspect('equal')
    if legend_handles_overlay:
        ax3.legend(handles=legend_handles_overlay, loc='upper right')

    plt.suptitle(f'Site: {site_id}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def should_plot_image(
    site_id: str,
    gt_date: datetime,
    comparison_dates: Dict[str, datetime],
    gt_operator: str,
    gdf_poly_dict: Dict[str, gpd.GeoDataFrame]
) -> bool:
    """
    Determine if an image should be plotted.
    Skip if no polygons were drawn by anyone.

    Parameters
    ----------
    site_id : str
        Site ID
    gt_date : datetime
        Ground truth date
    comparison_dates : Dict[str, datetime]
        Dictionary mapping comparison operator -> date
    gt_operator : str
        Ground truth operator initials
    gdf_poly_dict : Dict[str, gpd.GeoDataFrame]
        Dictionary mapping operator -> polygon geodataframe

    Returns
    -------
    bool
        True if should plot, False otherwise
    """
    # Check GT
    gt_polys = get_polygons_for_image(gdf_poly_dict[gt_operator], site_id, gt_date)
    if len(gt_polys) > 0:
        return True

    # Check comparison operators
    for comp_op, comp_date in comparison_dates.items():
        comp_polys = get_polygons_for_image(gdf_poly_dict[comp_op], site_id, comp_date)
        if len(comp_polys) > 0:
            return True

    return False


# ============================================================================
# IMAGE-LEVEL DETECTION METRICS
# ============================================================================

def compute_image_detection_metrics(
    matches_df: pd.DataFrame,
    gt_operator: str,
    comparison_operator: str,
    gdf_poly_dict: Dict[str, gpd.GeoDataFrame]
) -> Dict[str, any]:
    """
    Compute image-level irrigation detection metrics (TP, FP, FN, TN).

    Parameters
    ----------
    matches_df : pd.DataFrame
        Matched images dataframe
    gt_operator : str
        Ground truth operator
    comparison_operator : str
        Comparison operator
    gdf_poly_dict : Dict[str, gpd.GeoDataFrame]
        Dictionary mapping operator -> polygon geodataframe

    Returns
    -------
    dict
        Dictionary with:
        - tp: True positives (both saw irrigation)
        - fp: False positives (only comp saw irrigation)
        - fn: False negatives (only GT saw irrigation)
        - tn: True negatives (neither saw irrigation)
        - fpr: False positive rate
        - fnr: False negative rate
        - precision: Precision
        - recall: Recall (same as TPR)
        - f1: F1 score
        - total_images: Total matched images
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for _, row in matches_df.iterrows():
        site_id = row['site_id']
        gt_date = row['gt_date']
        comp_date = row['comp_date']

        # Get polygon counts
        gt_polys = get_polygons_for_image(gdf_poly_dict[gt_operator], site_id, gt_date)
        comp_polys = get_polygons_for_image(gdf_poly_dict[comparison_operator], site_id, comp_date)

        gt_has_irr = len(gt_polys) > 0
        comp_has_irr = len(comp_polys) > 0

        if gt_has_irr and comp_has_irr:
            tp += 1
        elif not gt_has_irr and comp_has_irr:
            fp += 1
        elif gt_has_irr and not comp_has_irr:
            fn += 1
        else:  # not gt_has_irr and not comp_has_irr
            tn += 1

    # Compute rates
    total = tp + fp + fn + tn

    # FPR = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # FNR = FN / (FN + TP)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 = 2 * (precision * recall) / (precision + recall)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'fpr': fpr,
        'fnr': fnr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_images': total
    }


def plot_confusion_matrix(
    metrics: Dict[str, any],
    gt_operator: str,
    comparison_operator: str,
    ax=None
):
    """
    Plot confusion matrix for image-level detection.

    Parameters
    ----------
    metrics : dict
        Dictionary from compute_image_detection_metrics
    gt_operator : str
        Ground truth operator
    comparison_operator : str
        Comparison operator
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure.

    Returns
    -------
    matplotlib figure or None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = None

    # Create confusion matrix
    # Standard layout: rows=prediction, cols=actual
    # [TP, FP]  <- Comp says Irrigation
    # [FN, TN]  <- Comp says No Irrigation
    cm = np.array([
        [metrics['tp'], metrics['fp']],
        [metrics['fn'], metrics['tn']]
    ])

    # Plot
    im = ax.imshow(cm, cmap='Blues', aspect='auto')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center", color="black", fontsize=20)

    # Labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f'{gt_operator}\nIrrigation', f'{gt_operator}\nNo Irrigation'])
    ax.set_yticklabels([f'{comparison_operator}\nIrrigation', f'{comparison_operator}\nNo Irrigation'])
    ax.set_xlabel('Ground Truth', fontsize=12, fontweight='bold')
    ax.set_ylabel('Prediction', fontsize=12, fontweight='bold')

    # Title
    title = f'Image-Level Detection: {gt_operator} (GT) vs {comparison_operator}\n'
    title += f'Total Images: {metrics["total_images"]}'
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Add colorbar
    if fig is not None:
        plt.colorbar(im, ax=ax)

    return fig


def plot_detection_metrics_bar(
    metrics: Dict[str, any],
    gt_operator: str,
    comparison_operator: str,
    ax=None
):
    """
    Plot bar chart of detection metrics.

    Parameters
    ----------
    metrics : dict
        Dictionary from compute_image_detection_metrics
    gt_operator : str
        Ground truth operator
    comparison_operator : str
        Comparison operator
    ax : matplotlib axis, optional
        Axis to plot on. If None, creates new figure.

    Returns
    -------
    matplotlib figure or None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = None

    metric_names = ['FPR', 'FNR', 'Precision', 'Recall', 'F1']
    metric_values = [
        metrics['fpr'],
        metrics['fnr'],
        metrics['precision'],
        metrics['recall'],
        metrics['f1']
    ]

    colors = ['red', 'orange', 'green', 'blue', 'purple']
    bars = ax.bar(metric_names, metric_values, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title(f'Detection Metrics: {gt_operator} (GT) vs {comparison_operator}',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)

    return fig


# ============================================================================
# AREA OVERLAP METRICS
# ============================================================================

def compute_iou(poly1, poly2):
    """Compute Intersection over Union for two polygons."""
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area

    return intersection / union if union > 0 else 0.0


def match_polygons(gdf1, gdf2, iou_threshold=0.1):
    """
    Match polygons between two GeoDataFrames using greedy IoU matching.

    Returns list of (idx1, idx2, iou) tuples.
    """
    matches = []
    used_idx2 = set()

    for idx1, row1 in gdf1.iterrows():
        poly1 = row1.geometry
        best_match = None
        best_iou = iou_threshold

        for idx2, row2 in gdf2.iterrows():
            if idx2 in used_idx2:
                continue

            poly2 = row2.geometry
            iou = compute_iou(poly1, poly2)

            if iou > best_iou:
                best_iou = iou
                best_match = idx2

        if best_match is not None:
            matches.append((idx1, best_match, best_iou))
            used_idx2.add(best_match)

    return matches


def compute_image_area_metrics(
    site_id: str,
    gt_date: datetime,
    comp_date: datetime,
    gt_operator: str,
    comparison_operator: str,
    gdf_poly_dict: Dict[str, gpd.GeoDataFrame],
    iou_threshold: float = 0.1
) -> Dict[str, any]:
    """
    Compute area overlap metrics for a single image.

    Parameters
    ----------
    site_id : str
        Site ID
    gt_date : datetime
        Ground truth date
    comp_date : datetime
        Comparison date
    gt_operator : str
        Ground truth operator
    comparison_operator : str
        Comparison operator
    gdf_poly_dict : Dict[str, gpd.GeoDataFrame]
        Dictionary mapping operator -> polygon geodataframe
    iou_threshold : float, default=0.1
        Minimum IoU to consider a match

    Returns
    -------
    dict
        Dictionary with:
        - site_id
        - gt_date
        - comp_date
        - n_gt: Number of GT polygons
        - n_comp: Number of comparison polygons
        - n_matched: Number of matched polygons
        - mean_iou: Mean IoU of matches (or 0 if no matches)
        - precision: Polygon precision
        - recall: Polygon recall
        - f1: Polygon F1
        - both_no_irrigation: Whether both saw no irrigation
    """
    gt_polys = get_polygons_for_image(gdf_poly_dict[gt_operator], site_id, gt_date)
    comp_polys = get_polygons_for_image(gdf_poly_dict[comparison_operator], site_id, comp_date)

    n_gt = len(gt_polys)
    n_comp = len(comp_polys)

    # Special case: both agree there's no irrigation
    if n_gt == 0 and n_comp == 0:
        return {
            'site_id': site_id,
            'gt_date': gt_date,
            'comp_date': comp_date,
            'n_gt': 0,
            'n_comp': 0,
            'n_matched': 0,
            'mean_iou': np.nan,
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'both_no_irrigation': True
        }

    # One saw irrigation, other didn't
    if n_gt == 0 or n_comp == 0:
        return {
            'site_id': site_id,
            'gt_date': gt_date,
            'comp_date': comp_date,
            'n_gt': n_gt,
            'n_comp': n_comp,
            'n_matched': 0,
            'mean_iou': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'both_no_irrigation': False
        }

    # Both have polygons - compute matches
    matches = match_polygons(gt_polys, comp_polys, iou_threshold)
    n_matched = len(matches)

    mean_iou = np.mean([iou for _, _, iou in matches]) if n_matched > 0 else 0.0
    precision = n_matched / n_gt if n_gt > 0 else 0.0
    recall = n_matched / n_comp if n_comp > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'site_id': site_id,
        'gt_date': gt_date,
        'comp_date': comp_date,
        'n_gt': n_gt,
        'n_comp': n_comp,
        'n_matched': n_matched,
        'mean_iou': mean_iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'both_no_irrigation': False
    }


def plot_area_overlap_histograms(
    area_metrics_df: pd.DataFrame,
    gt_operator: str,
    comparison_operator: str,
    figsize: Tuple[int, int] = (12, 10)
):
    """
    Plot histograms of area overlap metrics.

    Parameters
    ----------
    area_metrics_df : pd.DataFrame
        DataFrame from compute_image_area_metrics
    gt_operator : str
        Ground truth operator
    comparison_operator : str
        Comparison operator
    figsize : Tuple[int, int], default=(12, 10)
        Figure size

    Returns
    -------
    matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Precision
    axes[0, 0].hist(area_metrics_df['precision'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Precision')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Polygon Precision Distribution')
    axes[0, 0].axvline(area_metrics_df['precision'].mean(), color='red',
                       linestyle='--', linewidth=2, label='Mean')
    axes[0, 0].legend()

    # Recall
    axes[0, 1].hist(area_metrics_df['recall'], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Polygon Recall Distribution')
    axes[0, 1].axvline(area_metrics_df['recall'].mean(), color='red',
                       linestyle='--', linewidth=2, label='Mean')
    axes[0, 1].legend()

    # F1
    axes[1, 0].hist(area_metrics_df['f1'], bins=20, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('F1 Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Polygon F1 Score Distribution')
    axes[1, 0].axvline(area_metrics_df['f1'].mean(), color='red',
                       linestyle='--', linewidth=2, label='Mean')
    axes[1, 0].legend()

    # IoU (excluding both-no-irrigation)
    iou_data = area_metrics_df[~area_metrics_df['both_no_irrigation']]['mean_iou'].dropna()
    axes[1, 1].hist(iou_data, bins=20, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Mean IoU')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Mean IoU Distribution\n(excluding both-no-irrigation)')
    if len(iou_data) > 0:
        axes[1, 1].axvline(iou_data.mean(), color='red',
                          linestyle='--', linewidth=2, label='Mean')
    axes[1, 1].legend()

    plt.suptitle(f'Area Overlap Metrics: {gt_operator} (GT) vs {comparison_operator}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# ============================================================================
# MAIN COMPARISON FUNCTION
# ============================================================================

def compare_labelers(
    irrigation_table_path: str,
    polygons_path: str,
    image_boundaries_path: str,
    gt_operator: str,
    comparison_operators: List[str],
    min_certainty: int = 3,
    date_tolerance_days: int = 1,
    plot_images: bool = True,
    iou_threshold: float = 0.1,
    output_dir: Optional[str] = None
) -> Dict[str, any]:
    """
    Main function to compare ground truth labeler against comparison labelers.

    Parameters
    ----------
    irrigation_table_path : str
        Path to latest_irrigation_table.csv
    polygons_path : str
        Path to latest_polygons.geojson
    image_boundaries_path : str
        Path to latest_irrigation_data.geojson
    gt_operator : str
        Ground truth operator initials (e.g., 'AB')
    comparison_operators : List[str]
        List of comparison operator initials (e.g., ['PS', 'JL'])
    min_certainty : int, default=3
        Minimum certainty level for polygons to include
    date_tolerance_days : int, default=1
        Date matching tolerance in days
    plot_images : bool, default=True
        Whether to generate comparison plots
    iou_threshold : float, default=0.1
        IoU threshold for polygon matching
    output_dir : str, optional
        Directory to save outputs. If None, plots are shown but not saved.

    Returns
    -------
    dict
        Dictionary containing:
        - matches_dict: Dict mapping comparison_op -> matched images DataFrame
        - detection_metrics: Dict mapping comparison_op -> detection metrics
        - area_metrics: Dict mapping comparison_op -> area metrics DataFrame
        - figures: Dict of generated figures
    """
    print(f"Comparing {gt_operator} (ground truth) vs {comparison_operators}")
    print("="*70)

    # Load data
    print("\n1. Loading data...")
    df, gdf_polygons, gdf_images = load_comparison_data(
        irrigation_table_path, polygons_path, image_boundaries_path
    )
    print(f"   Loaded {len(df)} image labels, {len(gdf_polygons)} polygons")

    # Filter by operators
    print("\n2. Filtering by operators...")
    all_operators = [gt_operator] + comparison_operators
    df_dict, gdf_poly_dict_raw, gdf_img_dict = filter_by_operators(
        df, gdf_polygons, gdf_images, all_operators
    )

    for op in all_operators:
        print(f"   {op}: {len(df_dict[op])} images, {len(gdf_poly_dict_raw[op])} polygons")

    # Filter by certainty
    print(f"\n3. Filtering polygons by certainty >= {min_certainty}...")
    gdf_poly_dict = {}
    for op in all_operators:
        gdf_poly_dict[op] = filter_by_certainty(gdf_poly_dict_raw[op], min_certainty)
        print(f"   {op}: {len(gdf_poly_dict[op])} polygons (after filtering)")

    # Match images
    print(f"\n4. Matching images (±{date_tolerance_days} day tolerance)...")
    matches_dict = match_images_across_operators(
        df_dict, gt_operator, comparison_operators, date_tolerance_days
    )

    # Report matching results
    for comp_op in comparison_operators:
        print(f"   {gt_operator} vs {comp_op}: {len(matches_dict[comp_op])} matched images")

    # Initialize results
    detection_metrics = {}
    area_metrics = {}
    figures = {}

    # Process each comparison operator
    for comp_op in comparison_operators:
        print(f"\n{'='*70}")
        print(f"Processing: {gt_operator} vs {comp_op}")
        print(f"{'='*70}")

        matches_df = matches_dict[comp_op]

        # Compute image-level detection metrics
        print(f"\n5. Computing image-level detection metrics...")
        det_metrics = compute_image_detection_metrics(
            matches_df, gt_operator, comp_op, gdf_poly_dict
        )
        detection_metrics[comp_op] = det_metrics

        print(f"   TP={det_metrics['tp']}, FP={det_metrics['fp']}, "
              f"FN={det_metrics['fn']}, TN={det_metrics['tn']}")
        print(f"   Precision: {det_metrics['precision']:.3f}")
        print(f"   Recall: {det_metrics['recall']:.3f}")
        print(f"   F1: {det_metrics['f1']:.3f}")

        # Plot confusion matrix and metrics
        print(f"\n6. Generating detection visualizations...")
        fig_cm = plot_confusion_matrix(det_metrics, gt_operator, comp_op)
        figures[f'{comp_op}_confusion_matrix'] = fig_cm

        fig_bar = plot_detection_metrics_bar(det_metrics, gt_operator, comp_op)
        figures[f'{comp_op}_detection_metrics'] = fig_bar

        # Compute area overlap metrics
        print(f"\n7. Computing area overlap metrics...")
        area_results = []
        for _, row in matches_df.iterrows():
            site_id = row['site_id']
            gt_date = row['gt_date']
            comp_date = row['comp_date']

            metrics = compute_image_area_metrics(
                site_id, gt_date, comp_date, gt_operator, comp_op,
                gdf_poly_dict, iou_threshold
            )
            area_results.append(metrics)

        area_metrics_df = pd.DataFrame(area_results)
        area_metrics[comp_op] = area_metrics_df

        print(f"   Mean Precision: {area_metrics_df['precision'].mean():.3f}")
        print(f"   Mean Recall: {area_metrics_df['recall'].mean():.3f}")
        print(f"   Mean F1: {area_metrics_df['f1'].mean():.3f}")

        # Plot area overlap histograms
        print(f"\n8. Generating area overlap histograms...")
        fig_area = plot_area_overlap_histograms(area_metrics_df, gt_operator, comp_op)
        figures[f'{comp_op}_area_histograms'] = fig_area

    # Plot comparison images
    if plot_images:
        print(f"\n{'='*70}")
        print(f"Plotting image comparisons...")
        print(f"{'='*70}\n")

        # Collect all unique (site_id, gt_date) pairs from all matches
        all_images = {}  # key: (site_id, gt_date), value: {comp_op: comp_date}
        for comp_op in comparison_operators:
            for _, row in matches_dict[comp_op].iterrows():
                site_id = row['site_id']
                gt_date = row['gt_date']
                comp_date = row['comp_date']

                key = (site_id, gt_date)
                if key not in all_images:
                    all_images[key] = {}
                all_images[key][comp_op] = comp_date

        plotted = 0
        for (site_id, gt_date), comparison_dates in all_images.items():
            # Check if should plot
            if not should_plot_image(site_id, gt_date, comparison_dates,
                                    gt_operator, gdf_poly_dict):
                continue

            fig = plot_comparison(
                site_id, gt_date, comparison_dates, gt_operator,
                gdf_poly_dict, gdf_img_dict
            )

            if output_dir:
                import os
                os.makedirs(output_dir, exist_ok=True)
                fig.savefig(f"{output_dir}/{site_id}_{gt_date.date()}.png",
                           dpi=150, bbox_inches='tight')
                plt.close(fig)
            else:
                plt.show()

            plotted += 1
            print(f"   Plotted {plotted}/{len(all_images)}: {site_id}")

    # Save other figures
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(f"{output_dir}/{name}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        print(f"\nSaved all outputs to {output_dir}")
    else:
        # Show the metric figures
        for fig in figures.values():
            plt.show()

    print(f"\n{'='*70}")
    print("Comparison complete!")
    print(f"{'='*70}\n")

    return {
        'matches_dict': matches_dict,
        'detection_metrics': detection_metrics,
        'area_metrics': area_metrics,
        'figures': figures
    }
