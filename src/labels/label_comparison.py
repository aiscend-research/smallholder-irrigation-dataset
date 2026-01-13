"""
Inter-Rater Label Comparison for Quality Control.

This module provides the LabelComparison class for assessing labeling consistency
by comparing irrigation polygon labels between a ground truth (GT) labeler and
one or more comparison labelers.

Usage
-----
```python
from src.labels.label_comparison import LabelComparison

comparison = LabelComparison(
    irrigation_table_path='data/labels/.../latest_irrigation_table.csv',
    polygons_path='data/labels/.../latest_polygons.geojson',
    image_boundaries_path='data/labels/.../latest_irrigation_data.geojson',
    gt_operator='AB',
    comparison_operators=['DSB', 'JL', 'KL', 'MV', 'PS'],
    min_certainty=4,
    date_tolerance_days=1,
    output_dir='outputs/comparison'
)

# Generate plots and metrics for each operator
for op in comparison.comparison_operators:
    comparison.plot_confusion_matrix(op)
    comparison.plot_detection_metrics_bar(op)
    comparison.plot_area_metrics_bar(op)
    comparison.plot_area_histograms(op)
    comparison.print_summary(op)

# Generate summary tables with weighted averages
detection_df, area_df = comparison.generate_summary_tables()
```

Metrics
-------
Two levels of metrics are computed:

1. **Image-Level Detection** (binary: did labeler detect ANY irrigation?)
   - TP: Both GT and comparison labeled irrigation polygons
   - FP: Only comparison labeled (false alarm)
   - FN: Only GT labeled (missed detection)
   - TN: Neither labeled
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)

2. **Area Overlap** (continuous: how much do polygon areas agree?)
   - For each image, union all polygons per labeler
   - Precision = intersection_area / comp_area
     (What fraction of area marked by comparison was correct?)
   - Recall = intersection_area / gt_area
     (What fraction of GT area was found by comparison?)
   - IoU = intersection_area / union_area
   - Overall metrics sum areas across all images before computing ratios

Output Files
------------
When output_dir is set, the following files are saved:
- {op}_confusion_matrix.png: Image detection confusion matrix
- {op}_detection_metrics.png: Detection metrics bar chart
- {op}_area_metrics.png: Area overlap metrics bar chart
- {op}_area_histograms.png: Per-image metric distributions
- {site_id}_{date}.png: Side-by-side polygon comparison plots
- image_detection_metrics.csv: Summary table with all operators
- area_overlap_metrics.csv: Summary table with all operators

See Also
--------
- notebooks/labeler_comparison.ipynb: Interactive notebook for running comparisons
- src/labels/inter_rater_comparison.py: Helper functions used by this class
"""

import os

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Import helper functions from the original module
from .inter_rater_comparison import (
    load_comparison_data,
    filter_by_operators,
    filter_by_certainty,
    get_polygons_for_image,
    get_image_boundary,
    get_internal_id,
)


class LabelComparison:
    """
    Compare irrigation labels between a ground truth labeler and comparison labelers.

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
        Minimum certainty level for polygons
    date_tolerance_days : int, default=1
        Date matching tolerance in days
    output_dir : str, optional
        Directory to save all figures. If None, figures are shown but not saved.
    """

    def __init__(
        self,
        irrigation_table_path: str,
        polygons_path: str,
        image_boundaries_path: str,
        gt_operator: str,
        comparison_operators: List[str],
        min_certainty: int = 3,
        date_tolerance_days: int = 1,
        output_dir: Optional[str] = None
    ):
        self.irrigation_table_path = irrigation_table_path
        self.polygons_path = polygons_path
        self.image_boundaries_path = image_boundaries_path
        self.gt_operator = gt_operator
        self.comparison_operators = comparison_operators
        self.min_certainty = min_certainty
        self.date_tolerance_days = date_tolerance_days
        self.output_dir = output_dir

        # Create output directory if specified
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Figures will be saved to: {self.output_dir}")

        # Initialize data containers
        self._matches_dict = None
        self._detection_metrics = {}
        self._area_metrics = {}

        # Load and prepare data
        self._load_data()
        self._match_images()

    def _load_data(self):
        """Load and filter data."""
        print(f"Loading data for {self.gt_operator} vs {self.comparison_operators}...")

        # Load raw data
        df, gdf_polygons, gdf_images = load_comparison_data(
            self.irrigation_table_path,
            self.polygons_path,
            self.image_boundaries_path
        )
        print(f"  Loaded {len(df)} image labels, {len(gdf_polygons)} polygons")

        # Filter by operators
        all_operators = [self.gt_operator] + self.comparison_operators
        df_dict, gdf_poly_dict_raw, gdf_img_dict = filter_by_operators(
            df, gdf_polygons, gdf_images, all_operators
        )

        for op in all_operators:
            print(f"  {op}: {len(df_dict[op])} images, {len(gdf_poly_dict_raw[op])} polygons")

        # Filter by certainty
        print(f"  Filtering by certainty >= {self.min_certainty}...")
        gdf_poly_dict = {}
        for op in all_operators:
            gdf_poly_dict[op] = filter_by_certainty(gdf_poly_dict_raw[op], self.min_certainty)

        self.df_dict = df_dict
        self.gdf_poly_dict = gdf_poly_dict
        self.gdf_img_dict = gdf_img_dict

    def _match_images(self):
        """Match images between GT and each comparison operator."""
        print(f"Matching images (±{self.date_tolerance_days} day tolerance)...")

        df_gt = self.df_dict[self.gt_operator].copy()
        df_gt['date'] = pd.to_datetime(df_gt[['year', 'month', 'day']])
        gt_images = df_gt[['site_id', 'date']].drop_duplicates()

        matches_dict = {}

        for comp_op in self.comparison_operators:
            matches = []
            df_comp = self.df_dict[comp_op].copy()
            df_comp['date'] = pd.to_datetime(df_comp[['year', 'month', 'day']])
            comp_images = df_comp[['site_id', 'date']].drop_duplicates()

            for _, gt_row in gt_images.iterrows():
                site = gt_row['site_id']
                gt_date = gt_row['date']

                comp_at_site = comp_images[comp_images['site_id'] == site]
                best_match = None
                best_diff = float('inf')

                for _, comp_row in comp_at_site.iterrows():
                    comp_date = comp_row['date']
                    date_diff = abs((gt_date - comp_date).days)

                    if date_diff <= self.date_tolerance_days and date_diff < best_diff:
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
            print(f"  {self.gt_operator} vs {comp_op}: {len(matches_dict[comp_op])} matched images")

        self._matches_dict = matches_dict

    def get_matches(self, comp_op: str) -> pd.DataFrame:
        """
        Get matched images for a specific comparison operator.

        Parameters
        ----------
        comp_op : str
            Comparison operator initials

        Returns
        -------
        pd.DataFrame
            Matched images with columns: site_id, gt_date, comp_date, date_diff
        """
        if comp_op not in self._matches_dict:
            raise ValueError(f"Unknown comparison operator: {comp_op}")
        return self._matches_dict[comp_op].copy()

    def compute_detection_metrics(self, comp_op: str) -> Dict:
        """
        Compute image-level detection metrics (TP, FP, FN, TN).

        Parameters
        ----------
        comp_op : str
            Comparison operator initials

        Returns
        -------
        dict
            Detection metrics including tp, fp, fn, tn, precision, recall, f1, etc.
        """
        if comp_op in self._detection_metrics:
            return self._detection_metrics[comp_op]

        matches_df = self._matches_dict[comp_op]

        tp = fp = fn = tn = 0

        for _, row in matches_df.iterrows():
            site_id = row['site_id']
            gt_date = row['gt_date']
            comp_date = row['comp_date']

            gt_polys = get_polygons_for_image(
                self.gdf_poly_dict[self.gt_operator], site_id, gt_date
            )
            comp_polys = get_polygons_for_image(
                self.gdf_poly_dict[comp_op], site_id, comp_date
            )

            gt_has_irr = len(gt_polys) > 0
            comp_has_irr = len(comp_polys) > 0

            if gt_has_irr and comp_has_irr:
                tp += 1
            elif not gt_has_irr and comp_has_irr:
                fp += 1
            elif gt_has_irr and not comp_has_irr:
                fn += 1
            else:
                tn += 1

        total = tp + fp + fn + tn
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'fpr': fpr, 'fnr': fnr,
            'precision': precision, 'recall': recall, 'f1': f1,
            'total_images': total
        }

        self._detection_metrics[comp_op] = metrics
        return metrics

    def compute_area_metrics(self, comp_op: str) -> pd.DataFrame:
        """
        Compute area-based overlap metrics at the image level.

        For each matched image, unions all GT polygons and all comparison polygons,
        then records the areas for computing overall precision/recall.

        Parameters
        ----------
        comp_op : str
            Comparison operator initials

        Returns
        -------
        pd.DataFrame
            Per-image area data with:
            - gt_area: total area marked by GT
            - comp_area: total area marked by comparison labeler
            - intersection_area: overlapping area
            - union_area: total area covered by either labeler
        """
        from shapely.ops import unary_union

        if comp_op in self._area_metrics:
            return self._area_metrics[comp_op]

        matches_df = self._matches_dict[comp_op]
        results = []

        for _, row in matches_df.iterrows():
            site_id = row['site_id']
            gt_date = row['gt_date']
            comp_date = row['comp_date']

            gt_polys = get_polygons_for_image(
                self.gdf_poly_dict[self.gt_operator], site_id, gt_date
            )
            comp_polys = get_polygons_for_image(
                self.gdf_poly_dict[comp_op], site_id, comp_date
            )

            n_gt = len(gt_polys)
            n_comp = len(comp_polys)

            # Compute areas (0 if no polygons)
            if n_gt > 0:
                gt_union = unary_union(gt_polys.geometry)
                gt_area = gt_union.area
            else:
                gt_union = None
                gt_area = 0.0

            if n_comp > 0:
                comp_union = unary_union(comp_polys.geometry)
                comp_area = comp_union.area
            else:
                comp_union = None
                comp_area = 0.0

            # Compute intersection and union areas
            if gt_union is not None and comp_union is not None:
                intersection_area = gt_union.intersection(comp_union).area
                union_area = gt_union.union(comp_union).area
            else:
                intersection_area = 0.0
                union_area = gt_area + comp_area  # No overlap possible

            results.append({
                'site_id': site_id, 'gt_date': gt_date, 'comp_date': comp_date,
                'n_gt': n_gt, 'n_comp': n_comp,
                'gt_area': gt_area, 'comp_area': comp_area,
                'intersection_area': intersection_area, 'union_area': union_area
            })

        area_metrics_df = pd.DataFrame(results)
        self._area_metrics[comp_op] = area_metrics_df
        return area_metrics_df

    def compute_overall_area_metrics(self, comp_op: str) -> Dict:
        """
        Compute overall area-based precision, recall, and IoU across all images.

        Parameters
        ----------
        comp_op : str
            Comparison operator initials

        Returns
        -------
        dict
            Overall metrics:
            - precision: total_intersection / total_comp_area
            - recall: total_intersection / total_gt_area
            - iou: total_intersection / total_union
            - f1: harmonic mean of precision and recall
        """
        area_df = self.compute_area_metrics(comp_op)

        total_gt_area = area_df['gt_area'].sum()
        total_comp_area = area_df['comp_area'].sum()
        total_intersection = area_df['intersection_area'].sum()
        total_union = area_df['union_area'].sum()

        precision = total_intersection / total_comp_area if total_comp_area > 0 else 0.0
        recall = total_intersection / total_gt_area if total_gt_area > 0 else 0.0
        iou = total_intersection / total_union if total_union > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'f1': f1,
            'total_gt_area': total_gt_area,
            'total_comp_area': total_comp_area,
            'total_intersection': total_intersection,
            'total_union': total_union
        }

    def plot_confusion_matrix(self, comp_op: str, ax=None, show=True):
        """Plot confusion matrix for image-level detection."""
        metrics = self.compute_detection_metrics(comp_op)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            created_fig = True
        else:
            fig = None
            created_fig = False

        cm = np.array([
            [metrics['tp'], metrics['fp']],
            [metrics['fn'], metrics['tn']]
        ])

        ax.imshow(cm, cmap='Blues', aspect='auto')

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                       color="black", fontsize=20)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels([f'{self.gt_operator}\nIrrigation',
                            f'{self.gt_operator}\nNo Irrigation'])
        ax.set_yticklabels([f'{comp_op}\nIrrigation',
                            f'{comp_op}\nNo Irrigation'])
        ax.set_xlabel('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_ylabel('Prediction', fontsize=12, fontweight='bold')

        title = f'Image-Level Detection: {self.gt_operator} (GT) vs {comp_op}\n'
        title += f'Total Images: {metrics["total_images"]}'
        ax.set_title(title, fontsize=13, fontweight='bold')

        if created_fig:
            plt.tight_layout()
            if self.output_dir:
                filepath = os.path.join(self.output_dir, f'{comp_op}_confusion_matrix.png')
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"Saved: {filepath}")
            if show:
                plt.show()
            if self.output_dir:
                plt.close(fig)

        return fig if created_fig else None

    def plot_detection_metrics_bar(self, comp_op: str, ax=None, show=True):
        """Plot bar chart of detection metrics."""
        metrics = self.compute_detection_metrics(comp_op)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            created_fig = True
        else:
            fig = None
            created_fig = False

        metric_names = ['FPR', 'FNR', 'Precision', 'Recall', 'F1']
        metric_values = [
            metrics['fpr'], metrics['fnr'],
            metrics['precision'], metrics['recall'], metrics['f1']
        ]

        colors = ['red', 'orange', 'green', 'blue', 'purple']
        bars = ax.bar(metric_names, metric_values, color=colors,
                     alpha=0.7, edgecolor='black')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Detection Metrics: {self.gt_operator} (GT) vs {comp_op}',
                    fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        if created_fig:
            plt.tight_layout()
            if self.output_dir:
                filepath = os.path.join(self.output_dir, f'{comp_op}_detection_metrics.png')
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"Saved: {filepath}")
            if show:
                plt.show()
            if self.output_dir:
                plt.close(fig)

        return fig if created_fig else None

    def plot_area_histograms(self, comp_op: str, figsize=(12, 10), show=True):
        """Plot histograms of per-image area metrics (precision, recall, F1, IoU)."""
        area_df = self.compute_area_metrics(comp_op)

        # Filter to images where at least one labeler saw irrigation
        has_irrigation = (area_df['gt_area'] > 0) | (area_df['comp_area'] > 0)
        area_with_irr = area_df[has_irrigation].copy()

        # Compute per-image metrics
        # Precision: intersection / comp_area (0 if comp_area == 0, i.e., FN case - no prediction)
        # But FN means comp didn't label, so precision is undefined. We include FP cases (precision=0).
        # Recall: intersection / gt_area (0 if gt_area == 0, i.e., FP case - no GT)
        # But FP means GT didn't label, so recall is undefined. We include FN cases (recall=0).

        # For precision: include images where comp labeled (comp_area > 0)
        # If GT also labeled: precision = intersection/comp_area
        # If GT didn't label (FP): precision = 0 (all predictions wrong)
        area_with_irr['precision'] = np.where(
            area_with_irr['comp_area'] > 0,
            area_with_irr['intersection_area'] / area_with_irr['comp_area'],
            np.nan  # No prediction made (FN case) - exclude from precision histogram
        )

        # For recall: include images where GT labeled (gt_area > 0)
        # If comp also labeled: recall = intersection/gt_area
        # If comp didn't label (FN): recall = 0 (missed everything)
        area_with_irr['recall'] = np.where(
            area_with_irr['gt_area'] > 0,
            area_with_irr['intersection_area'] / area_with_irr['gt_area'],
            np.nan  # No GT (FP case) - exclude from recall histogram
        )

        # IoU: intersection / union (always defined if at least one labeled)
        area_with_irr['iou'] = area_with_irr['intersection_area'] / area_with_irr['union_area']
        area_with_irr['iou'] = area_with_irr['iou'].fillna(0)

        # F1: harmonic mean (0 if either precision or recall is 0 or undefined)
        area_with_irr['f1'] = np.where(
            (area_with_irr['precision'] > 0) & (area_with_irr['recall'] > 0),
            2 * area_with_irr['precision'] * area_with_irr['recall'] / (area_with_irr['precision'] + area_with_irr['recall']),
            0.0
        )

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Precision (images where comp labeled something)
        precision_data = area_with_irr['precision'].dropna()
        axes[0, 0].hist(precision_data, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Precision')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title(f'Per-Image Precision\n({len(precision_data)} images where {comp_op} labeled)')
        if len(precision_data) > 0:
            axes[0, 0].axvline(precision_data.mean(), color='red',
                              linestyle='--', linewidth=2, label=f"Mean: {precision_data.mean():.3f}")
            axes[0, 0].legend()

        # Recall (images where GT labeled something)
        recall_data = area_with_irr['recall'].dropna()
        axes[0, 1].hist(recall_data, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title(f'Per-Image Recall\n({len(recall_data)} images where {self.gt_operator} labeled)')
        if len(recall_data) > 0:
            axes[0, 1].axvline(recall_data.mean(), color='red',
                              linestyle='--', linewidth=2, label=f"Mean: {recall_data.mean():.3f}")
            axes[0, 1].legend()

        # F1 (all images with irrigation)
        axes[1, 0].hist(area_with_irr['f1'], bins=20, edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('F1 Score')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Per-Image F1\n({len(area_with_irr)} images with irrigation)')
        if len(area_with_irr) > 0:
            axes[1, 0].axvline(area_with_irr['f1'].mean(), color='red',
                              linestyle='--', linewidth=2, label=f"Mean: {area_with_irr['f1'].mean():.3f}")
            axes[1, 0].legend()

        # IoU (all images with irrigation)
        axes[1, 1].hist(area_with_irr['iou'], bins=20, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('IoU')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title(f'Per-Image IoU\n({len(area_with_irr)} images with irrigation)')
        if len(area_with_irr) > 0:
            axes[1, 1].axvline(area_with_irr['iou'].mean(), color='red',
                              linestyle='--', linewidth=2, label=f"Mean: {area_with_irr['iou'].mean():.3f}")
            axes[1, 1].legend()

        plt.suptitle(f'Per-Image Area Metrics: {self.gt_operator} (GT) vs {comp_op}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if self.output_dir:
            filepath = os.path.join(self.output_dir, f'{comp_op}_area_histograms.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        if show:
            plt.show()
        if self.output_dir:
            plt.close(fig)

        return fig

    def plot_area_metrics_bar(self, comp_op: str, ax=None, show=True):
        """Plot bar chart of overall area-based metrics."""
        metrics = self.compute_overall_area_metrics(comp_op)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            created_fig = True
        else:
            fig = None
            created_fig = False

        metric_names = ['Precision', 'Recall', 'F1', 'IoU']
        metric_values = [
            metrics['precision'], metrics['recall'], metrics['f1'], metrics['iou']
        ]

        colors = ['green', 'blue', 'purple', 'orange']
        bars = ax.bar(metric_names, metric_values, color=colors,
                     alpha=0.7, edgecolor='black')

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylabel('Value', fontsize=12, fontweight='bold')
        ax.set_title(f'Area Overlap Metrics: {self.gt_operator} (GT) vs {comp_op}',
                    fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        if created_fig:
            plt.tight_layout()
            if self.output_dir:
                filepath = os.path.join(self.output_dir, f'{comp_op}_area_metrics.png')
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"Saved: {filepath}")
            if show:
                plt.show()
            if self.output_dir:
                plt.close(fig)

        return fig if created_fig else None

    def plot_image_comparison(self, site_id: str, gt_date: datetime,
                            figsize=(15, 5), show=True):
        """
        Plot comparison for a specific image.

        Parameters
        ----------
        site_id : str
            Site ID
        gt_date : datetime
            Ground truth date
        figsize : tuple, default=(15, 5)
            Figure size
        show : bool, default=True
            Whether to show the plot

        Returns
        -------
        matplotlib.figure.Figure
            The figure object
        """
        # Get GT polygons
        gt_polys = get_polygons_for_image(
            self.gdf_poly_dict[self.gt_operator], site_id, gt_date
        )
        gt_boundary = get_image_boundary(
            self.gdf_img_dict[self.gt_operator], site_id, gt_date
        )
        gt_internal_id = get_internal_id(
            self.gdf_img_dict[self.gt_operator], site_id, gt_date
        )

        # Get comparison polygons and dates
        comparison_dates = {}
        comp_poly_dict = {}
        comp_boundary_dict = {}
        comp_id_dict = {}

        for comp_op in self.comparison_operators:
            matches_df = self._matches_dict[comp_op]
            match = matches_df[(matches_df['site_id'] == site_id) &
                              (matches_df['gt_date'] == gt_date)]

            if len(match) > 0:
                comp_date = match.iloc[0]['comp_date']
                comparison_dates[comp_op] = comp_date
                comp_poly_dict[comp_op] = get_polygons_for_image(
                    self.gdf_poly_dict[comp_op], site_id, comp_date
                )
                comp_boundary_dict[comp_op] = get_image_boundary(
                    self.gdf_img_dict[comp_op], site_id, comp_date
                )
                comp_id_dict[comp_op] = get_internal_id(
                    self.gdf_img_dict[comp_op], site_id, comp_date
                )

        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink']

        # Panel 1: Ground truth
        if gt_boundary is not None:
            gpd.GeoSeries([gt_boundary]).plot(
                ax=ax1, facecolor='lightgray', edgecolor='black', alpha=0.3
            )
        if len(gt_polys) > 0:
            gt_polys.plot(ax=ax1, facecolor='blue', edgecolor='none', alpha=0.3)

        gt_title = f'{self.gt_operator} ({gt_date.date()})'
        if gt_internal_id:
            gt_title += f' [ID: {gt_internal_id}]'
        gt_title += f'\n{len(gt_polys)} polygons'
        ax1.set_title(gt_title, fontsize=11)
        ax1.set_aspect('equal')

        # Panel 2: Comparison operators
        if len(comparison_dates) == 1:
            comp_op = list(comparison_dates.keys())[0]
            comp_date = comparison_dates[comp_op]
            comp_polys = comp_poly_dict[comp_op]
            comp_boundary = comp_boundary_dict[comp_op]
            comp_id = comp_id_dict[comp_op]

            if comp_boundary is not None:
                gpd.GeoSeries([comp_boundary]).plot(
                    ax=ax2, facecolor='lightgray', edgecolor='black', alpha=0.3
                )
            if len(comp_polys) > 0:
                comp_polys.plot(ax=ax2, facecolor='red', edgecolor='none', alpha=0.3)

            comp_title = f'{comp_op} ({comp_date.date()})'
            if comp_id:
                comp_title += f' [ID: {comp_id}]'
            comp_title += f'\n{len(comp_polys)} polygons'
            ax2.set_title(comp_title, fontsize=11)
        else:
            # Multiple comparisons
            boundary = comp_boundary_dict.get(list(comparison_dates.keys())[0])
            if boundary is not None:
                gpd.GeoSeries([boundary]).plot(
                    ax=ax2, facecolor='lightgray', edgecolor='black', alpha=0.3
                )

            # FIXED: Show ALL comparison operators in legend, even if no polygons
            legend_handles = []
            total_polys = 0

            for i, comp_op in enumerate(self.comparison_operators):
                color = colors[i % len(colors)]

                if comp_op in comp_poly_dict:
                    comp_polys = comp_poly_dict[comp_op]
                    if len(comp_polys) > 0:
                        comp_polys.plot(ax=ax2, facecolor=color, edgecolor='none', alpha=0.3)
                        total_polys += len(comp_polys)
                        legend_handles.append(Patch(facecolor=color, edgecolor='none',
                                                   label=f'{comp_op} ({len(comp_polys)})'))
                    else:
                        legend_handles.append(Patch(facecolor=color, edgecolor='none',
                                                   label=f'{comp_op} (0)', alpha=0.3))
                else:
                    # Operator didn't label this image - show in legend as "N/A"
                    legend_handles.append(Patch(facecolor='gray', edgecolor='gray',
                                               label=f'{comp_op} (N/A)', alpha=0.2))

            ax2.set_title(f'Comparison Operators\n{total_polys} total polygons', fontsize=11)
            if legend_handles:
                ax2.legend(handles=legend_handles, loc='upper right')

        ax2.set_aspect('equal')

        # Panel 3: Overlay
        boundary = gt_boundary if gt_boundary is not None else comp_boundary_dict.get(
            list(comparison_dates.keys())[0] if comparison_dates else None
        )
        if boundary is not None:
            gpd.GeoSeries([boundary]).plot(
                ax=ax3, facecolor='lightgray', edgecolor='black', alpha=0.3
            )

        legend_handles_overlay = []
        if len(gt_polys) > 0:
            gt_polys.plot(ax=ax3, facecolor='blue', edgecolor='none', alpha=0.25)
            legend_handles_overlay.append(Patch(facecolor='blue', edgecolor='none',
                                                label=self.gt_operator))

        if len(comparison_dates) == 1:
            comp_op = list(comparison_dates.keys())[0]
            comp_polys = comp_poly_dict[comp_op]
            if len(comp_polys) > 0:
                comp_polys.plot(ax=ax3, facecolor='red', edgecolor='none', alpha=0.25)
                legend_handles_overlay.append(Patch(facecolor='red', edgecolor='none',
                                                    label=comp_op))
        else:
            for i, comp_op in enumerate(self.comparison_operators):
                if comp_op in comp_poly_dict:
                    comp_polys = comp_poly_dict[comp_op]
                    if len(comp_polys) > 0:
                        color = colors[i % len(colors)]
                        comp_polys.plot(ax=ax3, facecolor=color, edgecolor='none', alpha=0.25)
                        legend_handles_overlay.append(Patch(facecolor=color,
                                                           edgecolor='none',
                                                           label=comp_op))

        ax3.set_title('Overlay', fontsize=11)
        ax3.set_aspect('equal')
        if legend_handles_overlay:
            ax3.legend(handles=legend_handles_overlay, loc='upper right')

        plt.suptitle(f'Site: {site_id}', fontsize=13, fontweight='bold')
        plt.tight_layout()

        if self.output_dir:
            filepath = os.path.join(self.output_dir, f'{site_id}_{gt_date.date()}.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        if self.output_dir:
            plt.close(fig)

        return fig

    def get_images_with_polygons(self) -> List[Tuple[str, datetime]]:
        """
        Get list of (site_id, gt_date) tuples for images where at least one labeler drew polygons.

        Returns
        -------
        List[Tuple[str, datetime]]
            List of (site_id, gt_date) tuples
        """
        images_with_polys = []

        # Collect all unique (site_id, gt_date) pairs
        all_images = set()
        for comp_op in self.comparison_operators:
            for _, row in self._matches_dict[comp_op].iterrows():
                all_images.add((row['site_id'], row['gt_date']))

        for site_id, gt_date in all_images:
            # Check if GT has polygons
            gt_polys = get_polygons_for_image(
                self.gdf_poly_dict[self.gt_operator], site_id, gt_date
            )
            if len(gt_polys) > 0:
                images_with_polys.append((site_id, gt_date))
                continue

            # Check if any comparison operator has polygons
            for comp_op in self.comparison_operators:
                matches_df = self._matches_dict[comp_op]
                match = matches_df[(matches_df['site_id'] == site_id) &
                                  (matches_df['gt_date'] == gt_date)]
                if len(match) > 0:
                    comp_date = match.iloc[0]['comp_date']
                    comp_polys = get_polygons_for_image(
                        self.gdf_poly_dict[comp_op], site_id, comp_date
                    )
                    if len(comp_polys) > 0:
                        images_with_polys.append((site_id, gt_date))
                        break

        return sorted(images_with_polys)

    def plot_images_with_polygons(self, show=True):
        """
        Plot only images where at least one labeler drew polygons.

        Parameters
        ----------
        show : bool, default=True
            Whether to show plots (only if output_dir is None)
        """
        images = self.get_images_with_polygons()
        print(f"Plotting {len(images)} images with polygons...")

        for i, (site_id, gt_date) in enumerate(images, 1):
            self.plot_image_comparison(site_id, gt_date, show=show)
            if not self.output_dir:
                print(f"  Plotted {i}/{len(images)}: {site_id}")

        if self.output_dir:
            print(f"Saved {len(images)} images to {self.output_dir}")

    def plot_all_images(self, output_dir: Optional[str] = None, show=True):
        """
        Plot all matched images.

        Parameters
        ----------
        output_dir : str, optional
            Directory to save figures. If None, shows inline.
        show : bool, default=True
            Whether to show plots (only if output_dir is None)
        """
        # Collect all unique (site_id, gt_date) pairs
        all_images = set()
        for comp_op in self.comparison_operators:
            for _, row in self._matches_dict[comp_op].iterrows():
                all_images.add((row['site_id'], row['gt_date']))

        print(f"Plotting {len(all_images)} images...")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for i, (site_id, gt_date) in enumerate(sorted(all_images), 1):
            # Check if there's anything to plot
            gt_polys = get_polygons_for_image(
                self.gdf_poly_dict[self.gt_operator], site_id, gt_date
            )

            has_any_polys = len(gt_polys) > 0
            for comp_op in self.comparison_operators:
                matches_df = self._matches_dict[comp_op]
                match = matches_df[(matches_df['site_id'] == site_id) &
                                  (matches_df['gt_date'] == gt_date)]
                if len(match) > 0:
                    comp_date = match.iloc[0]['comp_date']
                    comp_polys = get_polygons_for_image(
                        self.gdf_poly_dict[comp_op], site_id, comp_date
                    )
                    if len(comp_polys) > 0:
                        has_any_polys = True
                        break

            if not has_any_polys:
                continue

            fig = self.plot_image_comparison(site_id, gt_date, show=False)

            if output_dir:
                fig.savefig(f"{output_dir}/{site_id}_{gt_date.date()}.png",
                           dpi=150, bbox_inches='tight')
                plt.close(fig)
            elif show:
                plt.show()

            print(f"  Plotted {i}/{len(all_images)}: {site_id}")

        if output_dir:
            print(f"Saved all plots to {output_dir}")

    def print_summary(self, comp_op: str):
        """Print summary statistics for a comparison."""
        print(f"\n{'='*70}")
        print(f"{self.gt_operator} vs {comp_op} Summary")
        print(f"{'='*70}")

        # Matches
        matches_df = self._matches_dict[comp_op]
        print(f"\nMatched images: {len(matches_df)}")

        # Detection metrics
        det = self.compute_detection_metrics(comp_op)
        print(f"\nImage-Level Detection:")
        print(f"  TP (both saw irrigation): {det['tp']}")
        print(f"  FP (GT no irr, {comp_op} saw irr): {det['fp']}")
        print(f"  FN (GT saw irr, {comp_op} no irr): {det['fn']}")
        print(f"  TN (both saw no irrigation): {det['tn']}")
        print(f"  Precision: {det['precision']:.3f}")
        print(f"  Recall: {det['recall']:.3f}")
        print(f"  F1: {det['f1']:.3f}")

        # Overall area metrics
        area_metrics = self.compute_overall_area_metrics(comp_op)
        print(f"\nArea Overlap (overall):")
        print(f"  Precision: {area_metrics['precision']:.3f}")
        print(f"  Recall: {area_metrics['recall']:.3f}")
        print(f"  F1: {area_metrics['f1']:.3f}")
        print(f"  IoU: {area_metrics['iou']:.3f}")

    def generate_summary_tables(self, save_csv: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate summary tables for image detection and area overlap metrics.

        Compiles metrics across all comparison operators and computes weighted
        averages based on total images labeled by each operator in the dataset.

        Parameters
        ----------
        save_csv : bool, default=True
            Whether to save tables as CSV files to output_dir

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (image_detection_table, area_overlap_table)
        """
        # Get total images labeled by each operator (from full dataset, not just matched)
        total_images_by_op = {}
        for op in self.comparison_operators:
            total_images_by_op[op] = len(self.df_dict[op])

        # Compile image detection metrics
        detection_rows = []
        for comp_op in self.comparison_operators:
            det = self.compute_detection_metrics(comp_op)
            matches_df = self._matches_dict[comp_op]
            detection_rows.append({
                'Operator': comp_op,
                'Matched Images': len(matches_df),
                'Total Images Labeled': total_images_by_op[comp_op],
                'TP': det['tp'],
                'FP': det['fp'],
                'FN': det['fn'],
                'TN': det['tn'],
                'Precision': det['precision'],
                'Recall': det['recall'],
                'F1': det['f1']
            })

        detection_df = pd.DataFrame(detection_rows)

        # Compute weighted average for detection metrics
        weights = detection_df['Total Images Labeled'].values
        weight_sum = weights.sum()

        detection_avg = {
            'Operator': 'Weighted Avg',
            'Matched Images': detection_df['Matched Images'].sum(),
            'Total Images Labeled': weight_sum,
            'TP': detection_df['TP'].sum(),
            'FP': detection_df['FP'].sum(),
            'FN': detection_df['FN'].sum(),
            'TN': detection_df['TN'].sum(),
            'Precision': np.average(detection_df['Precision'], weights=weights),
            'Recall': np.average(detection_df['Recall'], weights=weights),
            'F1': np.average(detection_df['F1'], weights=weights)
        }
        detection_df = pd.concat([detection_df, pd.DataFrame([detection_avg])], ignore_index=True)

        # Compile area overlap metrics
        area_rows = []
        for comp_op in self.comparison_operators:
            area = self.compute_overall_area_metrics(comp_op)
            area_rows.append({
                'Operator': comp_op,
                'Total Images Labeled': total_images_by_op[comp_op],
                'Precision': area['precision'],
                'Recall': area['recall'],
                'F1': area['f1'],
                'IoU': area['iou']
            })

        area_df = pd.DataFrame(area_rows)

        # Compute weighted average for area metrics
        area_avg = {
            'Operator': 'Weighted Avg',
            'Total Images Labeled': weight_sum,
            'Precision': np.average(area_df['Precision'], weights=weights),
            'Recall': np.average(area_df['Recall'], weights=weights),
            'F1': np.average(area_df['F1'], weights=weights),
            'IoU': np.average(area_df['IoU'], weights=weights)
        }
        area_df = pd.concat([area_df, pd.DataFrame([area_avg])], ignore_index=True)

        # Display tables
        print("\n" + "="*80)
        print("IMAGE-LEVEL DETECTION METRICS")
        print("="*80)
        print(detection_df.to_string(index=False))

        print("\n" + "="*80)
        print("AREA OVERLAP METRICS")
        print("="*80)
        print(area_df.to_string(index=False))

        # Save CSVs
        if save_csv and self.output_dir:
            detection_path = os.path.join(self.output_dir, 'image_detection_metrics.csv')
            area_path = os.path.join(self.output_dir, 'area_overlap_metrics.csv')

            detection_df.to_csv(detection_path, index=False)
            area_df.to_csv(area_path, index=False)

            print(f"\nSaved: {detection_path}")
            print(f"Saved: {area_path}")

        return detection_df, area_df
