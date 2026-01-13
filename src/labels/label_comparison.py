"""
Object-oriented interface for inter-rater label comparison.

This module provides the LabelComparison class for comparing irrigation labels
between a ground truth labeler and one or more comparison labelers.
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import warnings

# Import helper functions from the original module
from .inter_rater_comparison import (
    load_comparison_data,
    filter_by_operators,
    filter_by_certainty,
    get_polygons_for_image,
    get_image_boundary,
    get_internal_id,
    compute_iou,
    match_polygons
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
    iou_threshold : float, default=0.1
        IoU threshold for polygon matching
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
        iou_threshold: float = 0.1,
        output_dir: Optional[str] = None
    ):
        self.irrigation_table_path = irrigation_table_path
        self.polygons_path = polygons_path
        self.image_boundaries_path = image_boundaries_path
        self.gt_operator = gt_operator
        self.comparison_operators = comparison_operators
        self.min_certainty = min_certainty
        self.date_tolerance_days = date_tolerance_days
        self.iou_threshold = iou_threshold
        self.output_dir = output_dir

        # Create output directory if specified
        if self.output_dir:
            import os
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
        Compute polygon-level area overlap metrics.

        Parameters
        ----------
        comp_op : str
            Comparison operator initials

        Returns
        -------
        pd.DataFrame
            Per-image area metrics with precision, recall, f1, mean_iou
        """
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

            # Both agree no irrigation
            if n_gt == 0 and n_comp == 0:
                results.append({
                    'site_id': site_id, 'gt_date': gt_date, 'comp_date': comp_date,
                    'n_gt': 0, 'n_comp': 0, 'n_matched': 0,
                    'mean_iou': np.nan, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                    'both_no_irrigation': True
                })
                continue

            # One saw irrigation, other didn't
            if n_gt == 0 or n_comp == 0:
                results.append({
                    'site_id': site_id, 'gt_date': gt_date, 'comp_date': comp_date,
                    'n_gt': n_gt, 'n_comp': n_comp, 'n_matched': 0,
                    'mean_iou': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                    'both_no_irrigation': False
                })
                continue

            # Both have polygons - compute matches
            matches = match_polygons(gt_polys, comp_polys, self.iou_threshold)
            n_matched = len(matches)
            mean_iou = np.mean([iou for _, _, iou in matches]) if n_matched > 0 else 0.0

            # FIXED: Precision and recall were swapped!
            # Precision = of comp's polygons, how many matched GT
            precision = n_matched / n_comp if n_comp > 0 else 0.0
            # Recall = of GT's polygons, how many were found by comp
            recall = n_matched / n_gt if n_gt > 0 else 0.0

            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            results.append({
                'site_id': site_id, 'gt_date': gt_date, 'comp_date': comp_date,
                'n_gt': n_gt, 'n_comp': n_comp, 'n_matched': n_matched,
                'mean_iou': mean_iou, 'precision': precision, 'recall': recall, 'f1': f1,
                'both_no_irrigation': False
            })

        area_metrics_df = pd.DataFrame(results)
        self._area_metrics[comp_op] = area_metrics_df
        return area_metrics_df

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

        im = ax.imshow(cm, cmap='Blues', aspect='auto')

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
                import os
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
                import os
                filepath = os.path.join(self.output_dir, f'{comp_op}_detection_metrics.png')
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"Saved: {filepath}")
            if show:
                plt.show()
            if self.output_dir:
                plt.close(fig)

        return fig if created_fig else None

    def plot_area_histograms(self, comp_op: str, figsize=(12, 10), show=True):
        """Plot histograms of area overlap metrics."""
        area_metrics_df = self.compute_area_metrics(comp_op)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Precision
        axes[0, 0].hist(area_metrics_df['precision'], bins=20,
                       edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Precision')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Polygon Precision Distribution')
        axes[0, 0].axvline(area_metrics_df['precision'].mean(),
                          color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0, 0].legend()

        # Recall
        axes[0, 1].hist(area_metrics_df['recall'], bins=20,
                       edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Polygon Recall Distribution')
        axes[0, 1].axvline(area_metrics_df['recall'].mean(),
                          color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0, 1].legend()

        # F1
        axes[1, 0].hist(area_metrics_df['f1'], bins=20,
                       edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('F1 Score')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Polygon F1 Score Distribution')
        axes[1, 0].axvline(area_metrics_df['f1'].mean(),
                          color='red', linestyle='--', linewidth=2, label='Mean')
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

        plt.suptitle(f'Area Overlap Metrics: {self.gt_operator} (GT) vs {comp_op}',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        if self.output_dir:
            import os
            filepath = os.path.join(self.output_dir, f'{comp_op}_area_histograms.png')
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        if show:
            plt.show()
        if self.output_dir:
            plt.close(fig)

        return fig

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
            gt_polys.plot(ax=ax1, facecolor='blue', edgecolor='darkblue',
                         alpha=0.6, linewidth=1.5)

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
                comp_polys.plot(ax=ax2, facecolor='red', edgecolor='darkred',
                               alpha=0.6, linewidth=1.5)

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
                        comp_polys.plot(ax=ax2, facecolor=color, edgecolor=color,
                                       alpha=0.6, linewidth=1.5)
                        total_polys += len(comp_polys)
                        legend_handles.append(Patch(facecolor=color, edgecolor=color,
                                                   label=f'{comp_op} ({len(comp_polys)})'))
                    else:
                        # Still add to legend even if no polygons
                        legend_handles.append(Patch(facecolor=color, edgecolor=color,
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
            gt_polys.plot(ax=ax3, facecolor='blue', edgecolor='darkblue',
                         alpha=0.4, linewidth=1.2)
            legend_handles_overlay.append(Patch(facecolor='blue', edgecolor='darkblue',
                                                label=self.gt_operator))

        if len(comparison_dates) == 1:
            comp_op = list(comparison_dates.keys())[0]
            comp_polys = comp_poly_dict[comp_op]
            if len(comp_polys) > 0:
                comp_polys.plot(ax=ax3, facecolor='red', edgecolor='darkred',
                               alpha=0.4, linewidth=1.2)
                legend_handles_overlay.append(Patch(facecolor='red', edgecolor='darkred',
                                                    label=comp_op))
        else:
            for i, comp_op in enumerate(self.comparison_operators):
                if comp_op in comp_poly_dict:
                    comp_polys = comp_poly_dict[comp_op]
                    if len(comp_polys) > 0:
                        color = colors[i % len(colors)]
                        comp_polys.plot(ax=ax3, facecolor=color, edgecolor=color,
                                       alpha=0.4, linewidth=1.2)
                        legend_handles_overlay.append(Patch(facecolor=color,
                                                           edgecolor=color,
                                                           label=comp_op))

        ax3.set_title('Overlay', fontsize=11)
        ax3.set_aspect('equal')
        if legend_handles_overlay:
            ax3.legend(handles=legend_handles_overlay, loc='upper right')

        plt.suptitle(f'Site: {site_id}', fontsize=13, fontweight='bold')
        plt.tight_layout()

        if self.output_dir:
            import os
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
            import os
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

        # Area metrics
        area = self.compute_area_metrics(comp_op)
        print(f"\nPolygon-Level Area Overlap (mean):")
        print(f"  Precision: {area['precision'].mean():.3f}")
        print(f"  Recall: {area['recall'].mean():.3f}")
        print(f"  F1: {area['f1'].mean():.3f}")

        iou_data = area[~area['both_no_irrigation']]['mean_iou'].dropna()
        if len(iou_data) > 0:
            print(f"  Mean IoU: {iou_data.mean():.3f}")
