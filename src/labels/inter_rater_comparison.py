"""
Helper functions for inter-rater label comparison.

This module provides reusable functions for loading, filtering, and processing
irrigation label data for comparison between labelers.
"""

import pandas as pd
import geopandas as gpd
from datetime import datetime
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
