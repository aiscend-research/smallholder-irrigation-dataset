#!/usr/bin/env python3
"""
Adjust PS (Peter Siame) annotation dates back by one day.

Peter's annotations were recorded with dates one day later than the actual
image dates due to timezone differences during labeling. This script corrects
those dates in the master dataset files.

Usage:
    python src/processing/adjust_ps_dates.py [--data-dir DATA_DIR]

By default, adjusts files in data/labels/labeled_surveys/random_sample/
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd


def adjust_date_back_one_day(year: int, month: int, day: int) -> tuple:
    """Subtract one day from a date and return (year, month, day)."""
    date = datetime(int(year), int(month), int(day))
    new_date = date - timedelta(days=1)
    return new_date.year, new_date.month, new_date.day


def adjust_csv_ps_dates(csv_path: Path, operator_col: str = 'operator_initials') -> int:
    """
    Adjust PS dates in a CSV file back by one day.

    Args:
        csv_path: Path to the CSV file
        operator_col: Name of the column containing operator initials

    Returns:
        Number of rows adjusted
    """
    df = pd.read_csv(csv_path)

    # Find PS rows
    ps_mask = df[operator_col] == 'PS'
    n_adjusted = ps_mask.sum()

    if n_adjusted == 0:
        print(f"  No PS rows found in {csv_path.name}")
        return 0

    # Adjust dates for PS rows
    for idx in df[ps_mask].index:
        year, month, day = adjust_date_back_one_day(
            df.loc[idx, 'year'],
            df.loc[idx, 'month'],
            df.loc[idx, 'day']
        )
        df.loc[idx, 'year'] = year
        df.loc[idx, 'month'] = month
        df.loc[idx, 'day'] = day

    # Save back
    df.to_csv(csv_path, index=False)
    print(f"  Adjusted {n_adjusted} PS rows in {csv_path.name}")
    return n_adjusted


def adjust_geojson_ps_dates(geojson_path: Path, operator_col: str = 'operator_initials') -> int:
    """
    Adjust PS dates in a GeoJSON file back by one day.

    Args:
        geojson_path: Path to the GeoJSON file
        operator_col: Name of the property containing operator initials

    Returns:
        Number of features adjusted
    """
    gdf = gpd.read_file(geojson_path)

    # Find PS rows
    ps_mask = gdf[operator_col] == 'PS'
    n_adjusted = ps_mask.sum()

    if n_adjusted == 0:
        print(f"  No PS features found in {geojson_path.name}")
        return 0

    # Adjust dates for PS rows
    for idx in gdf[ps_mask].index:
        year, month, day = adjust_date_back_one_day(
            gdf.loc[idx, 'year'],
            gdf.loc[idx, 'month'],
            gdf.loc[idx, 'day']
        )
        gdf.loc[idx, 'year'] = year
        gdf.loc[idx, 'month'] = month
        gdf.loc[idx, 'day'] = day

    # Save back
    gdf.to_file(geojson_path, driver='GeoJSON')
    print(f"  Adjusted {n_adjusted} PS features in {geojson_path.name}")
    return n_adjusted


def main():
    parser = argparse.ArgumentParser(
        description='Adjust PS annotation dates back by one day'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data/labels/labeled_surveys/random_sample'),
        help='Directory containing the master dataset files'
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return 1

    print(f"Adjusting PS dates in {data_dir}")
    print()

    total_adjusted = 0

    # Adjust CSV files
    csv_files = [
        'latest_irrigation_table.csv',
        'latest_polygons_table.csv',
    ]

    for csv_name in csv_files:
        csv_path = data_dir / csv_name
        if csv_path.exists():
            total_adjusted += adjust_csv_ps_dates(csv_path)
        else:
            print(f"  Warning: {csv_name} not found")

    # Adjust GeoJSON files
    geojson_files = [
        'latest_irrigation_data.geojson',
        'latest_polygons.geojson',
    ]

    for geojson_name in geojson_files:
        geojson_path = data_dir / geojson_name
        if geojson_path.exists():
            total_adjusted += adjust_geojson_ps_dates(geojson_path)
        else:
            print(f"  Warning: {geojson_name} not found")

    print()
    print(f"Done! Total rows/features adjusted: {total_adjusted}")
    return 0


if __name__ == '__main__':
    exit(main())
