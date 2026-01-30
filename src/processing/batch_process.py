import os
import sys
import pandas as pd
import re
from datetime import datetime, timedelta
from survey_to_csv import process_xml_zip
import geopandas as gpd
from shapely.geometry import box
from polygons_to_geojson import kml_to_geojson
from merge_survey_and_polygons import merge_and_check
import glob

import sys
import os

# Add the project root to the system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.utils import save_data, get_data_root
from src.utils.geometries import bounding_box, survey_polygon


def adjust_ps_dates(df, operator_col='operator_initials'):
    """
    Adjust PS (Peter Siame) annotation dates back by one day.

    Peter's annotations were recorded with dates one day later than the actual
    image dates due to timezone differences during labeling (Zambia UTC+2 vs
    California UTC-7/8, meaning when Peter labeled an image late at night in
    Zambia, the date had already rolled over to the next day).

    Args:
        df: DataFrame with year, month, day, and operator columns
        operator_col: Name of the column containing operator initials

    Returns:
        DataFrame with adjusted dates for PS rows
    """
    df = df.copy()
    ps_mask = df[operator_col] == 'PS'
    n_adjusted = ps_mask.sum()

    if n_adjusted == 0:
        return df

    for idx in df[ps_mask].index:
        date = datetime(int(df.loc[idx, 'year']), int(df.loc[idx, 'month']), int(df.loc[idx, 'day']))
        new_date = date - timedelta(days=1)
        df.loc[idx, 'year'] = new_date.year
        df.loc[idx, 'month'] = new_date.month
        df.loc[idx, 'day'] = new_date.day

    print(f"  Adjusted {n_adjusted} PS rows back by one day")
    return df


def generate_latest_irrigation_data(group_name="random_sample"):
    """
    Generate the latest irrigation data by merging labeled survey files,
    identifying the most recent surveys, and filtering the data accordingly.

    Returns:
        pd.DataFrame: A DataFrame containing the most recent irrigation data.
    """

    # Define the folder containing merged survey files
    merged_folder = os.path.join(get_data_root(), f"labels/labeled_surveys/{group_name}/merged")

    # List all files in the merged folder
    files = os.listdir(merged_folder)

    # Read and merge all CSV files into a single DataFrame
    df = pd.concat(
        [pd.read_csv(os.path.join(merged_folder, file)) for file in files if file.endswith('.csv')],
        ignore_index=True
    )

    # Identify the most recent source file for each survey (plot_file)
    survey_files = df['plot_file'].unique()
    most_recent_source = []

    for survey_file in survey_files:
        # Get all source files associated with the current survey file
        source_files = df[df['plot_file'] == survey_file]['source_file'].unique().tolist()

        # Determine the most recent source file based on priority
        corrected_v2 = [s for s in source_files if re.match(r'^[A-Z]+_[A-Z]+_v2', s)]
        corrected = [s for s in source_files if re.match(r'^[A-Z]+_[A-Z]+_', s) and 'v2' not in s]
        uncorrected_v2 = [s for s in source_files if re.match(r'^[A-Z]+_v2', s)]

        if corrected_v2:
            most_recent_source.append(corrected_v2)
        elif corrected:
            most_recent_source.append(corrected)
        elif uncorrected_v2:
            most_recent_source.append(uncorrected_v2)
        elif source_files:
            most_recent_source.append(source_files)
        else:
            most_recent_source.append(None)  # Handle cases with no source files

    # Add a column to indicate if the source file is the most recent for its survey
    df['most_recent'] = df.apply(
        lambda x: 1 if x['source_file'] in most_recent_source[survey_files.tolist().index(x['plot_file'])] else 0,
        axis=1
    )

    # Manually mark certain surveys as the most recent
    # (these don't follow the logic above but we still want them since they are the most recent for these labelers)
    # - AB_JL_101-125, PS_101-125: QC surveys that overlap with other labelers
    # - PS_1025-1049: PS survey that overlaps with KL_v2_1025-1049 (different labelers, both should be included)
    df.loc[df['source_file'].isin(['AB_JL_101-125', 'PS_101-125', 'PS_1025-1049']), 'most_recent'] = 1

    # Filter the DataFrame to keep only the most recent surveys
    df = df[df['most_recent'] == 1]

    # Exclude surveys with 'MV_76-100' in the source file name
    # This survey was never corrected due to a read issue
    df = df[~df['source_file'].str.contains('MV_76-100')]

    return df


def process_and_merge_folder(folder_path):
    """
    Processes and merges all raw survey matching polygon files in a specified folder.
    This function iterates through all files in the given folder, processes
    `.kml` and `.zip` files using specific helper functions, and merges the
    resulting processed `.csv` files into a single DataFrame.
    Args:
        folder_path (str): The path to the folder containing the files to process.
    Returns:
        pandas.DataFrame: A DataFrame containing the merged results of all processed `.csv` files.
    Notes:
        - `.kml` files are converted to GeoJSON using the `kml_to_geojson` function.
        - `.zip` files are processed using the `process_xml_zip` function.
        - The processed files are expected to be stored in a subfolder named "processed".
        - Only `.csv` files in the "processed" folder are merged.
    Raises:
        FileNotFoundError: If the specified folder or required files do not exist.
        ValueError: If there are issues during the merging process.
    """
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith('.kml'):
            kml_to_geojson(file_path)
        elif file_name.endswith('.zip'):
            process_xml_zip(file_path)

    # Merge the processed files
    processed_path = folder_path.replace("/raw", "/processed")
    merged_result = [merge_and_check(os.path.join(processed_path, file)) for file in os.listdir(processed_path) if file.endswith('.csv')]
    merged_result = pd.concat(merged_result, ignore_index=True)
    return merged_result


def pool_latest_labels_and_save(group_name="random_sample"):
    """
    Pools the latest labeled irrigation data for the specified group, saves as CSV and GeoJSON with bounding boxes.
    """
    latest_irrigation_data = generate_latest_irrigation_data(group_name)

    # Ensure it's a DataFrame
    if not isinstance(latest_irrigation_data, pd.DataFrame):
        latest_irrigation_data = pd.DataFrame(latest_irrigation_data)

    # Adjust PS dates back by one day (timezone correction)
    latest_irrigation_data = adjust_ps_dates(latest_irrigation_data)

    # Add a unique_id as the first column
    latest_irrigation_data.insert(0, 'unique_id', pd.Series(range(1, len(latest_irrigation_data) + 1), index=latest_irrigation_data.index))
    
    # Save the pandas df as a csv in the labels folder as "latest_irrigation_table.csv"
    csv_path = f"labels/labeled_surveys/{group_name}/latest_irrigation_table.csv"
    description = "The latest labeled irrigation data"
    save_data(latest_irrigation_data, csv_path, description=description, file_format="csv")
    
    # Generate bounding boxes as Shapely geometries for each row
    latest_irrigation_data['geometry'] = latest_irrigation_data.apply(survey_polygon, axis=1)
    
    # Convert the DataFrame to a GeoDataFrame
    latest_irrigation_data_gdf = gpd.GeoDataFrame(latest_irrigation_data, geometry='geometry', crs="EPSG:4326")
    
    # Save the GeoDataFrame to a GeoJSON file
    geojson_path = f"labels/labeled_surveys/{group_name}/latest_irrigation_data.geojson"
    description = "The latest labeled irrigation data with a bounding box"
    save_data(latest_irrigation_data_gdf, geojson_path, description=description, file_format="json")


def pool_latest_polygons_and_save(group_name="random_sample"):
    """
    Merges all enriched polygon GeoJSONs in the merged folder, filters to only those with 'source_file' in the latest irrigation table,
    and saves as a single CSV and GeoJSON.
    """
    # 1. Load the latest irrigation table to get the unique source_file values
    latest_irrigation_table_path = f"data/labels/labeled_surveys/{group_name}/latest_irrigation_table.csv"
    latest_irrigation_df = pd.read_csv(latest_irrigation_table_path)
    latest_source_files = set(latest_irrigation_df['source_file'])

    # 2. Find all merged polygon GeoJSONs in the merged folder
    merged_folder = f"data/labels/labeled_surveys/{group_name}/merged"
    polygon_files = glob.glob(f"{merged_folder}/*_polygons.geojson")

    # 3. Read and concatenate all polygon GeoJSONs
    all_polygons = gpd.GeoDataFrame(pd.concat(
        [gpd.read_file(f) for f in polygon_files],
        ignore_index=True
    ), crs="EPSG:4326")

    # 4. Filter to only polygons from the most recent surveys (by source_file)
    filtered_polygons = all_polygons[all_polygons['source_file'].isin(list(latest_source_files))]

    # 5. Adjust PS dates back by one day (timezone correction)
    filtered_polygons = gpd.GeoDataFrame(adjust_ps_dates(filtered_polygons), crs="EPSG:4326")

    # 6. Save as CSV and GeoJSON using save_data
    csv_path = f"labels/labeled_surveys/{group_name}/latest_polygons_table.csv"
    geojson_path = f"labels/labeled_surveys/{group_name}/latest_polygons.geojson"
    description_csv = "The latest labeled irrigation polygons (CSV) for the most recent surveys."
    description_geojson = "The latest labeled irrigation polygons (GeoJSON) for the most recent surveys."
    save_data(filtered_polygons.drop(columns='geometry'), csv_path, description=description_csv, file_format="csv")
    save_data(filtered_polygons, geojson_path, description=description_geojson, file_format="json")
    print(f"Saved merged latest polygons CSV at {csv_path}")
    print(f"Saved merged latest polygons GeoJSON at {geojson_path}")


if __name__ == '__main__':

    # test code
    # folder_path = "data/labels/labeled_surveys/random_sample/raw/"
    # process_and_merge_folder(folder_path)
    # group_name = "random_sample"
    # pool_latest_polygons_and_save(group_name)

    import argparse
    
    parser = argparse.ArgumentParser(description="Process and merge all survey files in a folder, then pool the latest labeled irrigation data and output both a CSV and a GeoJSON file with bounding boxes.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing survey files.")
    parser.add_argument("--group_name", type=str, default="random_sample", help="Name of the group/sample set (default: random_sample)")
    args = parser.parse_args()
    folder_path = args.folder_path
    group_name = args.group_name

    merged_result = process_and_merge_folder(folder_path)
    print(f"Merged result has {len(merged_result)} rows")

    pool_latest_labels_and_save(group_name)
    print(f"Pooled latest labeled irrigation data for group '{group_name}' and saved CSV and GeoJSON.")

    pool_latest_polygons_and_save(group_name)
    print(f"Merged latest polygons for group '{group_name}' and saved CSV and GeoJSON.")