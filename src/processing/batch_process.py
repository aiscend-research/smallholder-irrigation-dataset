import os
import sys
import pandas as pd
from survey_to_csv import process_xml_zip
from polygons_to_geojson import kml_to_geojson
from merge_survey_and_polygons import merge_and_check
from src.utils.utils import generate_latest_irrigation_data, save_data
from src.utils.geometries import bounding_box
import geopandas as gpd
from shapely.geometry import box


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
    # Add a unique_id as the first column
    latest_irrigation_data.insert(0, 'unique_id', pd.Series(range(1, len(latest_irrigation_data) + 1), index=latest_irrigation_data.index))
    # Save the pandas df as a csv in the labels folder as "latest_irrigation_table.csv"
    csv_path = f"labels/labeled_surveys/{group_name}/latest_irrigation_table.csv"
    description = "The latest labeled irrigation data"
    save_data(latest_irrigation_data, csv_path, description=description, file_format="csv")
    # Generate bounding boxes as Shapely geometries for each row
    latest_irrigation_data['geometry'] = latest_irrigation_data.apply(
        lambda row: box(*bounding_box(row['y'], row['x'], half_side_km=0.5)), axis=1
    )
    # Convert the DataFrame to a GeoDataFrame
    latest_irrigation_data_gdf = gpd.GeoDataFrame(latest_irrigation_data, geometry='geometry', crs="EPSG:4326")
    # Save the GeoDataFrame to a GeoJSON file
    geojson_path = f"labels/labeled_surveys/{group_name}/latest_irrigation_data.geojson"
    description = "The latest labeled irrigation data with a bounding box"
    save_data(latest_irrigation_data_gdf, geojson_path, description=description, file_format="json")


if __name__ == '__main__':
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