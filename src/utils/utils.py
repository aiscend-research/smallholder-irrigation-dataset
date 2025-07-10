import os
import yaml
import json
import pandas as pd
import pickle
from datetime import datetime
import geopandas as gpd
import rasterio
import inspect
import re

# Helper function to find the project root
def find_project_root(current_path):
    """
    Recursively find the project root by locating the config.yaml file
    and ensuring the directory is named 'smallholder-irrigation-dataset'.

    Parameters:
        current_path (str): The starting path to search upwards from.

    Returns:
        str: The absolute path to the project root directory.
    """
    while current_path != os.path.dirname(current_path):
        if ("config.yaml" in os.listdir(current_path) and 
            os.path.basename(current_path) == "smallholder-irrigation-dataset"):
            return current_path
        current_path = os.path.dirname(current_path)
    raise FileNotFoundError("Could not find 'smallholder-irrigation-dataset' directory with config.yaml in any parent directory.")

# Load configuration
def load_config():
    """
    Load project configuration from the project root directory.

    Returns:
        dict: Configuration settings.
    """
    current_dir = os.getcwd()
    project_root = find_project_root(current_dir)
    config_path = os.path.join(project_root, "config.yaml")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Determine the root data directory
def get_data_root():
    """
    Determine whether the code is running locally or on a server
    and return the appropriate data root directory.

    Returns:
        str: Path to the data root directory.
    """
    config = load_config()
    local_root = os.path.join(find_project_root(os.path.abspath("")), 'data')
    server_root = config.get('server_data_root', '/home/waves/data/smallholder-irrigation-dataset/data/')

    # Check for server environment
    if os.path.exists(server_root):
        return server_root
    else:
        return local_root

# Save data with metadata
def save_data(data, output_path, description=None, file_format=None):
    """
    Save data to the specified output path in a flexible format, creating directories as needed,
    and optionally save metadata.

    Parameters:
        data (any): Data to be saved (supports JSON, CSV, Pickle, YAML).
        output_path (str): Path where the data should be saved.
        description (str, optional): Description of the data.
        file_format (str, optional): Format to save the data (json, csv, pickle, yaml). Inferred from file extension if not provided.
    """
    output_path = get_data_root() + "/" + output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Infer file format from extension if not provided
    if not file_format:
        file_format = output_path.split('.')[-1].lower()

    # Save data based on the format
    if file_format == 'json':
        # Check if the data is a GeoDataFrame
        if isinstance(data, gpd.GeoDataFrame):
            data.to_file(output_path, driver='GeoJSON')  # Save as GeoJSON
        else:
            with open(output_path, "w") as f:
                json.dump(data, f)
    elif file_format == 'csv':
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        else:
            raise ValueError("Data must be a pandas DataFrame to save as CSV.")
    elif file_format == 'pickle':
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
    elif file_format == 'yaml':
        with open(output_path, "w") as f:
            yaml.dump(data, f)
    elif file_format == 'tif':
        with rasterio.open(output_path, 'w', **data.meta) as dst:
            dst.write(data.read())
    elif file_format == 'png':
        if hasattr(data, 'savefig'):
            data.savefig(output_path)
        else:
            raise ValueError("Data must be a Matplotlib figure to save as PNG.")
    else:
        raise ValueError(f"Unsupported file format: {file_format}")

    # Automatically generate metadata

    # Automatically determine the **calling script** instead of utils.py
    caller_frame = inspect.stack()[1]  # Get the frame of the function that called save_data()
    caller_script = os.path.abspath(caller_frame.filename)

    metadata = {
        "date": datetime.now().isoformat(),
        "file": os.path.basename(output_path),
        "description": description,
        "file_format": file_format,
        "source": os.path.relpath(caller_script, start=find_project_root(os.getcwd()))  # Captures the file that created the data
    }

    # Save metadata alongside the data file
    metadata_path = output_path.rsplit('.', 1)[0] + "_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

# Generate the latest irrigation data from completed surveys
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

    # Manually mark 'AB_JL_101-125' as the most recent survey
    df.loc[df['source_file'] == 'AB_JL_101-125', 'most_recent'] = 1

    # Filter the DataFrame to keep only the most recent surveys
    df = df[df['most_recent'] == 1]

    # Exclude surveys with 'MV_76-100' in the source file name
    df = df[~df['source_file'].str.contains('MV_76-100')]

    return df