# This script takes in a survey response file that you download from Earth Collect and turns it into a usable csv. 
# It expects the file name to start with the operator initials and end with the range of survey locations, separated by underscores. There can be anything else in between. @

import os
import xml.etree.ElementTree as ET
import pandas as pd
import shutil

def parse_xml(file_path, original_location_file=None):
    """
    Parses an XML file to extract site-level and day-level irrigation data.
    Args:
        file_path (str): The file path to the XML file to be parsed.
    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains the following keys:
            - site_id (str or None): The site ID extracted from the XML.
            - internal_id (int): The internal ID derived from the filename (without extension) or by crosswalking with id file.
            - plot_file (str or None): The plot file value from the XML.
            - operator (str or None): The operator value from the XML.
            - operator_initials (str): The initials of the operator, derived from the file path.
            - x (str or None): The x-coordinate of the location from the XML.
            - y (str or None): The y-coordinate of the location from the XML.
            - water_source (str or None): The water source value from the XML.
            - image_number (int): The image number (1 to 10) corresponding to the day record.
            - year (str or None): The year value for the day record.
            - month (str or None): The month value for the day record.
            - day (str or None): The day value for the day record.
            - irrigation (str or None): The irrigation value for the day record.
    Notes:
        - The function assumes that the XML file contains up to 10 day records, each with fields
          such as year, month, day, and irrigation.
        - If a field is missing or not found in the XML, its value will be set to None.
        - The operator initials are extracted from the file path by splitting the directory name
          preceding the file name.
    """

    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Extract site-level information
    operator_initials = str.split(str.split(file_path, "/")[-3], "_")[0]
    site_id = root.find("id/value").text if root.find("id/value") is not None else None
    x = root.find("location/x").text if root.find("location/x") is not None else None
    y = root.find("location/y").text if root.find("location/y") is not None else None
    operator = root.find("operator/value").text if root.find("operator/value") is not None else None
    plot_file = root.find("plot_file/value").text if root.find("plot_file/value") is not None else None
    water_source = root.find("natural_dicoloration/value").text if root.find("natural_dicoloration/value") is not None else None

    if original_location_file:
        # the original location file lists the ids in order. 
        # We can use the index of the ids (index + 1) to get the internal id
        
        ids_cross = pd.read_csv(original_location_file)
        ids_cross = ids_cross[ids_cross["id"] == site_id]
        if ids_cross.empty:
            # print(f"Warning: {site_id} not found in original location file. Removing this line from the output.")
            return []
        else:
            internal_id = ids_cross.index[0] + 1
    else:
        internal_id = int(os.path.splitext(os.path.basename(file_path))[0]) # The filename without extension

    records = []
    # Iterate over potential day records (assuming up to 10)
    for i in range(1, 11):
        year_elem = root.find(f"year{i}")
        month_elem = root.find(f"month{i}")
        day_elem = root.find(f"day{i}")
        irrigation_elem = root.find(f"irrigation{i}")

        if year_elem is not None and year_elem.find("code") is not None:
            year = year_elem.find("code").text
            month = month_elem.find("code").text if (month_elem is not None and month_elem.find("code") is not None) else None
            day = day_elem.find("value").text if (day_elem is not None and day_elem.find("value") is not None) else None
            irrigation = irrigation_elem.find("code").text if (irrigation_elem is not None and irrigation_elem.find("code") is not None) else None

            # Turn the year month and day into a date object to check its validity
            try:
                date = pd.to_datetime(f"{year}-{month}-{day}", format="%Y-%m-%d")
            except ValueError:
                date = None
            
            # Only create a row if the date is valid and there is an irrigation code
            if date and irrigation:

                records.append({
                    "site_id": site_id,
                    "internal_id": internal_id, 
                    "plot_file": plot_file,
                    "operator": operator,
                    "operator_initials": operator_initials,
                    "x": x,
                    "y": y,
                    "water_source": water_source,
                    "image_number": i,
                    "year": year,
                    "month": month,
                    "day": day,
                    "irrigation": irrigation,
                })
    return records

def process_xml_zip(xml_zip, original_location_file=None):
    """
    Processes a ZIP file containing XML files, extracts the data, and converts it into a CSV file.
    Args:
        xml_zip (str): The file path to the ZIP file containing XML files.
    Returns:
        list: A list of all records extracted from the XML files.
    Functionality:
        1. Unzips the provided ZIP file into a folder.
        2. Assumes the XML files are located in a subfolder named "1" within the extracted folder.
        3. Iterates through all XML files in the folder, parsing their contents.
        4. Combines the parsed data into a pandas DataFrame.
        5. Exports the DataFrame to a CSV file in the same directory as the extracted folder.
        6. Prints the location of the generated CSV file.
    """

    # Unzip the folder
    xml_folder = os.path.splitext(xml_zip)[0]
    shutil.unpack_archive(xml_zip, xml_folder)

    xml_folder = xml_folder + '/1' # move into the "1" folder

    # Automatically generate the original_location_file path based on the xml_zip name
    if original_location_file is None:
        group_name = xml_zip.split("/")[-3]
        sample_range = os.path.basename(xml_zip).split("_")[-1].replace(".zip", "")
        original_location_file = f"data/sampling/samples/{group_name}/Zambia_0.05_n_{sample_range}.csv"

    all_records = []
    for filename in os.listdir(xml_folder):
        if filename.endswith(".xml"):
            file_path = os.path.join(xml_folder, filename)
            all_records.extend(parse_xml(file_path, original_location_file))

    # Create a DataFrame and export to CSV
    df = pd.DataFrame(all_records)
    processed_folder = os.path.join(os.path.dirname(xml_zip), "processed")
    os.makedirs(processed_folder, exist_ok=True)
    output_csv = xml_zip.replace("/raw/", "/processed/").replace(".zip", ".csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"CSV successfully created at: {output_csv}")
    
    return df

if __name__ == '__main__':

    # Example usage/test code

    # xml_zip = "data/labels/labeled_surveys/random_sample/raw/JL_400-424.zip"
    # df = process_xml_zip(xml_zip)

    # CLI argument parsing
    import argparse

    parser = argparse.ArgumentParser(description="Process a survey ZIP file exported from Earth Collect into a CSV.")
    parser.add_argument("zip_path", help="Path to the .zip file containing the XML survey export")

    args = parser.parse_args()

    df = process_xml_zip(args.zip_path)
    print(f"Parsed {len(df)} records from {args.zip_path}")