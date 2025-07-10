import xml.etree.ElementTree as ET
import json
import geopandas as gpd
import os

# Define the KML namespace
ns = {'kml': 'http://www.opengis.net/kml/2.2'}

def parse_name(name_text):
    """
    Parse the name string (e.g., "AB_3_9.6.2021") into its parts.
    Returns a dict with operator_initials, internal_id, month, day, and year.
    """
    try:
        parts = name_text.split('_')
        operator_initials = parts[0]
        internal_id = int(parts[1])
        # Expecting the date to be in the format month.day.year (e.g., "9.6.2021")
        date_parts = parts[2].split('.')
        if len(date_parts) != 3:
            print(f"Issue parsing name '{name_text}': Date portion does not have three parts. Returning the whole thing for the month, day and year for manual fixing.")
            month, day, year = name_text, name_text, name_text
        else:
            # Convert month, day, year to integers
            month, day, year = map(int, date_parts)
            # Check if year is 2 or 4 digits
            if len(str(year)) != 4:
                if len(str(year)) == 2:
                    print(f"Warning: internal_id {internal_id}; Year '{year}' is only two digits. Adding '20' prefix.")
                    year += 2000
                else:
                    print(f"Warning: internal_id {internal_id}; Year '{year}' is not 2 or four digits. Please manually fix")
        return {
            "operator_initials": operator_initials,
            "internal_id": internal_id,
            "month": month,
            "day": day,
            "year": year
        }
    except Exception as e:
        raise ValueError(f"Error parsing name '{name_text}': {e}. Name properties not returned.")

def parse_description(desc_text): # Note you will need to update to handle special classes (agroforestry etc.)
    """
    Parse the description text into certainty, uncertainty_explanation, and special_category.
    Expects the first line to be certainty (default to 5 if empty)
    and the second line to be uncertainty_explanation.
    Adds a special_category field containing one or more flag group names separated by semicolons.
    """
    # Split lines and remove empty lines
    lines = [line.strip().lower() for line in desc_text.strip().splitlines() if line.strip()]

    # If there is nothing in the description, assume certainty 5
    if not lines:
        return {"certainty": 5, "uncertainty_explanation": "", "special_category": ""}
    
    # If the first line is not an integer, assume certainty 5
    try:
        certainty = int(lines[0]) if lines[0] else 5
    except ValueError:
        # If conversion fails, default to 5
        certainty = 5

    flag_groups = {
        "plantation": ["agroforestry", "plantation"],
        "industrial": ["industrial", "commercial"],
        "lawn": ["lawn"],
        "covered": ["covered"]
    }

    all_flag_keywords = [
        keyword 
        for flag_list in flag_groups.values() 
        for keyword in flag_list
    ]

    # For lines 2 and onward, any line that does not contain a special class flag is added to the explanation
    explanation_lines = [
        line for line in lines[1:]
        if not any(keyword in line for keyword in all_flag_keywords)
    ]
    explanation = "; ".join(explanation_lines)

    # Collect special categories present in any line
    special_categories = []
    for flag_name, flag_keywords in flag_groups.items():
        if any(keyword in line for keyword in flag_keywords for line in lines):
            special_categories.append(flag_name)
    special_category_str = ";".join(special_categories)

    return {
        "certainty": certainty,
        "uncertainty_explanation": explanation,
        "special_category": special_category_str
    }

def convert_geometry(placemark):
    """
    Converts a KML geometry element (Point, LineString, or Polygon) to a GeoJSON geometry dict.
    """
    # Check for Point
    point = placemark.find("kml:Point", ns)
    if point is not None:
        print ("Warning: Point found in kml. Passing over and not converting to GeoJSON")
        return None
    
    # Check for LineString
    linestring = placemark.find("kml:LineString", ns)
    if linestring is not None:
        print ("Warning: LineString found in kml. Passing over and not converting to GeoJSON")
        return None

    # Check for Polygon (only handling the outer boundary)
    polygon = placemark.find("kml:Polygon", ns)
    if polygon is not None:
        outer = polygon.find("kml:outerBoundaryIs/kml:LinearRing", ns)
        if outer is None:
            raise ValueError("Polygon without an outerBoundaryIs/LinearRing element")
        coords_text = outer.find("kml:coordinates", ns).text.strip()
        coords = []
        for coord in coords_text.split():
            parts = coord.split(',')
            lon, lat = float(parts[0]), float(parts[1])
            coords.append([lon, lat])
        # GeoJSON expects polygons as a list of linear rings.
        return {"type": "Polygon", "coordinates": [coords]}
    
    # If no supported geometry is found, return None.
    return None

def kml_to_geojson(kml_file):
    """
    Converts a KML file that contains a folder of polygons exported from Google 
    Earth Pro to a GeoJSON file and returns a GeoPandas GeoDataFrame.
    This function parses a KML file, extracts placemark data, converts the geometries 
    to GeoJSON format, and writes the resulting GeoJSON to a file. It also returns 
    a GeoPandas GeoDataFrame created from the GeoJSON features.
    Args:
        kml_file (str): The file path to the input KML file.
    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the features from the 
        converted GeoJSON file.
    Notes:
        - The function expects the KML file to have placemarks with <name>, 
          <description>, and geometry elements.
        - The <name> element is parsed to extract properties using the `parse_name` function.
        - The <description> element is parsed to extract additional properties using 
          the `parse_description` function.
        - If a placemark lacks a supported geometry, it is skipped.
        - The resulting GeoJSON file is saved in the same directory as the input KML file, 
          with the same name but a `.geojson` extension.
    Raises:
        ValueError: If the <name> element cannot be parsed by `parse_name`.
    Example:
        >>> gdf = kml_to_geojson("example.kml")
        GeoJSON written to example.geojson
        >>> print(gdf.head())
    """
    

    tree = ET.parse(kml_file)
    root = tree.getroot()

    features = []

    # Iterate over each Placemark in the KML
    for placemark in root.findall(".//kml:Placemark", ns):
        # Extract and parse <name>
        name_elem = placemark.find("kml:name", ns)
        if name_elem is None or not name_elem.text:
            continue  # Skip placemarks without a name
        try:
            props = parse_name(name_elem.text.strip())
        except ValueError as e:
            print(e)
            props = {}

        # Extract and parse <description>
        desc_elem = placemark.find("kml:description", ns)
        if desc_elem is not None and desc_elem.text:
            desc_props = parse_description(desc_elem.text)
        else:
            desc_props = {"certainty": 5, "uncertainty_explanation": ""}

        # Merge properties
        properties = {"name": name_elem.text, **props, **desc_props}

        # Convert the geometry
        geometry = convert_geometry(placemark)
        if geometry is None:
            print(f"No supported geometry found for placemark {name_elem.text}")
            continue

        # Build a GeoJSON feature
        feature = {
            "type": "Feature",
            "properties": properties,
            "geometry": geometry
        }
        features.append(feature)

    # Build the FeatureCollection
    feature_collection = {
        "type": "FeatureCollection",
        "features": features
    }

    # Write the GeoJSON to a file
    processed_folder = os.path.dirname(kml_file).replace("/raw", "/processed")
    os.makedirs(processed_folder, exist_ok=True)
    geojson_file = os.path.join(processed_folder, os.path.basename(kml_file).replace(".kml", ".geojson"))

    with open(geojson_file, "w") as f:
        json.dump(feature_collection, f, indent=2)
    print(f"GeoJSON written to {geojson_file}")

    gdf = gpd.GeoDataFrame.from_features(feature_collection["features"])
    return gdf

# Example usage:
if __name__ == "__main__":

    # Example usage/test code

    # kml = "data/labels/labeled_surveys/random_sample/raw/AB_JL_101-125.kml"
    # gdf = kml_to_geojson(kml)
    # print(gdf.head)

    import argparse

    parser = argparse.ArgumentParser(description="Convert KML to GeoJSON.")
    parser.add_argument("kml_file", type=str, help="Path to the KML file to convert.")
    args = parser.parse_args()
    
    kml_file = args.kml_file
    gdf = kml_to_geojson(kml_file)
