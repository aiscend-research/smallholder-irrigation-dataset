import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from typing import Optional
from shapely.validation import make_valid
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add src to the path so utils can be found
from utils.geometries import survey_polygon
from polygons_to_geojson import CATEGORIES

def check_irrigation_polygon_consistency(row, matching_polys, irrigation, idx):
    """
    Checks consistency between irrigation value and matching polygons for a survey row.
    Returns a list of report lines (strings) describing any inconsistencies found.
    """
    issues = []
    # Check 1: If irrigation == 1, there should be no polygons.
    if irrigation == 1 and not matching_polys.empty:
        issues.append(f"Row {idx} (internal_id {row['internal_id']}, {row['day']}/{row['month']}/{row['year']}): survey marked irrigation as 1 (no irrigation) but found {len(matching_polys)} matching polygon(s).")
    # Check 2: If irrigation > 1, there should be at least one polygon.
    if irrigation is not None and irrigation > 1 and matching_polys.empty:
        issues.append(f"Row {idx} (internal_id {row['internal_id']}, {row['day']}/{row['month']}/{row['year']}): survey marked irrigation {irrigation} (possible irrigation) but no matching polygons found.")
    # Check 3: If irrigation == 5, at least one polygon should have certainty == 5.
    if irrigation == 5:
        if matching_polys.empty or (matching_polys["certainty"].max() < 5):
            issues.append(f"Row {idx} (internal_id {row['internal_id']}, {row['day']}/{row['month']}/{row['year']}): survey marked irrigation 5 (definitely irrgation) but no polygon with certainty 5 found.")
    # Check 4: If irrigation == 2, 3, or 4, all matched polygons should have certainty <= 4.
    if irrigation in [2, 3, 4]:
        if not matching_polys.empty and matching_polys["certainty"].max() > 4:
            issues.append(f"Row {idx} (internal_id {row['internal_id']}, {row['day']}/{row['month']}/{row['year']}): survey marked irrigation {irrigation} (uncertain) but found a polygon with certainty 5 (certain).")
    return issues

def process_survey_row(row, polygons, certainty_cutoff, idx):
    """
    Processes a single survey row: matches polygons, computes coverage and stats, and returns results and report lines.
    Returns (result_dict, report_lines), where the results include: 
        - percent_coverage: Percentage (%) of the survey area covered by polygons.
        - percent_coverage_hc: Percentage (%) of the survey area covered by polygons with certainty >= certainty_cutoff.
        - poly_avg_size: Average size of the polygons covering the survey area (square meters).
        - poly_avg_size_hc: Average size of the polygons with certainty >= the certainty_cutoff (square meters).
        - poly_min_size: Minimum size of the polygons covering the survey area (square meters).
        - poly_min_size_hc: Minimum size of the polygons with certainty >= the certainty_cutoff (square meters).
        - percent_coverage_hc_{category}: Percent (%) coverage of high-certainty polygons with a specific category.
    """
    # Initialize result dict with default values and consistent names
    result = {
        "percent_coverage": 0.0,
        "percent_coverage_hc": 0.0,
        "poly_avg_size": None,
        "poly_avg_size_hc": None,
        "poly_min_size": None,
        "poly_min_size_hc": None,
    }
    for category in CATEGORIES:
        result[f"percent_coverage_hc_{category}"] = 0.0

    # Find polygons that match by internal_id (or site_id if the labeler accidentally used that), year, month, and day.
    condition = (
        ((polygons["internal_id"] == row["internal_id"]) | 
         (polygons["internal_id"] == int(row["site_id"][3:]))) &
        (polygons["year"] == row["year"]) &
        (polygons["month"] == row["month"]) &
        (polygons["day"] == row["day"])
    )
    matching_polys = polygons[condition].copy()

    # Get irrigation value and perform checks.
    irrigation = int(row["irrigation"])

    # Consistency checks between irrigation and polygons
    report_lines = check_irrigation_polygon_consistency(row, matching_polys, irrigation, idx)

    # Add the site_id to matched polygons
    if not matching_polys.empty:
        polygons.loc[condition, "site_id"] = row["site_id"]

    # Compute percent coverage and related stats if there are matching polygons
    survey_area = row["geometry"].area
    if not matching_polys.empty:
        # Clean the geometries to ensure they are valid
        matching_polys.geometry = [make_valid(geom) for geom in matching_polys.geometry]

        # Check that all polygons are at least partially overlapping the survey area
        for poly_idx, poly in matching_polys.iterrows():
            if not row["geometry"].intersects(poly["geometry"]):
                report_lines.append(f"Polygon {poly_idx} (internal_id {poly['internal_id']}, {poly['day']}/{poly['month']}/{poly['year']}) does not overlap the survey area.")
        
        # Calculate the average and min size of the polygons in square meters (use local CRS)
        result["poly_avg_size"] = matching_polys.to_crs("EPSG:32735").geometry.area.mean()
        result["poly_min_size"] = matching_polys.to_crs("EPSG:32735").geometry.area.min()
        
        # Calculate the overlap
        union_all = unary_union(matching_polys.geometry)
        intersection_all = row["geometry"].intersection(union_all)
        result["percent_coverage"] = (intersection_all.area / survey_area) * 100 if survey_area > 0 else 0.0

        # For high-certainty coverage, filter for certainty >= certainty_cutoff.
        high_polys = matching_polys[matching_polys["certainty"] >= certainty_cutoff]

        # Calculate the average and min size of the high-certainty polygons in square meters (use local CRS)
        if not high_polys.empty:
            result["poly_avg_size_hc"] = high_polys.to_crs("EPSG:32735").geometry.area.mean()
            result["poly_min_size_hc"] = high_polys.to_crs("EPSG:32735").geometry.area.min()
            # Calculate the coverage/overlap for high-certainty polygons
            union_high = unary_union(high_polys.geometry.tolist())
            intersection_high = row["geometry"].intersection(union_high)
            result["percent_coverage_hc"] = (intersection_high.area / survey_area) * 100 if survey_area > 0 else 0.0

            # For each special category, calculate percent coverage
            for category in CATEGORIES:
                special_polys = high_polys[high_polys["category"].astype(str).str.contains(category, case=False, na=False)]
                if not special_polys.empty:
                    union_special = unary_union(special_polys.geometry.tolist())
                    intersection_special = row["geometry"].intersection(union_special)
                    result[f"percent_coverage_hc_{category}"] = (intersection_special.area / survey_area) * 100 if survey_area > 0 else 0.0

    return result, report_lines

def merge_and_check(survey_path: str, polygons_path: Optional[str] = None, certainty_cutoff: Optional[int] = 3):
    """
    Loads, merges, and validates survey and polygon data, computes coverage statistics, and enriches polygons with site-level info.

    This function:
      - Loads survey data (CSV) and polygon data (GeoJSON)
      - Calculates polygon area in square meters
      - Updates the 'category' to 'industrial' for polygons marked 'small-scale' with area > 100,000 m^2
      - Adds a 'site_id' to polygons based on survey matches
      - For each survey row, matches polygons by internal_id (or site_id), year, month, and day
      - Computes percent coverage and statistics for all and high-certainty polygons
      - Checks for consistency between survey irrigation status and polygons, reporting any issues
      - Merges survey and polygon data, saving:
          * A merged survey CSV with coverage stats
          * An enriched polygons GeoJSON with area and site-level info
          * A report of any inconsistencies or issues

    Args:
        survey_path (str): Path to the survey CSV file.
        polygons_path (str, optional): Path to the polygons GeoJSON file. If not provided, uses the same base name as the survey CSV.
        certainty_cutoff (int, optional): Certainty threshold for high-certainty polygon stats (default: 3).

    Returns:
        gpd.GeoDataFrame: Survey data with added coverage/statistics columns.
    """

    # Load the survey and polygon data.
    survey = pd.read_csv(survey_path)
    if polygons_path:
        polygons = gpd.read_file(polygons_path)
    else: 
        polygons = gpd.read_file(survey_path.replace(".csv", ".geojson"))

    # Initialize the report as a list of strings.
    report = []

    # For area calculations we need a geometry for each survey row.
    # The survey CSV includes columns: internal_id, year, month, day, irrigation, x (lon), and y (lat).

    survey["geometry"] = survey.apply(survey_polygon, axis=1)
    survey_gdf = gpd.GeoDataFrame(survey, geometry="geometry", crs="EPSG:4326")

    # We will add a column for the overall location id to the geojson too
    # This will help us ensure that all polygons get matched to a location
    polygons["site_id"] = None 

    # Enrich and save polygons GeoJSON with polygon size and site level info ---
    # Calculate polygon area (in m^2)
    polygons['polygon_area_m2'] = polygons.to_crs("EPSG:32735").geometry.area

    # Update category to 'industrial' if small-scale and area > 100,000 m^2
    polygons.loc[(polygons['category'] == 'small-scale') & (polygons['polygon_area_m2'] > 100000), 'category'] = 'industrial'

    # Process each survey row, collect results and reports
    results = []
    for idx, row in survey_gdf.iterrows():
        result, row_report = process_survey_row(row, polygons, certainty_cutoff, idx)
        results.append(result)
        report.extend(row_report)

    # After processing, add the results as new columns
    results_df = pd.DataFrame(results)
    survey_gdf = pd.concat([survey_gdf.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

    # After processing all survey rows, check for any polygons that were not matched.
    unmatched_polys = polygons[~polygons["site_id"].notnull()]
    for poly_idx, poly in unmatched_polys.iterrows():
        report.append(f"Polygon {poly_idx} (internal_id {poly['internal_id']}, {poly['day']}/{poly['month']}/{poly['year']}) has no matching survey row.")

    # Output the report.
    print("----- CHECK REPORT -----")
    if report:
        for line in report:
            print(line)
    else:
        report.append("All checks passed successfully.")
        print(report[0])

    # Save the report
    merged_folder = os.path.join(os.path.dirname(os.path.dirname(survey_path)), "merged")
    os.makedirs(merged_folder, exist_ok=True)
    report_path = os.path.join(merged_folder, os.path.basename(survey_path).replace(".csv", "_report.txt"))
    with open(report_path, "w") as f:
        for line in report:
            f.write(line + "\n")
    print(f"Saved report at {report_path}")

    # Save  the updated survey results to a CSV, dropping geometry
    survey_gdf["source_file"] = os.path.basename(survey_path).replace(".csv", "")
    survey_results = survey_gdf.copy()
    results_path = os.path.join(merged_folder, os.path.basename(survey_path).replace(".csv", "_merged.csv"))
    survey_results.drop(columns="geometry").to_csv(results_path, index=False)
    print(f"Saved merged dataset at {results_path}")

    # Select relevant site-level columns from survey_gdf to add to the polygons
    survey_info_cols = ['site_id', 'internal_id', 'plot_file', 'x', 'y', 'water_source', 'year', 'month', 'day', 'source_file']
    survey_info = survey_gdf[survey_info_cols].drop_duplicates()

    # Merge polygons with survey info on site_id, internal_id, year, month, day
    polygons_merged = polygons.merge(
        survey_info,
        on=['site_id', 'internal_id', 'year', 'month', 'day'],
        how='left'
    )

    # Save enriched polygons GeoJSON
    polygons_path_out = os.path.join(
        merged_folder,
        os.path.basename(survey_path).replace('.csv', '_polygons.geojson')
    )
    polygons_merged.to_file(polygons_path_out, driver='GeoJSON')
    print(f"Saved merged polygons at {polygons_path_out}")

    # Return the survey results GeoDataFrame (with added percent coverage columns)
    return survey_gdf


if __name__ == "__main__":
    
    # Example usage/test code

    # survey = "data/labels/labeled_surveys/random_sample/processed/MV_v2_425-449.csv"
    # survey_results = merge_and_check(survey)
    # print(survey_results.head())

    # CLI argument parsing

    import argparse

    parser = argparse.ArgumentParser(description="Merge survey data with polygon data and perform consistency checks.")
    parser.add_argument("survey_path", type=str, help="Path to the survey CSV file.")
    parser.add_argument("--polygons_path", type=str, help="Path to the polygons GeoJSON file (optional).")
    args = parser.parse_args()
    
    survey_path = args.survey_path
    polygons_path = args.polygons_path if args.polygons_path else None
    
    survey_results = merge_and_check(survey_path, polygons_path)
    
    print(f"Merged results have {len(survey_results)} rows.")
