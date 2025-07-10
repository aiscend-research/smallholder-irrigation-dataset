#!/bin/bash

# Exit if no argument is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <DATASET_ID> (e.g., JL_26-50)"
  exit 1
fi

# Use the first argument as the dataset ID
ID="$1"

# Paths
RAW_DIR="data/labels/labeled_surveys/random_sample/raw"
PROCESSED_DIR="data/labels/labeled_surveys/random_sample/processed"

# Step 1: Convert survey ZIP to CSV
python src/processing/survey_to_csv.py "$RAW_DIR/${ID}.zip"

# Step 2: Convert KML polygons to GeoJSON
python src/processing/polygons_to_geojson.py "$RAW_DIR/${ID}.kml"

# Step 3: Merge survey and polygon data
python src/processing/merge_survey_and_polygons.py "$PROCESSED_DIR/${ID}.csv"
