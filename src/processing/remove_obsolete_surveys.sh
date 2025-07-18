#!/bin/zsh

# This script moves surveys that are not part of the latest ones made to the raw_obsolete folder

# Get the directory of this script (assumes script is in src/processing/)
SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# Paths (relative to project root)
CSV_PATH="${PROJECT_ROOT}/data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv"
RAW_DIR="${PROJECT_ROOT}/data/labels/labeled_surveys/random_sample/raw"
OBSOLETE_DIR="${PROJECT_ROOT}/data/labels/labeled_surveys/random_sample/raw_obsolete"

# Load allowed values from the CSV (skip header)
allowed=("${(@f)$(tail -n +2 "$CSV_PATH" | awk -F',' '{print $(NF-1)}')}")

# If not in the allowed values (latest surveys), move bothe the .zip and .kml files to the raw_obsolete folder
# Make sure the obsolete directory exists
mkdir -p "$OBSOLETE_DIR"

# Loop through all .zip and .kml files in the raw directory
for f in "$RAW_DIR"/*(.kml|.zip); do
  # Get the base name without extension
  base="${f:t:r}"
  # If the base name is not in the allowed array, move both .zip and .kml (if they exist)
  if [[ ${allowed[(Ie)$base]} -eq 0 ]]; then
    for ext in zip kml; do
      file_to_move="${RAW_DIR}/${base}.${ext}"
      if [[ -e "$file_to_move" ]]; then
        echo "Moving $file_to_move to $OBSOLETE_DIR"
        mv "$file_to_move" "$OBSOLETE_DIR/"
      fi
    done
  fi
done
