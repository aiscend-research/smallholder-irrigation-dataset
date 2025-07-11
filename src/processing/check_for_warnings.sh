#!/bin/zsh

# This script lists out the surveys that are the most recent versions but still have warnings associated with them.

# Get the directory of this script (assumes script is in src/processing/)
SCRIPT_DIR="${0:A:h}"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# Paths (relative to project root)
CSV_PATH="${PROJECT_ROOT}/data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv"
MERGED_DIR="${PROJECT_ROOT}/data/labels/labeled_surveys/random_sample/merged"

# Load allowed values from the CSV (skip header)
allowed=("${(@f)$(tail -n +2 "$CSV_PATH" | awk -F',' '{print $(NF-1)}')}")

# Now filter files
for f in "$MERGED_DIR"/*.txt; do
  name="${${f:t:r}%_report}"
  if [[ "$(cat "$f")" != $'All checks passed successfully.' ]]; then
    # Check if $name is in $allowed array
    if [[ ${allowed[(Ie)$name]} -ne 0 ]]; then
      echo "$name"
    fi
  fi
done