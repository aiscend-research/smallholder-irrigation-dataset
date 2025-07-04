## Processing Scripts README

This folder contains scripts that help you convert your survey and labeling files into usable formats and calculate useful statistics. You can use these scripts to:

* Convert Earth Collect `.zip` survey files into CSVs
* Convert labeled polygons from Google Earth Pro `.kml` files into `.geojson`
* Merge survey responses with labeled polygons
* Generate summary statistics and identify potential labeling errors

Each script expects files to follow a standard directory layout:

* Raw input files (e.g., `.zip` and `.kml`) should be placed in a `raw/` subfolder
    * Generally, these should be in `data/labels/labeled_surveys/<SAMPLE-GROUP-NAME>/raw/`
    * Our surveys have thus far been drawn from the sample group `random_sample` 
* Processed files (e.g., `.csv` and `.geojson`) are saved to a `processed/` subfolder
* Merged outputs from survey + polygon data are saved to a `merged/` subfolder

---

### 🔧 Setup Instructions

Before using any scripts, follow these steps:

1. **Set up your Python environment**

   See the [setup instructions](../../README.md#setup) for details on how to set up your Python environment. Make sure you have all the required packages installed.

2. **Make sure your input files are organized**
   Each `.zip` survey file should have a matching `.kml` polygon file, and both should follow the expected naming format. 

---

### 🚀 Script Overview

| Step | Script                         | Purpose                                                                           |
| ---- | ------------------------------ | --------------------------------------------------------------------------------- |
| 1    | `survey_to_csv.py`             | Converts Earth Collect `.zip` survey files into usable `.csv` files               |
| 2    | `polygons_to_geojson.py`       | Converts Google Earth Pro `.kml` files into `.geojson` format                     |
| 3    | `merge_survey_and_polygons.py` | Merges processed survey data with labeled polygons and computes coverage stats    |
| 4    | `batch_process.py`            | Batch processes a folder of `.zip` and `.kml` files and merges them automatically |
| 5    | `pool_latest_labels.py`   | Pools the latest labeled irrigation data and outputs both a CSV and a GeoJSON file      |

---

### 📘 Example Commands

Each script creates or saves new files as it runs:

* `survey_to_csv.py` creates a `.csv` file in the `processed/` folder with survey results
* `polygons_to_geojson.py` creates a `.geojson` file in the `processed/` folder with labeled polygons
* `merge_survey_and_polygons.py` creates a merged CSV with survey and polygon data in the `merged/` folder, **and also saves a log file** summarizing issues (e.g., missing polygons, duplicate IDs, or outliers)
* `process_folder.py` runs all three steps in sequence and saves outputs to `processed/` and `merged/`
* `pool_latest_labels.py` pools the latest labeled irrigation data for the `random_sample` group and outputs both a CSV and a GeoJSON file with bounding box geometries for each label.

You can either fully process a single pair of survey and polygon files, or batch process an entire folder.

#### ✅ Option 1: Fully process a single pair of files

Run each of the following commands in sequence for your `.zip` and `.kml` file:

```bash
# Convert survey ZIP to CSV
python src/processing/survey_to_csv.py data/labels/labeled_surveys/random_sample/raw/JL_26-50.zip

# Convert KML polygons to GeoJSON
python src/processing/polygons_to_geojson.py data/labels/labeled_surveys/random_sample/raw/JL_26-50.kml

# Merge survey and polygon data. 
# This assumes the previous two steps were successful and there is a geojson with a matching name in the same folder as the csv to match it with. 
python src/processing/merge_survey_and_polygons.py data/labels/labeled_surveys/random_sample/processed/JL_26-50.csv
```

Alternatively, to run all of this in one command, you can instead run `process_file_pair.sh`. To set this up on a Mac, run the following to make it an executable file
```bash
chmod +x src/processing/process_file_pair.sh
```

Then, run the following script, replacing "JL_26-50" with whichever set you want.
```bash
./src/processing/process_file_pair.sh JL_26-50
```

On Windows, open Git Bash and run 
```bash
./src/processing/process_file_pair.sh JL_26-50
```

#### ✅ Option 2: Batch process an entire folder

This command runs all three steps on everything inside the `raw/` folder:

```bash
python src/processing/batch_process.py data/labels/labeled_surveys/random_sample/raw/
```

---

### 📁 File Naming Guidelines

To ensure the scripts work correctly and your files can be matched and processed automatically:

* Filenames must follow the pattern: `<INITIALS>_<ID-RANGE>.<ext>`

  * Examples:

    * `AB_1-25.zip`
    * `AB_1-25.kml`
* `<INITIALS>` should be your operator initials (e.g., `AB`, `JL`, `DSB`)
* `<ID-RANGE>` should match the sample number range (e.g., `1-25`, `26-50`)
* The `.zip` and `.kml` filenames for each sample range must match exactly (aside from file extension)
* Do not include extra characters or spaces in filenames

These naming conventions ensure that:

* Surveys and polygons can be matched automatically
* Output files will be named correctly and saved in the appropriate folder

#### 🔄 If you edit someone else's survey responses:

* Save your updated version using the format: `<YOUR_INITIALS>_<OriginalFileName>`

  * Example: `JL_AB_1-25.zip` if JL edited AB's reponses

#### 🔁 If you revise your own survey responses:

* Add a version number to the filename: `<INITIALS>_v2_<ID-RANGE>.<ext>`, `_v3`, etc.

  * Example: `AB_v2_1-25.kml`, `AB_v3_1-2.zip`

---

### 🤝 Collaboration Guidelines

To help everyone work smoothly together on this project:

* **Use Git branches**: Create and work from your own branch named with your initials (e.g., `jl-working`, `ab-dev`)
* **Commit frequently**: Commit your code and data changes often with clear messages
* **Include data**: If you add or change labeling data, commit those files too -- these are excluded from the `.gitignore`.
* **Pull requests**: When you're ready to merge your work, open a pull request describing what you changed. Aim to do this weekly. 
* **Communicate**: Leave comments or notes in your pull request if you hit issues, made a decision, or want feedback

These habits will help keep the project organized, make collaboration easier, and ensure that we don't lose or overwrite each other's work.

---
