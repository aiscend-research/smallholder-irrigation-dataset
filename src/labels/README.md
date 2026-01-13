# Earth Collect Labeling Guide

## Overview

This folder contains tools and instructions for labeling smallholder irrigation using **Earth Collect** and **Google Earth Pro**, as part of a broader effort to build a machine learning benchmark dataset for dry season irrigation in Zambia. 

📃 **Refer to the full labeling protocol here**:  
[📝 Labeling Guide (Google drive link)](https://docs.google.com/document/d/1F-5uTBTCsP3ZU5hwj1NE4RYbmBofbXzeytcCKIT6iz8/edit?usp=sharing)

This README complements the guide with technical instructions and script documentation.

---

## Software prerequisites

- **Google Earth Pro** provides access to historical high-resolution imagery. Download and install Google Earth Pro from the [official website](https://www.google.com/earth/about/versions/)
- **Open Foris Collect + Earth Collect** allow structured data collection across 
  - **Open Foris Collect**: This software is used to create and manage the labeling design. The survey used in this project was created using this tool and can be found in the folder `data/labels/survey_template/irrigation_survey_3_6_published_20250411T143124.zip`. If you are just using this already created survey, you do not need to download Open Foris Collect. To create your own survey, download and install Open Foris Collect from the [official website](https://openforis.org/solutions/collect/).
  - **Earth Collect**: Earth Collect takes in a survey created using Collect and administers it through Google Earth Pro. Download and install Earth Collect from the [official website](https://openforis.org/solutions/collect-earth/).


---

## Generating surveys

To create a survey to label, you will need: 
1. A survey template, saved in `data/labels/survey_template/` (e.g. `irrigation_survey_3_6_published_20250411T143124.zip`)
2. Sample location file(s) created using `src/sampling/`, stored in `data/sampling/samples/<SAMPLE-GROUP-NAME>`

To generate surveys that can be read into Earth Collect/Google Earth Pro, you can use `surveys_with_locations.py`, which will generate surveys for each location file in your sample group folder and save them in the `data/labels/unlabeled_surveys/<SAMPLE-GROUP-NAME>` folder. Note this script will also change the bounding box size to 1km in all surveys. 

Example usage for the `random_sample` sample group:

```bash
python surveys_with_locations.py --survey_name irrigation_survey_3_6_published_20250411T143124.zip --sample_group random_sample
```

More information on what exactly is being changed in the survey template when this command is run: 

<details>

<summary>Modifying the bounding box sizes and sampling locations</summary>
<br>

The survey is exported as a `.cep` file. This file's extenstion can be changed to `.zip` and can then be unzipped and modified to follow the specifications you would like, e.g. how big you want the bounding box to be and the list of locations you would like to sample. 

Specifically, to change the bounding boxes to be 1km across, modify the `distance_to_plot_boundaries` variable in the `project_definition.properties` file to to 500, since this indicates that the center point will be 500 meters to the boundary. 

Additionally, the survey will include some test locations, which are example locations that the survey can be tested on in Google Earth Pro (see `test_samples.ced`). You can provide your own locations, for example a `.csv` generated using the files in the `sampling` section in this repository). To do so, add the `.csv` file to the folder and modify the `csv` variable in the `project_definition.properties` to by replacing "test_samples.ced" with the name of the new `.csv` file you added. 

Once the survey is modified, it can be zipped back up and imported into Collect Earth (`Files > Import CEP` file and then choose `Files of Type: All Types` so it finds your `.zip` file). Make sure that when zipping you zip the *files* together, not the folder containing all the files, otherwise Collect Earth will not be able to open it properly. 
</details>

---

# Generating and exporting labels

Load in and fill out the survey in Earth Collect/Google Earth Pro. The survey responses can be exported as a zip file in Earth Collect, and polygons can be placed in a folder and exported as a `.kml` file in Google Earth Pro. 

For more information on generating and exporting labels, please refer to the [📝 Labeling Guide](https://docs.google.com/document/d/1F-5uTBTCsP3ZU5hwj1NE4RYbmBofbXzeytcCKIT6iz8/edit?usp=sharing)

---

# Quality Control: Inter-Rater Comparison

After labels are collected from multiple labelers, use the `LabelComparison` class to assess labeling consistency by comparing a ground truth (GT) labeler against other labelers.

## Quick Start

```python
from src.labels.label_comparison import LabelComparison

comparison = LabelComparison(
    irrigation_table_path='data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv',
    polygons_path='data/labels/labeled_surveys/random_sample/latest_polygons.geojson',
    image_boundaries_path='data/labels/labeled_surveys/random_sample/latest_irrigation_data.geojson',
    gt_operator='AB',                              # Ground truth labeler initials
    comparison_operators=['DSB', 'JL', 'KL', 'MV', 'PS'],
    min_certainty=4,                               # Filter polygons by certainty
    date_tolerance_days=1,                         # Match images ±1 day
    output_dir='outputs/labeler_comparison'
)

# Generate all plots and summary
for op in comparison.comparison_operators:
    comparison.plot_confusion_matrix(op)
    comparison.plot_detection_metrics_bar(op)
    comparison.plot_area_metrics_bar(op)
    comparison.plot_area_histograms(op)
    comparison.print_summary(op)

# Generate summary tables with weighted averages
detection_table, area_table = comparison.generate_summary_tables()
```

See `notebooks/labeler_comparison.ipynb` for a complete interactive example.

## Metrics

Two levels of metrics are computed:

### 1. Image-Level Detection
Binary classification: Did the labeler detect ANY irrigation in the image?
- **Precision** = TP / (TP + FP) — Of images where comparison labeled irrigation, how many did GT agree?
- **Recall** = TP / (TP + FN) — Of images where GT saw irrigation, how many did comparison detect?

### 2. Area Overlap
How much do the labeled polygon areas agree?
- **Precision** = intersection_area / comp_area — What % of area marked by comparison was correct?
- **Recall** = intersection_area / gt_area — What % of GT area was found by comparison?
- **IoU** = intersection_area / union_area

Overall area metrics sum areas across all matched images before computing ratios.

## Output Files

When `output_dir` is specified, the following files are saved:
- `{op}_confusion_matrix.png` — Image detection confusion matrix
- `{op}_detection_metrics.png` — Detection metrics bar chart
- `{op}_area_metrics.png` — Area overlap bar chart
- `{op}_area_histograms.png` — Per-image metric distributions
- `{site_id}_{date}.png` — Side-by-side polygon comparison plots
- `image_detection_metrics.csv` — Summary table with weighted averages
- `area_overlap_metrics.csv` — Summary table with weighted averages

## Module Structure

- `label_comparison.py` — Main `LabelComparison` class with all plotting and metric methods
- `inter_rater_comparison.py` — Helper functions for loading/filtering data
