# Irrigation Explorer – Shiny App

An interactive R Shiny dashboard for exploring manually labeled smallholder irrigation data across Zambia. This app is part of the [`smallholder-irrigation-dataset`](https://github.com/anna-boser/smallholder-irrigation-dataset) repository.

---

## 📋 Table of Contents

* [Project Overview](#project-overview)
* [Features](#features)
* [Folder Structure](#folder-structure)
* [Data Access](#data-access)
* [Running the App](#running-the-app)
* [Dependencies](#dependencies)
* [Acknowledgments](#acknowledgments)
* [Authors](#authors)

---

## Project Overview

This repository contains a dashboard to:

* Visualize the spatial distribution and intensity of smallholder irrigation.
* Filter by time, certainty score, and water source.
* Explore temporal trends in high-certainty irrigation presence.

Labeled data was generated via Google Earth Pro and EarthCollect, representing 1,000 sites across Zambia. The app was developed to support research and policy targeting groundwater and surface water resilience.

---

## Features

* **Map Viewer**: Interactive map with leaflet-based filtering and site-specific metadata
* **Coverage Time Series**: Average irrigation coverage with confidence intervals by year
* **About the Data**: A static tab explaining labeling methodology, visuals, and irrigation basics

---

## Folder Structure

```
shiny_app/
├── app.R                          # Launch script
├── ui.R                           # UI layout (dashboardPage)
├── server.R                       # Server logic for all tabs
├── scripts/
│   └── data_cleaning.R            # Processes raw data to generate cleaned inputs
├── shiny_data/
│   ├── cleaned_shiny_map_data.csv         # Location-level summaries
│   └── cleaned_shiny_timeseries_data.csv  # Full image-level data for time series
├── www/
│   ├── logo.png
│   ├── irrigationdiagram.png
│   ├── labelingex1.png, labelingex2.png, labelingex3.png
│   └── context.html               # Rendered Quarto HTML page
```

---

## Data Access

* **Raw image labeling data** is stored in: `data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv`
* **District shapefile**: `data/zambia_districts/Zambia_-_Administrative_District_Boundaries_2022.shp`
* **Processed datasets** for Shiny app use:

  * `shiny_data/cleaned_shiny_map_data.csv`
  * `shiny_data/cleaned_shiny_timeseries_data.csv`

To update with new labelings, re-run `scripts/data_cleaning.R`.

---

## Running the App

From the project root:

```r
shiny::runApp("shiny_app")
```

This will launch a dashboard with three tabs:

1. **Map Viewer**: Leaflet map with filters (year, certainty, water source, 0% toggle)
2. **Coverage Time Series**: Annual trends in high-certainty percent coverage by province
3. **About the Data**: Static tab with visual explanations and labeling protocol

---

## Dependencies

Make sure the following packages are installed:

```r
install.packages(c(
  "shiny", "shinydashboard", "shinyjs", "leaflet", "dplyr",
  "sf", "readr", "ggplot2", "lubridate", "viridisLite"
))
```

---

## Acknowledgments

This app was developed by researchers affiliated with the Bren School of Environmental Science & Management. The labeling process was conducted in EarthCollect using high-resolution satellite imagery from Google Earth Pro.

Special thanks to contributors involved in data labeling, district shapefile preparation, and time series validation.

---

## Authors

* [Anna Boser](https://github.com/anna-boser)
* [Jackson Coldiron](https://github.com/cycoldiron)
* Additional contributors: imagery labelers, EarthCollect reviewers, project advisors

---

## License

This repository currently does **not** include an explicit license. By default, all rights are reserved. For public reuse or redistribution, please contact the authors or open an issue.

---

## Reporting Issues / Feedback

Please use the [Issues](https://github.com/anna-boser/smallholder-irrigation-dataset/issues) tab to report bugs, suggest features, or request improvements. Include clear reproduction steps and context if possible.

---

## Contributing

We welcome feedback and contributions! To propose changes:

1. Fork the repo
2. Create a new branch
3. Submit a pull request with a clear summary

Please ensure all new code is reproducible and well-commented. If adding data or imagery, note the source and license.

---
