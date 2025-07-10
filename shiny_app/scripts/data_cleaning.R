# Prepare Merged Data
# This script cleans and prepares the data for use in the Shiny App
# The scripts write cleaned_shiny_irrigation_data.csv to properly run the Shiny App when hosted online
# Run this script manually when labeling data is updated

library(dplyr)
library(tidyverse)

### --- Map Data --- ###

##### RAW DATA #####
# Load the map data
raw <- read_csv("data/labels/labeled_surveys/random_sample/latest_irrigation_table.csv")

# Check to see if site_id corresponds to unique lat / lon locations
raw |>
  group_by(site_id) |>
  summarise(n_unique_coords = n_distinct(paste(x, y))) |>
  filter(n_unique_coords > 1)
    # The tibble returned no rows, so we can confirm that site_id corresponds to unique lat/lon locations

# Removing unwanted columns
raw_clean <- raw |>
  select(-c(internal_id, plot_file, operator, source_file)) |>
  # Extract the numeric ID for each location
  mutate(location_num = as.integer(gsub("id_", "", site_id))) |>
  # Drop ID name
  select(-site_id)

#### DISTRICTS ####
# Add in district boundaries 
library(sf)

# Read district boundaries shapefile
districts <- st_read("data/zambia_districts/Zambia_-_Administrative_District_Boundaries_2022.shp")

# Match the CRS to the lat / lon
districts <- st_transform(districts, crs = 4326)

# Convert the raw data to an sf object
raw_sf <- st_as_sf(raw_clean, coords = c("x", "y"), crs = 4326)

# Perform a spatial join to add district information
joined <- st_join(raw_sf, districts)

# Convert geometry back to x, y
join_clean <- joined |>  # your sf object after spatial join
  mutate(
    x = st_coordinates(geometry)[, 1],
    y = st_coordinates(geometry)[, 2]
  ) |>
  st_drop_geometry() |>
  rename(
    district = DISTRICT,
    province = PROVINCE
  )

# Write to shiny_data folder for use in Time Series 
write_csv(join_clean, "shiny_app/shiny_data/cleaned_shiny_timeseries_data.csv")


#### GROUP BY LOCATION ####
# Group by location and average the percent cover values
summary_data <- join_clean |>
  group_by(location_num, x, y, district, province) |>
  summarise(
    images = max(image_number, na.rm = TRUE),
    year = first(year),
    month = first(month),
    day = first(day),
    avg_certainty = mean(irrigation, na.rm = TRUE),
    avg_percent_coverage = mean(percent_coverage, na.rm = TRUE),
    avg_percent_coverage_high = mean(percent_coverage_high_certainty, na.rm = TRUE),
    n_labelers = n_distinct(operator_initials),
    water_source_mode = names(sort(table(water_source), decreasing = TRUE))[1],
    .groups = "drop"
  )

# Add a new column for log transform to visualize coverage easier
summary_data <- summary_data |>
  mutate(log_coverage = if_else(
    avg_percent_coverage == 0,
    0,
    log1p(avg_percent_coverage)
  ))

# Write the cleaned data to a CSV file for local use in shiny for map
write_csv(summary_data, "shiny_app/shiny_data/cleaned_shiny_map_data.csv")



