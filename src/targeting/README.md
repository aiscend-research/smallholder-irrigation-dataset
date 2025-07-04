## Targeting

This module helps determine which areas are more likely to have irrigation in order to 
1. Help target sampling efforts 
1. Understand what factors drive or inhibit irrigation use in Zambia

To do so, we train the following models on verious geospatial data layers:  
1. A logistic regresison model to determine areas with higher likelihood of having irrigation present at all at a given location 
1. A linear regression model to determine areas with more irrigation coverage 

### Contents

#### Features
The regressions are run using geospatial data downloaded and manipulated using code in this module. 

The `features` folder contains: 
1. `data_download` which contains scripts that download raw feature data. These data are saved under `data/targeting/features/raw`
1. `data_manipulation` which contains scripts or functions that turn the raw downloaded data into meaningful features to predict irrigation presence and intensity. Generated data are saved under `data/targeting/features/final`, though intermediate layers may be saved . *All data is saved using the `save_data()` utils function with appropriate descriptions so that metadata is generated.*

To increase organization, all scripts in `features` and data in `data/targeting/features` that belong to the same feature or family of features should contain the same or similar names. For example, climate data may be processed using the scripts `data_download/climate.py` and `data_manipulation/climate.py`, while if there are multiple data files associated with different features they may be saved under `data/targeting/features/final/climate_temperature.py` and `data/targeting/features/final/climate_precipitation.py`

##### Descriptions of features, including raw dataset names and locations: 
1. Rivers...

#### Dataset aggregation
`aggregate_dataset.py` extracts the features prepared in `targeting/features` over the geometries of the irrigation data to generate a DataFrame that includes both the required features and irrigation data. 

This file is saved under `data/targeting/irrigation_with_features.csv`. 

#### Analysis
The `analysis` folder contains scripts and/or notebooks that train models on the generated data. 