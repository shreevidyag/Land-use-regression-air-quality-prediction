# Land Use Regression Air Quality Prediction

This project applies Land Use Regression modelling to predict annual mean NO2 and PM2.5 concentrations at monitoring stations across Finland. It uses real measurement data from the European Environment Agency Air Quality e-Reporting database and compares three modelling approaches: Ridge regression as the classical LUR baseline, Random Forest, and Gradient Boosting. Model evaluation follows the conventions of Morley and Gulliver (2018), using both five-fold cross-validation and Leave-One-Out Cross-Validation.

The project connects my earlier research in LSTM-based spatial-temporal pollution modelling at RIT Bangalore to a European open data context. It forms part of a spatial data portfolio alongside the spatio-temporal-maps and flood-finland repositories.

---
## Live Demo
https://land-use-regression-air-quality-prediction-hpcre6dvkxac8qkjxpm.streamlit.app/


## Setup and running

Install the dependencies first.

```
pip install -r requirements.txt
```

Then run the data download script. This contacts the EEA Air Quality e-Reporting API, downloads verified hourly concentration data for Finland from 2019 to 2022, aggregates to annual means, and fetches road proximity counts from OpenStreetMap. It needs an internet connection and takes around five to ten minutes.

```
python scripts/download_eea_data.py
```

Once the download completes, start the dashboard.

```
streamlit run app.py
```

---

## Data sources

All data is publicly available from official European sources.

Air quality measurements come from the EEA Air Quality e-Reporting Verified dataset, also known as E1a, which contains validated annual mean concentrations from Finnish monitoring stations. The data is downloaded programmatically using the airbase Python library, which wraps the EEA Parquet download API. The original service is at https://eeadmz1-downloads-webapp.azurewebsites.net/

The road_count_500m predictor is fetched via the OpenStreetMap Overpass API. It counts the number of major road segments within 500 metres of each station, covering motorway, trunk, primary, secondary, and tertiary roads. This approximates the ROADLENGTH_500 variable in Morley and Gulliver (2018), though it uses a count of segments rather than total road length weighted by traffic flow.

Latitude and longitude are included as spatial covariates. In Finland, north-south gradients are meaningful because population density, traffic intensity, and heating-related emissions all decline sharply from south to north.

---

## How the predictor set relates to Morley and Gulliver (2018)

The variable construction approach follows the ESCAPE-derived conventions documented in Morley and Gulliver (2018). Their RLUR tool generates LUR variables at monitoring station locations from CORINE land cover polygons, OSM road networks, and Eurostat population centroids, using circular buffers at radii from 25 metres to 1000 metres. Table S1 in their supplementary material lists 28 candidate predictors.

This project implements a subset of that variable set using only sources that can be accessed programmatically without GIS file downloads, so the pipeline runs end-to-end from a single script. To build a richer predictor set matching the full RLUR specification, you would add CORINE land cover fractions and Eurostat population density to the processed CSV and then include them in the BASE_FEATURES list in models/lur_model.py.

CORINE land cover fractions within 300, 500, and 1000 metre buffers can be extracted using rasterio or QGIS from the Copernicus CORINE Land Cover 2018 dataset at https://land.copernicus.eu/pan-european/corine-land-cover

Population density within 500 and 1000 metre buffers can be extracted from the Eurostat GEOSTAT 2018 grid at https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/population-distribution-demography/geostat

---

## Modelling approach

All three models go through a consistent pipeline. Features are standardised using scikit-learn StandardScaler before fitting, which is necessary for Ridge regression and improves the interpretability of coefficients.

Ridge LUR is the classical baseline. In the LUR literature, linear regression with careful variable selection is the standard approach because coefficient signs must make physical sense: positive for pollution sources such as roads and industrial land, negative for sinks such as parks and forests. The model development guidelines in Morley and Gulliver (2018) Table 1 specify that variables with unexpected sign should be removed even if statistically significant, because they likely reflect spatial confounding rather than a true causal relationship. This project does not implement automated sign-checking, so the Ridge model here is closer to a regularised regression than a fully specified LUR in the epidemiological sense.

Random Forest and Gradient Boosting are included as machine learning comparisons. These methods do not enforce sign constraints and will fit any correlation in the training data. They can capture nonlinear relationships that Ridge cannot, but their predictions are not directly interpretable in the same way, and with a small dataset they can be less stable than Ridge.

---

## Model performance

Performance figures are generated when you run the dashboard. Realistic expectations for this predictor set on Finnish national monitoring data are LOOCV R2 between 0.40 and 0.65 for NO2 and between 0.30 and 0.55 for PM2.5. These are lower than what is reported in studies using the full ESCAPE variable set, which typically achieve LOOCV R2 of 0.55 to 0.80 for NO2, because the current three-variable predictor set cannot capture the local spatial variation that CORINE and population variables provide. Adding those variables will improve performance substantially.

---

## Repository structure

The scripts folder contains download_eea_data.py, which handles all data acquisition and preprocessing. The models folder contains lur_model.py, the modelling pipeline. The app.py file is the five-tab Streamlit dashboard. Raw Parquet files from EEA go into data/raw and the cleaned CSV goes into data/processed.

---

## Reference

Morley, D.W. and Gulliver, J. (2018). A land use regression variable generation, modelling and prediction tool for air pollution exposure assessment. Environmental Modelling and Software, 105, 17 to 23. The RLUR tool is available at https://github.com/dwmorley/RLUR

---

## Author

Shree Vidya Gurudath

Master of Business Informatics candidate, Metropolia University of Applied Sciences, Helsinki

https://linkedin.com/in/shreevidya-gurudath-6437b9200
