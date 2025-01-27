
# Vancouver Air Quality Predictor
### _Based on surrounding wildfires and wind velocity forecasts_

## Why? ##
Forecasting of the Air Quality Health Index (AQHI) is a service already provided by Environment Canada;
as such, I neither expect nor hope for more accurate predictions than what they already provide.
At the moment, this is a project for me to showcase my skills designing and building a climate service from scratch,
working with climate/meteorological data, and experimenting with some ML techniques... and maybe even have some fun!

## Environment Setup ##
Below are rough steps for setting up on Windows using python 3.12. Other OS / python versions haven't been tested.
- git clone to desired location (creates the aqhi-predictor folder)
- Install python 3.12 to default location without adding to PATH variables (see dependencies folder)
- create python venv: <path_to_python_312>\python.exe -m venv <path_to_ahqi-predictor>
- Install python libraries: <path_to_ahqi-predictor>\venv\scripts\python.exe -m pip install -r <path_to_aqhi-predictor>\src\requirements.txt
- Add a config.ini file to <path_to_ahqi-predictor>\src\app (see config_SAMPLE.ini)
- Obtain a FIRMS API key (free), then add it to your config.ini: https://firms.modaps.eosdis.nasa.gov/api/map_key/
- Create DB; create database, schema, and tables; update url in config.ini (TODO: provide more setup instructions for postgres)

## Running some commands ## 
### Gathering source data and save to DB ###
#### Note the following are for one month only; more months should be used to build a more accurate model. ####
- Wildfire observations (from FIRMS): <path_to_aqhi-predictor>\venv\scripts\python.exe <path_to_aqhi-predictor>\src\app\main.py --data-type wildfire --start-date 2024-07-01 --end-date 2024-07-31
- Wind velocity observations (from ECCC): <path_to_aqhi-predictor>\venv\scripts\python.exe <path_to_aqhi-predictor>\src\app\main.py --data-type wind --start-date 2024-07-01 --end-date 2024-07-31
- AQHI observations (from ECCC): <path_to_aqhi-predictor>\venv\scripts\python.exe <path_to_aqhi-predictor>\src\app\main.py --data-type aqhi --start-date 2024-07-01 --end-date 2024-07-31
- Wind velocity forecasts (from ECCC): <path_to_aqhi-predictor>\venv\scripts\python.exe <path_to_aqhi-predictor>\src\app\main.py --data-type wind-forecast --start-date 2024-07-01 --end-date 2024-07-31
- AQHI forecasts (from ECCC): <path_to_aqhi-predictor>\venv\scripts\python.exe <path_to_aqhi-predictor>\src\app\main.py --data-type aqhi-forecast --start-date 2024-07-01 --end-date 2024-07-31

### _Following is the project plan (phased approach). Further commentary is under doc/diary.txt._


## PHASE 0: Feasibility study
### _Before anyone gets too excited... can we even do this?_

### PHASE 0.1: Data accessibility
#### _What are the required datasets? Are they accessible and programmatically downloadable/scrapeable without violating license agreements?_

**COMPLETED:**
- Write & test Python scripts to download and store locally in a PostgreSQL DB:
  - **Wildfires (historical)**: FIRMS (chosen for its free, well-documented API and extensive historical data).
  - **Wildfires (near-real-time)**: FIRMS.
  - **Wind velocity (historical)**: ECCC.
  - **Wind velocity (forecasts)**: ECCC ("UMOS statistically post-processed Forecast of the Regional Deterministic Prediction System (RDPS-UMOS-MLR)" collection).
  - **Air Quality Health Index (AQHI, historical)**: ECCC ([MSC datamart link](https://eccc-msc.github.io/open-data/msc-data/aqhi/readme_aqhi_en/)).
  - **AQHI (forecasts)**: ECCC ([MSC Geomet collection link](https://api.weather.gc.ca/openapi?f=html#/aqhi-forecasts-realtime/items)).

---

### PHASE 0.2: Modeling capabilities
#### _Can the data be combined and used to model Vancouver's AQHI?_

**TODO:**
- Define a **baseline scope** for Phase 0:
  - Use one month of daily data for Vancouver, considering BC wildfires and wind within 1000 km.
- Train a simple baseline model (e.g., linear regression or decision tree) for predicting AQHI.
- Compare predicted AQHI with observed values using metrics like:
  - Mean Absolute Error (MAE).
  - Percentage of predictions within ±1 AQHI category.
- Document modeling results and challenges.

---

### PHASE 0.3: Reporting and visualization
#### _Convey findings effectively._

**TODO:**
- Create basic visualizations to explain the results, such as:
  - Scatterplots of predicted vs. actual AQHI.
  - Map visualizations of wildfire locations and their relative impact on AQHI.
- Highlight how the data engineering pipeline supports reproducibility and scalability.

---

## PHASE 1: BC-only data
#### _How does the model perform on an ongoing basis, using only BC data?_

**TODO:**
- Automate the pipeline for daily data ingestion, model training, and prediction.
- Implement basic alerting for unusual AQHI predictions.
- Evaluate ongoing performance and fine-tune the model.

---

## PHASE 2: Add data from surrounding states & provinces
#### _How does additional data from more geographic areas impact the model?_

**TODO:**
- Incorporate data from Alberta, Washington, Montana, Idaho, and Oregon for both wildfires and wind velocity.
- Evaluate improvements in prediction accuracy and any associated complexities (e.g., increased data volume, model runtime).
- Document results and rationale for inclusion or exclusion of additional data.

---

