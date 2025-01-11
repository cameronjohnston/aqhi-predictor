
# Vancouver Air Quality Predictor
### _Based on surrounding wildfires_

## PHASE 0: Feasibility study
### _Before anyone gets too excited... can we even do this?_

### PHASE 0.1: Data accessibility
#### _What are the required datasets? Are they accessible and programmatically downloadable/scrapeable without violating license agreements?_

**TODO:**
- Write & test Python scripts to download and store locally in a PostgreSQL DB:
  - **Wildfires (historical)**: FIRMS (chosen for its free, well-documented API and extensive historical data).
  - **Wildfires (near-real-time)**: FIRMS.
  - **Wind velocity (historical)**: ECCC.
  - **Wind velocity (forecasts)**: ECCC ("UMOS statistically post-processed Forecast of the Regional Deterministic Prediction System (RDPS-UMOS-MLR)" collection).
  - **Air Quality Health Index (AQHI, historical)**: ECCC ([collection link](https://api.weather.gc.ca/openapi?f=html#/aqhi-observations-realtime/getAqhi-observations-realtimeFeatures)).
  - **AQHI (forecasts)**: ECCC ([collection link](https://api.weather.gc.ca/openapi?f=html#/aqhi-forecasts-realtime/items)).

**Considerations:**
- Validate data formats and ensure alignment across datasets (e.g., timestamps, geographic resolution).
- Document licensing restrictions and ensure compliance.

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

## Final Notes:
- Focus on showcasing **data engineering skills**:
  - Data ingestion and ETL pipelines.
  - Handling large-scale, real-time datasets.
  - Integrating diverse data sources.
- Visualization should be minimal but impactful, aiding in communicating results rather than demonstrating proficiency in visualization tools.
- Regularly update documentation to reflect progress and key decisions.