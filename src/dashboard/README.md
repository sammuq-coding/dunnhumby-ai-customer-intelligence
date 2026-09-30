# Streamlit dashboard

The app in `app.py` presents the saved Complete Journey feature tables, finance
outputs, and model/calibration metrics in eight sidebar sections: Executive
Overview, Customer Explorer, Campaign Analytics, Customer Value / CLV, Offer
Economics, Model Performance, Methodology & Assumptions, and AI Analytics
Assistant.

## Data sources

`data_loader.py` resolves and caches these project artifacts:

* `data/processed/features/customer_features`
* `data/processed/features/product_features`
* `data/processed/features/campaign_household_features`
* `data/processed/features/campaign_features`
* `data/processed/finance/customer_economics.parquet`
* `data/processed/finance/campaign_offer_economics.parquet`
* the saved finance report, model metrics, calibration metrics, model split
  metadata, verified source inventory, and the small demographic CSV

The processed Spark 4.2 Parquet files are read through a cached local
SparkSession and converted to pandas only for presentation. The installed
PyArrow 19 reader raises `Repetition level histogram size mismatch` on these
files, so Spark handles Parquet reads. The dashboard therefore needs the same
Java 17 + PySpark setup as the feature pipeline. The 36M-row
`causal_data.csv` is never loaded.

`validate_dashboard_data` checks required columns, unique table keys, valid
campaign/household references, bounded probabilities, nonnegative finite CLV,
and saved metrics sections. Campaign descriptor joins use a many-to-one
validation and must preserve the scored row count. Parquet and JSON reads use
Streamlit caching; the shared Spark session is also a cached resource.

## Interpretation

Campaign redemption/spend summaries are descriptive for assigned households.
The response probability is a calibrated prediction of observed redemption in
the held-out test campaign-household population. It is not a treatment effect.
CLV and offer value are modeled estimates under the settings in
`configs/finance.yaml`; they are not observed future value or incremental
campaign profit. DAY values are ordered indices, not calendar dates. The app
shows campaign observation completeness and limits redemption-rate summaries
to the verified coupon-redemption data coverage where applicable.

## Launch

From the repository root, with the project's local Java 17 configured:

```bash
export JAVA_HOME="$(/usr/libexec/java_home -v 17)"
export PATH="$JAVA_HOME/bin:$PATH"
streamlit run src/dashboard/app.py
```

Build the feature, response-model, and finance artifacts first if they are not
present. The app does not retrain or refit any artifact.
