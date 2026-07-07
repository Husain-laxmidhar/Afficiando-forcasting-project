# Afficionado Coffee Roasters — Forecasting & Peak Demand Dashboard

## What's in this project

```
coffee_project/
├── scripts/                       # OFFLINE pipeline — run once to (re)generate all forecasts
│   ├── 01_prepare_data.py         # Cleans raw CSV, reconstructs calendar dates, builds aggregates
│   ├── 02_feature_engineering.py  # Lag features, rolling averages, calendar features
│   ├── 03_train_models.py         # Trains Naive/MovingAvg/SARIMA/Prophet/GradientBoosting per store
│   └── 04_build_dashboard_artifacts.py  # Heatmap, category mix, KPI summary
├── artifacts/                     # Full set of generated parquet/json files (offline pipeline output)
├── app/
│   ├── app.py                     # Streamlit dashboard — reads ONLY pre-computed artifacts
│   ├── requirements.txt           # Minimal: streamlit, pandas, plotly, pyarrow
│   └── artifacts/                 # Trimmed copy of only what app.py needs (~84 KB)
└── README.md
```

## Why this two-stage design?

Training SARIMA/Prophet/Gradient Boosting models at every Streamlit page load is
slow and — as encountered previously — heavy libraries like `prophet` and
`pmdarima` are fragile to install on Streamlit Community Cloud (conda/pip
conflicts, build timeouts). So the **training happens once, offline**
(`scripts/01` → `04`), and the **app only reads small parquet files**. The
deployed app's `requirements.txt` never needs `prophet` or `statsmodels` at all.

## Important data caveat

The source CSV (`Afficionado_Coffee_Roasters_xlsx_-_Transactions.csv`) has
**no date column** — only a constant `year` (2025) and a time-of-day field.
`01_prepare_data.py` reconstructs a continuous daily calendar index per store
by detecting the time-of-day "reset" between consecutive transactions
(a large backward jump signals a new day). This produced exactly **181
consecutive days per store**, mapped to **2025-01-01 through 2025-06-30**.
This reconstructed date is an assumption, not a fact recorded in the source
data — it's clearly noted in the app's "About" tab and in the research paper.

## How to re-run the pipeline (e.g. with new/updated data)

```bash
cd scripts
python3 01_prepare_data.py
python3 02_feature_engineering.py
python3 03_train_models.py            # takes ~1-2 min (SARIMA + Prophet fitting)
python3 04_build_dashboard_artifacts.py
```

Then copy the refreshed artifacts the app actually uses into `app/artifacts/`:

```bash
cp artifacts/daily_store.parquet artifacts/model_forecasts.parquet \
   artifacts/model_metrics.parquet artifacts/future_forecasts.parquet \
   artifacts/hourly_heatmap.parquet artifacts/category_summary.parquet \
   artifacts/kpi_summary.parquet artifacts/peak_hours.parquet \
   artifacts/best_model_per_store.json app/artifacts/
```

## How to run the app locally

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

1. Push the `app/` folder (app.py, requirements.txt, artifacts/) to a GitHub repo.
2. Point Streamlit Cloud at `app/app.py` as the entry point.
3. No `environment.yml`, no conda — `requirements.txt` alone is sufficient since
   there are no compiled/heavy dependencies at runtime.
