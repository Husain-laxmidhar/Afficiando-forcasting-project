# Afficionado Coffee Roasters -- Demand Forecasting & Peak Prediction

End-to-end Python project implementing the methodology in the project brief:
time-series construction, feature engineering, baseline/statistical/advanced
forecasting models, evaluation KPIs, and a Streamlit dashboard.

## 1. Set up the conda environment

```bash
conda env create -f environment.yml
conda activate acr-forecasting
```

If you'd rather use plain pip (e.g. inside an existing conda env):

```bash
pip install -r requirements.txt
```

> Prophet and pmdarima can be slow to build on some platforms. If `conda env
> create` fails on those two, comment them out of `environment.yml`, create
> the env, then install them individually:
> `pip install prophet pmdarima`.

## 2. Add the data

Place the transactions export at:

```
data/raw/transactions.csv
```

(A copy is already there if you built this project from the provided file.)

### Important data note

The raw export only contains `year` (constant, 2025) and `transaction_time`
(`HH:MM:SS`) -- there is **no explicit calendar date column**. The row order
in the file is chronological, and `transaction_time` resets to an early
morning value whenever the day rolls over (verified: 180 resets across the
whole file, i.e. 181 distinct trading days). `src/data_preprocessing.py`
reconstructs a `transaction_date` by treating each reset as a new day,
counting forward from `config.BASE_DATE` (default `2025-01-01`).

**What this means practically:** day-of-week seasonality, lag features, and
relative trends are all valid. The *absolute* calendar dates are a
placeholder, not the true dates of the original transactions. If you obtain
the real date column later, swap `rebuild_dates()` for a direct read of it.

## 3. Run the pipeline

```bash
python -m src.data_preprocessing   # builds data/processed/*.parquet
python -m src.feature_engineering  # sanity-check feature shapes
python -m src.train                # trains & evaluates all models, saves outputs/
```

`train.py` runs the full flow in one go (it calls preprocessing internally),
so in practice you only need:

```bash
python -m src.train
```

Outputs land in `outputs/`:
- `model_comparison.csv` -- MAE / RMSE / MAPE / Forecast Accuracy per model, per store
- `kpi_summary.txt` -- Store Forecast Stability and per-store MAPE
- `forecast_vs_actual.png` -- best model per store, actual vs. forecast plot

## 3b. Deploying to Streamlit Community Cloud

- Push this whole folder to a GitHub repo, keeping `app.py`, `requirements.txt`,
  and `src/` all in the same directory relative to each other (don't split
  them across nested folders).
- In the Streamlit Cloud app settings, set **Main file path** to wherever
  `app.py` ends up in the repo (e.g. `afficionado_forecasting/app.py` if you
  keep this folder as a subdirectory) -- Cloud installs `requirements.txt`
  from that same directory.
- `requirements.txt` deliberately excludes `prophet`, `pmdarima`, and
  `xgboost`. Those need compiled build toolchains (cmdstan / Fortran) that
  routinely fail to build on Cloud's image -- and a failed package build
  aborts the *entire* `pip install -r requirements.txt` step, so you'll see
  `ModuleNotFoundError` even for simple packages like `plotly` that were
  never the real problem. `app.py` doesn't need any of those three anyway.
  If you want them for local experimentation, use `requirements-extra.txt`.
- If you still hit a `ModuleNotFoundError` after this fix, open **Manage
  app -> logs** in Streamlit Cloud and check the `pip install` output near
  the top -- it will name the exact package that failed to build.

## 4. Launch the dashboard

```bash
streamlit run app.py
```

Dashboard features:
- Store selector, forecast horizon slider (7-30 days), revenue/quantity toggle
- Forecast chart with an approximate confidence band
- Live model comparison table + RMSE bar chart for the selected store
- Hourly demand heatmap (hour x day-of-week) to spot rush periods
- Category-level daily demand trends across all stores

## Project layout

```
├── environment.yml           # conda environment
├── requirements.txt          # pip fallback
├── app.py                    # Streamlit dashboard
├── data/
│   ├── raw/transactions.csv  # input (not overwritten by the pipeline)
│   └── processed/            # generated parquet files (hourly/daily aggregates)
├── src/
│   ├── config.py             # paths, constants, modeling assumptions
│   ├── data_preprocessing.py # date reconstruction + hourly/daily aggregation
│   ├── feature_engineering.py# lags, rolling means, calendar dummies
│   ├── models.py             # naive, moving avg, SARIMA, ETS, Prophet, GBM
│   ├── evaluate.py           # MAE, RMSE, MAPE, Peak Error Rate, KPIs
│   └── train.py              # orchestrates the full pipeline
└── outputs/                  # model_comparison.csv, kpi_summary.txt, plots
```

## Models implemented

| Category    | Models |
|-------------|--------|
| Baseline    | Naive, Seasonal Naive, Moving Average |
| Statistical | SARIMA, Exponential Smoothing (Holt-Winters) |
| Advanced    | Prophet (weekly seasonality), Gradient Boosting Regression on lag/rolling/calendar features |

## KPIs / metrics

- **MAE, RMSE, MAPE** -- standard forecast error metrics (`src/evaluate.py`)
- **Peak Error Rate** -- % of rush-hour rows (7-10am by default, see `config.PEAK_HOURS`) where the relative error exceeds a tolerance
- **Forecast Accuracy (%)** -- `100 - MAPE`, floored at 0
- **Revenue Forecast Error** -- signed % difference between total forecast and actual revenue
- **Store Forecast Stability** -- std. deviation of per-store MAPE (lower = more consistent across locations)

## Extending this

- Swap `config.BASE_DATE`/`rebuild_dates()` for a real date column if you get one.
- Add more stores/categories automatically -- everything loops over `groupby("store_id")`, no hardcoded store list.
- Tune SARIMA `order`/`seasonal_order` and Prophet seasonality flags in `src/models.py` per store if residuals show room for improvement.
