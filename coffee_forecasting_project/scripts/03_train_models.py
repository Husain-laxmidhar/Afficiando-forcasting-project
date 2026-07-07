"""
03_train_models.py
--------------------
Trains and evaluates 5 forecasting models (Naive, Moving Average, SARIMA,
Prophet, Gradient Boosting) per store on daily revenue, using a strict
time-based train/test split (no shuffling). Produces:

    artifacts/model_forecasts.parquet   -> test-period predictions for every model/store
    artifacts/model_metrics.parquet     -> MAE / RMSE / MAPE / Peak Error Rate per model/store
    artifacts/future_forecasts.parquet  -> 30-day-ahead forecast (with 80% CI) per model/store
    artifacts/best_model_per_store.json -> model chosen by lowest test RMSE, per store

This script is run OFFLINE (not inside the Streamlit app). The app only reads
the parquet artifacts produced here, so it starts instantly and has no heavy
runtime dependency on prophet/statsmodels being importable in the deploy
environment beyond what's already frozen into requirements.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ART = Path("/home/claude/coffee_project/artifacts")
TEST_DAYS = 30          # last 30 days held out for evaluation
FUTURE_DAYS = 30        # forecast horizon into the future
PEAK_QUANTILE = 0.90    # top 10% of days count as "peak demand" days


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100)

    # Peak Error Rate: of the actual "peak" days (top decile of demand),
    # what fraction did the model fail to flag as a peak (top decile of its own predictions)?
    peak_threshold_true = np.quantile(y_true, PEAK_QUANTILE)
    peak_threshold_pred = np.quantile(y_pred, PEAK_QUANTILE)
    actual_peaks = y_true >= peak_threshold_true
    predicted_peaks = y_pred >= peak_threshold_pred
    if actual_peaks.sum() > 0:
        missed = np.sum(actual_peaks & ~predicted_peaks) / actual_peaks.sum()
    else:
        missed = np.nan
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "Peak_Error_Rate": missed * 100}


# --------------------------------------------------------------------------
# Baseline models
# --------------------------------------------------------------------------
def naive_forecast(train: pd.Series, horizon: int) -> np.ndarray:
    """Repeat the last observed value."""
    return np.repeat(train.iloc[-1], horizon)


def moving_average_forecast(train: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    """Repeat the mean of the last `window` observations."""
    return np.repeat(train.iloc[-window:].mean(), horizon)


# --------------------------------------------------------------------------
# SARIMA
# --------------------------------------------------------------------------
def sarima_forecast(train: pd.Series, horizon: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train.values,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),   # weekly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    fc = fit.get_forecast(steps=horizon)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.20)  # 80% CI
    return mean, ci[:, 0], ci[:, 1]


# --------------------------------------------------------------------------
# Prophet
# --------------------------------------------------------------------------
def prophet_forecast(train_df: pd.DataFrame, horizon: int):
    from prophet import Prophet

    m = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
    )
    m.fit(train_df.rename(columns={"date": "ds", "y": "y"}))
    future = m.make_future_dataframe(periods=horizon)
    fc = m.predict(future)
    tail = fc.tail(horizon)
    return tail["yhat"].values, tail["yhat_lower"].values, tail["yhat_upper"].values


# --------------------------------------------------------------------------
# Gradient Boosting (uses engineered lag/rolling/calendar features)
# --------------------------------------------------------------------------
FEATURE_COLS = [
    "revenue_lag1", "revenue_lag7", "revenue_lag14",
    "revenue_roll3", "revenue_roll7", "revenue_roll14",
    "dow", "is_weekend", "day_of_month", "month_num", "day_of_year",
]


def gbr_forecast(feat_train: pd.DataFrame, feat_full: pd.DataFrame, test_dates, horizon_dates):
    """
    Trains on rows with complete lag features, then recursively predicts
    forward one day at a time (each prediction feeds the next day's lag1).
    """
    train_rows = feat_train.dropna(subset=FEATURE_COLS + ["revenue"])
    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )
    model.fit(train_rows[FEATURE_COLS], train_rows["revenue"])

    # Recursive forecasting over test_dates using actual history up to each point,
    # then continue recursively (using its own predictions) for horizon_dates beyond test.
    history = feat_full.set_index("date")["revenue"].copy()
    preds = {}

    all_dates = list(test_dates) + list(horizon_dates)
    for d in all_dates:
        row = {}
        row["revenue_lag1"] = history.get(d - pd.Timedelta(days=1), np.nan)
        row["revenue_lag7"] = history.get(d - pd.Timedelta(days=7), np.nan)
        row["revenue_lag14"] = history.get(d - pd.Timedelta(days=14), np.nan)
        prior = history[history.index < d]
        row["revenue_roll3"] = prior.tail(3).mean()
        row["revenue_roll7"] = prior.tail(7).mean()
        row["revenue_roll14"] = prior.tail(14).mean()
        row["dow"] = d.dayofweek
        row["is_weekend"] = int(d.dayofweek in (5, 6))
        row["day_of_month"] = d.day
        row["month_num"] = d.month
        row["day_of_year"] = d.dayofyear

        X = pd.DataFrame([row])[FEATURE_COLS]
        yhat = model.predict(X)[0]
        preds[d] = yhat
        # feed prediction back into history so later lags can use it (recursive forecasting)
        history.loc[d] = yhat

    test_preds = np.array([preds[d] for d in test_dates])
    horizon_preds = np.array([preds[d] for d in horizon_dates])
    return test_preds, horizon_preds, model


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    daily = pd.read_parquet(ART / "daily_store.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    feat_all = pd.read_parquet(ART / "daily_features.parquet")
    feat_all["date"] = pd.to_datetime(feat_all["date"])

    all_forecasts = []
    all_metrics = []
    all_future = []
    best_model_per_store = {}

    stores = daily[["store_id", "store_location"]].drop_duplicates().to_dict("records")

    for store in stores:
        sid, sloc = store["store_id"], store["store_location"]
        print(f"\n=== Store {sid} ({sloc}) ===")

        s = daily[daily["store_id"] == sid].sort_values("date").reset_index(drop=True)
        s_feat = feat_all[feat_all["store_id"] == sid].sort_values("date").reset_index(drop=True)

        series = s.set_index("date")["revenue"]
        split_point = len(series) - TEST_DAYS
        train_series = series.iloc[:split_point]
        test_series = series.iloc[split_point:]
        test_dates = test_series.index
        last_date = series.index.max()
        horizon_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=FUTURE_DAYS, freq="D")

        store_forecasts = {}
        store_metrics = {}
        store_future = {}

        # ---- Naive ----
        pred = naive_forecast(train_series, TEST_DAYS)
        m = compute_metrics(test_series.values, pred)
        store_metrics["Naive"] = m
        store_forecasts["Naive"] = pred
        fut = naive_forecast(series, FUTURE_DAYS)
        store_future["Naive"] = (fut, fut, fut)
        print(f"  Naive          MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.1f}%")

        # ---- Moving Average ----
        pred = moving_average_forecast(train_series, TEST_DAYS, window=7)
        m = compute_metrics(test_series.values, pred)
        store_metrics["Moving Average"] = m
        store_forecasts["Moving Average"] = pred
        fut = moving_average_forecast(series, FUTURE_DAYS, window=7)
        store_future["Moving Average"] = (fut, fut, fut)
        print(f"  Moving Average MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.1f}%")

        # ---- SARIMA ----
        try:
            mean, lo, hi = sarima_forecast(train_series, TEST_DAYS)
            m = compute_metrics(test_series.values, mean)
            store_metrics["SARIMA"] = m
            store_forecasts["SARIMA"] = mean
            fmean, flo, fhi = sarima_forecast(series, FUTURE_DAYS)
            store_future["SARIMA"] = (fmean, flo, fhi)
            print(f"  SARIMA         MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.1f}%")
        except Exception as e:
            print(f"  SARIMA failed: {e}")

        # ---- Prophet ----
        try:
            train_df = train_series.reset_index()
            train_df.columns = ["ds", "y"]
            mean, lo, hi = prophet_forecast(train_df, TEST_DAYS)
            m = compute_metrics(test_series.values, mean)
            store_metrics["Prophet"] = m
            store_forecasts["Prophet"] = mean
            full_df = series.reset_index()
            full_df.columns = ["ds", "y"]
            fmean, flo, fhi = prophet_forecast(full_df, FUTURE_DAYS)
            store_future["Prophet"] = (fmean, flo, fhi)
            print(f"  Prophet        MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.1f}%")
        except Exception as e:
            print(f"  Prophet failed: {e}")

        # ---- Gradient Boosting ----
        try:
            feat_train = s_feat.iloc[:split_point]
            test_pred, horizon_pred, gbr_model = gbr_forecast(feat_train, s_feat, test_dates, horizon_dates)
            m = compute_metrics(test_series.values, test_pred)
            store_metrics["Gradient Boosting"] = m
            store_forecasts["Gradient Boosting"] = test_pred
            # approximate CI using residual std from training
            resid_std = np.std(test_series.values - test_pred)
            store_future["Gradient Boosting"] = (
                horizon_pred, horizon_pred - 1.28 * resid_std, horizon_pred + 1.28 * resid_std
            )
            print(f"  Gradient Boost MAE={m['MAE']:.1f} RMSE={m['RMSE']:.1f} MAPE={m['MAPE']:.1f}%")
        except Exception as e:
            print(f"  Gradient Boosting failed: {e}")

        # ---- Assemble per-store test-period forecast table ----
        for model_name, pred in store_forecasts.items():
            df = pd.DataFrame({
                "store_id": sid, "store_location": sloc, "model": model_name,
                "date": test_dates, "actual": test_series.values, "predicted": pred,
            })
            all_forecasts.append(df)

        for model_name, m in store_metrics.items():
            all_metrics.append({"store_id": sid, "store_location": sloc, "model": model_name, **m})

        for model_name, (mean, lo, hi) in store_future.items():
            df = pd.DataFrame({
                "store_id": sid, "store_location": sloc, "model": model_name,
                "date": horizon_dates, "forecast": mean, "lower_80": lo, "upper_80": hi,
            })
            all_future.append(df)

        # ---- Best model selection (lowest RMSE on test set) ----
        best = min(store_metrics.items(), key=lambda kv: kv[1]["RMSE"])
        best_model_per_store[str(sid)] = {"store_location": sloc, "best_model": best[0], "rmse": best[1]["RMSE"]}
        print(f"  -> Best model: {best[0]} (RMSE={best[1]['RMSE']:.1f})")

    # ---- Save everything ----
    pd.concat(all_forecasts, ignore_index=True).to_parquet(ART / "model_forecasts.parquet", index=False)
    pd.DataFrame(all_metrics).to_parquet(ART / "model_metrics.parquet", index=False)
    pd.concat(all_future, ignore_index=True).to_parquet(ART / "future_forecasts.parquet", index=False)
    with open(ART / "best_model_per_store.json", "w") as f:
        json.dump(best_model_per_store, f, indent=2)

    print("\n[done] All models trained and artifacts saved.")


if __name__ == "__main__":
    main()
