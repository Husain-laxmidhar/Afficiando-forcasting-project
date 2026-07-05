"""
Model Evaluation & KPIs, as specified in the project methodology.

Metrics:
    MAE, RMSE, MAPE, Peak Error Rate (missed rush-hour events)

KPIs:
    Forecast Accuracy (%), Peak Demand Capture Rate, Revenue Forecast Error,
    Store Forecast Stability
"""
import numpy as np
import pandas as pd

from . import config


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred, epsilon: float = 1e-6) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < epsilon, epsilon, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def peak_error_rate(df: pd.DataFrame, actual_col: str, pred_col: str,
                     hour_col: str = "hour_of_day", tolerance: float = 0.2) -> float:
    """Share of rush-hour rows (config.PEAK_HOURS) where forecast error exceeds
    `tolerance` (default 20%) of the actual value -- i.e. "missed" peaks."""
    peaks = df[df[hour_col].isin(config.PEAK_HOURS)]
    if peaks.empty:
        return float("nan")
    rel_err = np.abs(peaks[actual_col] - peaks[pred_col]) / peaks[actual_col].clip(lower=1e-6)
    return float((rel_err > tolerance).mean() * 100)


def forecast_accuracy_pct(y_true, y_pred) -> float:
    """100 - MAPE, floored at 0. A simple, interpretable 'accuracy' KPI."""
    return float(max(0.0, 100 - mape(y_true, y_pred)))


def revenue_forecast_error(y_true_revenue, y_pred_revenue) -> float:
    """Signed % error of total forecast revenue vs actual total revenue."""
    total_true = np.sum(y_true_revenue)
    total_pred = np.sum(y_pred_revenue)
    if total_true == 0:
        return float("nan")
    return float((total_pred - total_true) / total_true * 100)


def store_forecast_stability(per_store_mape: dict) -> float:
    """Lower = more consistent forecast quality across stores.
    Defined as the standard deviation of each store's MAPE."""
    values = list(per_store_mape.values())
    return float(np.std(values)) if values else float("nan")


def summarize(y_true, y_pred, label: str = "") -> dict:
    result = {
        "model": label,
        "MAE": round(mae(y_true, y_pred), 3),
        "RMSE": round(rmse(y_true, y_pred), 3),
        "MAPE (%)": round(mape(y_true, y_pred), 2),
        "Forecast Accuracy (%)": round(forecast_accuracy_pct(y_true, y_pred), 2),
    }
    return result


def summarize_multiple(results: dict) -> pd.DataFrame:
    """results: {model_name: (y_true, y_pred)} -> tidy comparison table."""
    rows = [summarize(y_true, y_pred, label=name) for name, (y_true, y_pred) in results.items()]
    return pd.DataFrame(rows).sort_values("RMSE")
