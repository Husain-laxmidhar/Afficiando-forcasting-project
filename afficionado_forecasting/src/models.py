import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


# -------------------------------
# SARIMA
# -------------------------------
def sarima_forecast(series, steps=7):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7))
    res = model.fit(disp=False)
    forecast = res.get_forecast(steps=steps)
    return forecast.predicted_mean, forecast.conf_int()


# -------------------------------
# Exponential Smoothing
# -------------------------------
def ets_forecast(series, steps=7):
    model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=7)
    fit = model.fit()
    forecast = fit.forecast(steps)
    
    # simple interval
    std = np.std(series)
    lower = forecast - 1.96 * std
    upper = forecast + 1.96 * std
    
    return forecast, pd.DataFrame({"lower": lower, "upper": upper})


# -------------------------------
# GBM (BEST MODEL)
# -------------------------------
def gbm_forecast(df, target_col="revenue", steps=7):
    df = df.copy()

    # features
    df["lag1"] = df[target_col].shift(1)
    df["lag7"] = df[target_col].shift(7)
    df["rolling_mean"] = df[target_col].rolling(7).mean()

    df = df.dropna()

    X = df[["lag1", "lag7", "rolling_mean"]]
    y = df[target_col]

    model = GradientBoostingRegressor()
    model.fit(X, y)

    last_row = df.iloc[-1:].copy()
    preds = []

    for _ in range(steps):
        pred = model.predict(last_row[["lag1", "lag7", "rolling_mean"]])[0]
        preds.append(pred)

        # update features
        last_row["lag7"] = last_row["lag1"]
        last_row["lag1"] = pred
        last_row["rolling_mean"] = (last_row["rolling_mean"] * 6 + pred) / 7

    preds = pd.Series(preds)

    std = np.std(y)
    lower = preds - 1.96 * std
    upper = preds + 1.96 * std

    return preds, pd.DataFrame({"lower": lower, "upper": upper})


# -------------------------------
# AUTO MODEL SELECTION
# -------------------------------
def best_forecast(df, store_id, steps=7):
    series = df["revenue"]

    try:
        preds, ci = gbm_forecast(df, steps=steps)
        model_name = "GBM"
    except:
        try:
            preds, ci = sarima_forecast(series, steps)
            model_name = "SARIMA"
        except:
            preds, ci = ets_forecast(series, steps)
            model_name = "ETS"

    return preds, ci, model_name
