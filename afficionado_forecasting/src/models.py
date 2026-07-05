"""
Step 4 of the methodology: Forecasting Models.

Baseline   : Naive, Moving Average
Statistical: SARIMA, Exponential Smoothing
Advanced   : Prophet (seasonality-aware), Gradient Boosting Regression

Every `fit_*` function takes a training series/frame and a forecast horizon
(number of steps) and returns a numpy array of predictions of that length,
so they can be swapped into train.py interchangeably.
"""
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------
def naive_forecast(train_series: pd.Series, horizon: int) -> np.ndarray:
    """Repeats the last observed value for the whole horizon."""
    last_value = train_series.iloc[-1]
    return np.full(horizon, last_value, dtype=float)


def seasonal_naive_forecast(train_series: pd.Series, horizon: int, season_length: int = 7) -> np.ndarray:
    """Repeats the value from the same point in the previous season (e.g. same weekday)."""
    tail = train_series.iloc[-season_length:].values
    reps = int(np.ceil(horizon / season_length))
    return np.tile(tail, reps)[:horizon]


def moving_average_forecast(train_series: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    avg = train_series.iloc[-window:].mean()
    return np.full(horizon, avg, dtype=float)


# ---------------------------------------------------------------------------
# Statistical models
# ---------------------------------------------------------------------------
def sarima_forecast(train_series: pd.Series, horizon: int,
                     order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=horizon)
    return np.asarray(forecast)


def exponential_smoothing_forecast(train_series: pd.Series, horizon: int,
                                    seasonal_periods: int = 7) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = ExponentialSmoothing(
        train_series, trend="add", seasonal="add",
        seasonal_periods=seasonal_periods, initialization_method="estimated",
    )
    fitted = model.fit()
    forecast = fitted.forecast(horizon)
    return np.asarray(forecast)


# ---------------------------------------------------------------------------
# Advanced models
# ---------------------------------------------------------------------------
def prophet_forecast(train_df: pd.DataFrame, horizon: int) -> np.ndarray:
    """train_df must have columns ['ds', 'y']."""
    from prophet import Prophet

    model = Prophet(weekly_seasonality=True, yearly_seasonality=False, daily_seasonality=False)
    model.fit(train_df[["ds", "y"]])
    future = model.make_future_dataframe(periods=horizon, freq="D", include_history=False)
    forecast = model.predict(future)
    return forecast["yhat"].values


def gradient_boosting_forecast(train_X: pd.DataFrame, train_y: pd.Series,
                                test_X: pd.DataFrame, random_state: int = 42) -> np.ndarray:
    """Supervised regression on engineered lag/rolling/calendar features.
    Unlike the other models here, this needs the test-period features
    up front (produced by feature_engineering.py) rather than just a horizon."""
    from sklearn.ensemble import GradientBoostingRegressor

    model = GradientBoostingRegressor(random_state=random_state, n_estimators=300,
                                       max_depth=3, learning_rate=0.05)
    model.fit(train_X, train_y)
    return model.predict(test_X)


# ---------------------------------------------------------------------------
# Registry so train.py can loop over "all baseline+statistical models" easily
# ---------------------------------------------------------------------------
SERIES_MODELS = {
    "Naive": naive_forecast,
    "Seasonal Naive": seasonal_naive_forecast,
    "Moving Average (7d)": moving_average_forecast,
    "SARIMA": sarima_forecast,
    "Exponential Smoothing": exponential_smoothing_forecast,
}
