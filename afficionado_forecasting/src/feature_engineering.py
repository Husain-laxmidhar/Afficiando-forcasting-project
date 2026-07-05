"""
Step 2 of the methodology: Feature Engineering.

Adds lag features, rolling averages, and calendar dummy variables to the
daily-per-store and hourly-per-store tables produced by data_preprocessing.py.
"""
import pandas as pd

from . import config


def add_daily_features(daily_store: pd.DataFrame) -> pd.DataFrame:
    df = daily_store.sort_values(["store_id", "date"]).copy()

    out = []
    for store_id, g in df.groupby("store_id"):
        g = g.sort_values("date").copy()

        for lag in config.LAG_DAYS:
            g[f"revenue_lag_{lag}"] = g["revenue"].shift(lag)
            g[f"qty_lag_{lag}"] = g["transaction_qty"].shift(lag)

        for w in config.ROLLING_WINDOWS_DAYS:
            g[f"revenue_roll_{w}d"] = g["revenue"].shift(1).rolling(w).mean()
            g[f"qty_roll_{w}d"] = g["transaction_qty"].shift(1).rolling(w).mean()

        out.append(g)

    df = pd.concat(out, ignore_index=True)

    # Calendar dummies
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    dow_dummies = pd.get_dummies(df["day_of_week"], prefix="dow", dtype=int)
    store_dummies = pd.get_dummies(df["store_id"], prefix="store", dtype=int)
    df = pd.concat([df, dow_dummies, store_dummies], axis=1)

    return df


def add_hourly_features(hourly_store: pd.DataFrame) -> pd.DataFrame:
    df = hourly_store.sort_values(["store_id", "hour"]).copy()

    out = []
    for store_id, g in df.groupby("store_id"):
        g = g.sort_values("hour").copy()
        for lag in config.LAG_HOURS:
            g[f"qty_lag_{lag}h"] = g["transaction_qty"].shift(lag)
            g[f"revenue_lag_{lag}h"] = g["revenue"].shift(lag)
        g["qty_roll_24h"] = g["transaction_qty"].shift(1).rolling(24).mean()
        out.append(g)

    df = pd.concat(out, ignore_index=True)

    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    hod_dummies = pd.get_dummies(df["hour_of_day"], prefix="hod", dtype=int)
    store_dummies = pd.get_dummies(df["store_id"], prefix="store", dtype=int)
    df = pd.concat([df, hod_dummies, store_dummies], axis=1)

    return df


if __name__ == "__main__":
    daily = pd.read_parquet(config.DAILY_STORE_PATH)
    hourly = pd.read_parquet(config.HOURLY_STORE_PATH)

    daily_feat = add_daily_features(daily)
    hourly_feat = add_hourly_features(hourly)

    print("Daily features shape:", daily_feat.shape)
    print("Hourly features shape:", hourly_feat.shape)
    print(daily_feat.columns.tolist())
