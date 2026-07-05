"""
Step 1 of the methodology: Time-Series Construction.

- Loads the raw transaction-level export
- Reconstructs a calendar date (see config.py for why this is needed)
- Aggregates to Hour x Store and Day x Store, and Day x Category
- Fills missing intervals with explicit zero-sales rows (no gaps in the index)
"""
import pandas as pd
import numpy as np

from . import config


def load_raw(path=None) -> pd.DataFrame:
    path = path or config.RAW_DATA_PATH
    df = pd.read_csv(path)

    expected_cols = {
        "transaction_id", "year", "transaction_time", "transaction_qty",
        "store_id", "store_location", "product_id", "unit_price",
        "product_category", "product_type", "product_detail",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing expected columns: {missing}")

    return df


def rebuild_dates(df: pd.DataFrame, base_date: str = None) -> pd.DataFrame:
    """Reconstruct a `transaction_date` and full `transaction_datetime`.

    The file has no date column. Row order is chronological across the
    whole export, and `transaction_time` resets to an early value whenever
    the calendar day rolls over. We use that reset as a day boundary.
    """
    base_date = base_date or config.BASE_DATE
    df = df.copy()

    df["transaction_time"] = pd.to_timedelta(df["transaction_time"])

    # A "reset" = this row's time is earlier than the previous row's time.
    time_delta = df["transaction_time"].diff()
    is_reset = time_delta < pd.Timedelta(0)
    is_reset.iloc[0] = False

    day_index = is_reset.cumsum()  # 0, 0, 0, ..., 1, 1, ..., 2, ...
    df["transaction_date"] = pd.to_datetime(base_date) + pd.to_timedelta(day_index, unit="D")
    df["transaction_datetime"] = df["transaction_date"] + df["transaction_time"]
    df["hour"] = (df["transaction_datetime"].dt.floor("h"))
    df["date"] = df["transaction_date"].dt.normalize()

    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    return df


def build_hourly_store(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly transaction volume & revenue, one row per (store, hour)."""
    agg = (
        df.groupby(["store_id", "store_location", "hour"])
        .agg(transaction_qty=("transaction_qty", "sum"),
             revenue=("revenue", "sum"),
             n_transactions=("transaction_id", "count"))
        .reset_index()
    )

    # Fill missing hours per store with explicit zeros so the series has no gaps.
    filled = []
    for (store_id, store_loc), g in agg.groupby(["store_id", "store_location"]):
        full_range = pd.date_range(g["hour"].min(), g["hour"].max(), freq="h")
        g = g.set_index("hour").reindex(full_range)
        g["store_id"] = store_id
        g["store_location"] = store_loc
        g[["transaction_qty", "revenue", "n_transactions"]] = (
            g[["transaction_qty", "revenue", "n_transactions"]].fillna(0)
        )
        g.index.name = "hour"
        filled.append(g.reset_index())

    out = pd.concat(filled, ignore_index=True)
    out["hour_of_day"] = out["hour"].dt.hour
    out["day_of_week"] = out["hour"].dt.dayofweek
    out["date"] = out["hour"].dt.normalize()
    return out


def build_daily_store(df: pd.DataFrame) -> pd.DataFrame:
    """Daily revenue & quantity, one row per (store, date)."""
    agg = (
        df.groupby(["store_id", "store_location", "date"])
        .agg(transaction_qty=("transaction_qty", "sum"),
             revenue=("revenue", "sum"),
             n_transactions=("transaction_id", "count"))
        .reset_index()
    )

    filled = []
    for (store_id, store_loc), g in agg.groupby(["store_id", "store_location"]):
        full_range = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full_range)
        g["store_id"] = store_id
        g["store_location"] = store_loc
        g[["transaction_qty", "revenue", "n_transactions"]] = (
            g[["transaction_qty", "revenue", "n_transactions"]].fillna(0)
        )
        g.index.name = "date"
        filled.append(g.reset_index())

    out = pd.concat(filled, ignore_index=True)
    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def build_daily_category(df: pd.DataFrame) -> pd.DataFrame:
    """Daily demand per product category (all stores combined)."""
    agg = (
        df.groupby(["product_category", "date"])
        .agg(transaction_qty=("transaction_qty", "sum"),
             revenue=("revenue", "sum"))
        .reset_index()
    )

    filled = []
    for cat, g in agg.groupby("product_category"):
        full_range = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full_range)
        g["product_category"] = cat
        g[["transaction_qty", "revenue"]] = g[["transaction_qty", "revenue"]].fillna(0)
        g.index.name = "date"
        filled.append(g.reset_index())

    return pd.concat(filled, ignore_index=True)


def run_pipeline(raw_path=None, save: bool = True):
    """Full step-1 pipeline: load -> rebuild dates -> aggregate -> save."""
    df = load_raw(raw_path)
    df = rebuild_dates(df)

    hourly_store = build_hourly_store(df)
    daily_store = build_daily_store(df)
    daily_category = build_daily_category(df)

    if save:
        hourly_store.to_parquet(config.HOURLY_STORE_PATH, index=False)
        daily_store.to_parquet(config.DAILY_STORE_PATH, index=False)
        daily_category.to_parquet(config.DAILY_CATEGORY_PATH, index=False)

    return df, hourly_store, daily_store, daily_category


if __name__ == "__main__":
    raw, hourly, daily, cat = run_pipeline()
    print(f"Raw transactions: {len(raw):,}")
    print(f"Date range reconstructed: {raw['date'].min().date()} -> {raw['date'].max().date()}")
    print(f"Hourly x Store rows: {len(hourly):,}")
    print(f"Daily x Store rows: {len(daily):,}")
    print(f"Daily x Category rows: {len(cat):,}")
