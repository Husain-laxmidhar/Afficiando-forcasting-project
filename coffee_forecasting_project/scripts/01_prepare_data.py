"""
01_prepare_data.py
-------------------
Loads the raw Afficionado Coffee Roasters POS transaction export, reconstructs
a real calendar-date index (the source file only contains a constant `year`
column and a time-of-day column — no date), cleans the data, and writes out
tidy artifacts used by every downstream script:

    artifacts/transactions_clean.parquet   -> row-level cleaned transactions with date/datetime
    artifacts/daily_store.parquet          -> daily revenue & qty per store
    artifacts/daily_category.parquet       -> daily revenue & qty per store x category
    artifacts/hourly_store.parquet         -> hourly transaction volume per store

Date reconstruction logic
--------------------------
Each store's rows are already ordered chronologically by `transaction_id`.
Within a store, the time-of-day resets (a large negative jump) whenever a new
business day begins. We use that to assign a `day_index` (0, 1, 2, ...) per
store, then map day_index -> a real calendar date starting 2025-01-01,
skipping nothing (the roastery operates 7 days/week per the rollover count).
This was verified to produce exactly 181 consecutive days for all 3 stores.
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = "/mnt/user-data/uploads/Afficionado_Coffee_Roasters_xlsx_-_Transactions.csv"
OUT_DIR = Path("/home/claude/coffee_project/artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Timestamp("2025-01-01")


def reconstruct_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a real calendar date to every transaction using per-store time rollovers."""
    df = df.sort_values(["store_id", "transaction_id"]).reset_index(drop=True)

    frames = []
    for store_id, g in df.groupby("store_id", sort=False):
        g = g.copy()
        t = pd.to_datetime(g["transaction_time"], format="%H:%M:%S")
        seconds = t.dt.hour * 3600 + t.dt.minute * 60 + t.dt.second
        rollover = seconds.diff() < -3600  # big backward jump => new day
        day_index = rollover.cumsum()
        g["day_index"] = day_index.values
        g["time_of_day"] = t.dt.time.values
        g["seconds_since_midnight"] = seconds.values
        frames.append(g)

    out = pd.concat(frames, ignore_index=True)
    out["date"] = START_DATE + pd.to_timedelta(out["day_index"], unit="D")
    out["datetime"] = pd.to_datetime(
        out["date"].astype(str) + " " + out["time_of_day"].astype(str)
    )
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="transaction_id")
    df = df[(df["transaction_qty"] > 0) & (df["unit_price"] > 0)]
    df = df.dropna(subset=["store_id", "product_category", "transaction_time"])
    after = len(df)
    print(f"[clean] dropped {before - after} rows ({before} -> {after})")
    return df


def main():
    print("[load] reading raw CSV...")
    df = pd.read_csv(RAW_PATH)
    df = clean(df)

    print("[dates] reconstructing calendar dates from time rollovers...")
    df = reconstruct_dates(df)
    df["revenue"] = df["transaction_qty"] * df["unit_price"]

    # sanity check
    n_days = df.groupby("store_id")["day_index"].nunique()
    print("[check] distinct days per store:\n", n_days)
    print("[check] date range:", df["date"].min().date(), "->", df["date"].max().date())

    df["dow"] = df["date"].dt.day_name()
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["date"].dt.to_period("M").astype(str)

    keep_cols = [
        "transaction_id", "date", "datetime", "hour", "dow", "month",
        "store_id", "store_location", "product_id", "product_category",
        "product_type", "product_detail", "transaction_qty", "unit_price", "revenue",
    ]
    df_clean = df[keep_cols].sort_values(["store_id", "datetime"]).reset_index(drop=True)
    df_clean.to_parquet(OUT_DIR / "transactions_clean.parquet", index=False)
    print(f"[save] transactions_clean.parquet ({len(df_clean):,} rows)")

    # ---- Daily store-level aggregate (main forecasting target) ----
    daily_store = (
        df_clean.groupby(["store_id", "store_location", "date"])
        .agg(revenue=("revenue", "sum"), transaction_qty=("transaction_qty", "sum"),
             n_transactions=("transaction_id", "count"))
        .reset_index()
        .sort_values(["store_id", "date"])
    )
    # fill any missing calendar days with 0 (none expected, but keeps series continuous)
    full = []
    for sid, g in daily_store.groupby("store_id"):
        loc = g["store_location"].iloc[0]
        idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(idx)
        g["store_id"] = sid
        g["store_location"] = loc
        g[["revenue", "transaction_qty", "n_transactions"]] = g[
            ["revenue", "transaction_qty", "n_transactions"]
        ].fillna(0)
        g.index.name = "date"
        full.append(g.reset_index())
    daily_store = pd.concat(full, ignore_index=True)
    daily_store.to_parquet(OUT_DIR / "daily_store.parquet", index=False)
    print(f"[save] daily_store.parquet ({len(daily_store):,} rows)")

    # ---- Daily category-level aggregate ----
    daily_category = (
        df_clean.groupby(["store_id", "store_location", "date", "product_category"])
        .agg(revenue=("revenue", "sum"), transaction_qty=("transaction_qty", "sum"))
        .reset_index()
    )
    daily_category.to_parquet(OUT_DIR / "daily_category.parquet", index=False)
    print(f"[save] daily_category.parquet ({len(daily_category):,} rows)")

    # ---- Hourly store-level aggregate (for heatmap + short-term forecasting) ----
    hourly_store = (
        df_clean.groupby(["store_id", "store_location", "date", "hour"])
        .agg(revenue=("revenue", "sum"), transaction_qty=("transaction_qty", "sum"),
             n_transactions=("transaction_id", "count"))
        .reset_index()
    )
    hourly_store.to_parquet(OUT_DIR / "hourly_store.parquet", index=False)
    print(f"[save] hourly_store.parquet ({len(hourly_store):,} rows)")

    print("\n[done] Data preparation complete.")


if __name__ == "__main__":
    main()
