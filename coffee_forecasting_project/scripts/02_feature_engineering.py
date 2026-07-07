"""
02_feature_engineering.py
--------------------------
Builds the model-ready feature matrix for daily store-level revenue/quantity
forecasting: lag features, rolling averages, and calendar indicators.

Input:  artifacts/daily_store.parquet
Output: artifacts/daily_features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path

ART = Path("/home/claude/coffee_project/artifacts")


def build_features(daily: pd.DataFrame, target_cols=("revenue", "transaction_qty")) -> pd.DataFrame:
    daily = daily.sort_values(["store_id", "date"]).reset_index(drop=True)
    out_frames = []

    for store_id, g in daily.groupby("store_id"):
        g = g.sort_values("date").reset_index(drop=True)

        for col in target_cols:
            # Lag features (t-1 day, t-7 days = same weekday last week)
            g[f"{col}_lag1"] = g[col].shift(1)
            g[f"{col}_lag7"] = g[col].shift(7)
            g[f"{col}_lag14"] = g[col].shift(14)

            # Rolling averages (using only past data — shift(1) before rolling)
            g[f"{col}_roll3"] = g[col].shift(1).rolling(3).mean()
            g[f"{col}_roll7"] = g[col].shift(1).rolling(7).mean()
            g[f"{col}_roll14"] = g[col].shift(1).rolling(14).mean()

        # Calendar features
        g["dow"] = g["date"].dt.dayofweek           # 0=Mon
        g["is_weekend"] = g["dow"].isin([5, 6]).astype(int)
        g["day_of_month"] = g["date"].dt.day
        g["month_num"] = g["date"].dt.month
        g["day_of_year"] = g["date"].dt.dayofyear

        out_frames.append(g)

    feat = pd.concat(out_frames, ignore_index=True)
    return feat


def main():
    daily = pd.read_parquet(ART / "daily_store.parquet")
    daily["date"] = pd.to_datetime(daily["date"])

    feat = build_features(daily)
    feat.to_parquet(ART / "daily_features.parquet", index=False)
    print(f"[save] daily_features.parquet ({len(feat):,} rows, {feat.shape[1]} cols)")
    print("[cols]", list(feat.columns))


if __name__ == "__main__":
    main()
