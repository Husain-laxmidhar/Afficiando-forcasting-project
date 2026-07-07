"""
04_build_dashboard_artifacts.py
--------------------------------
Builds small, pre-aggregated summary tables the Streamlit app reads directly,
so the app never has to touch the 149K-row raw transaction file or recompute
anything expensive at runtime.

Output:
    artifacts/hourly_heatmap.parquet     -> avg transactions by store x hour x day-of-week
    artifacts/category_summary.parquet   -> revenue share by store x category
    artifacts/kpi_summary.parquet        -> headline KPIs per store (used on the overview page)
    artifacts/peak_hours.parquet         -> top peak hour(s) per store
"""

import pandas as pd
import numpy as np
from pathlib import Path

ART = Path("/home/claude/coffee_project/artifacts")


def main():
    txn = pd.read_parquet(ART / "transactions_clean.parquet")
    hourly = pd.read_parquet(ART / "hourly_store.parquet")
    daily = pd.read_parquet(ART / "daily_store.parquet")
    daily_cat = pd.read_parquet(ART / "daily_category.parquet")
    metrics = pd.read_parquet(ART / "model_metrics.parquet")

    txn["date"] = pd.to_datetime(txn["date"])
    txn["dow_name"] = txn["date"].dt.day_name()

    # ---- Hourly heatmap: avg transaction volume by store x hour x day-of-week ----
    heatmap = (
        txn.groupby(["store_id", "store_location", "dow_name", "hour"])
        .size()
        .reset_index(name="avg_transactions")
    )
    n_weeks = txn["date"].dt.isocalendar().week.nunique()
    heatmap["avg_transactions"] = heatmap["avg_transactions"] / max(n_weeks, 1)
    heatmap.to_parquet(ART / "hourly_heatmap.parquet", index=False)
    print(f"[save] hourly_heatmap.parquet ({len(heatmap)} rows)")

    # ---- Category revenue share per store ----
    cat_summary = (
        daily_cat.groupby(["store_id", "store_location", "product_category"])
        .agg(total_revenue=("revenue", "sum"), total_qty=("transaction_qty", "sum"))
        .reset_index()
    )
    cat_summary["revenue_share_pct"] = (
        cat_summary.groupby("store_id")["total_revenue"].transform(lambda x: 100 * x / x.sum())
    )
    cat_summary.to_parquet(ART / "category_summary.parquet", index=False)
    print(f"[save] category_summary.parquet ({len(cat_summary)} rows)")

    # ---- Peak hour per store (the single busiest hour-of-day, avg across all days) ----
    peak = (
        heatmap.groupby(["store_id", "store_location", "hour"])["avg_transactions"]
        .mean()
        .reset_index()
    )
    peak_hours = peak.loc[peak.groupby("store_id")["avg_transactions"].idxmax()].reset_index(drop=True)
    peak_hours.to_parquet(ART / "peak_hours.parquet", index=False)
    print(f"[save] peak_hours.parquet\n{peak_hours}")

    # ---- KPI summary per store ----
    kpi_rows = []
    for sid, g in daily.groupby("store_id"):
        sloc = g["store_location"].iloc[0]
        total_rev = g["revenue"].sum()
        avg_daily_rev = g["revenue"].mean()
        total_txn = txn[txn["store_id"] == sid].shape[0]
        best_model_row = metrics[metrics["store_id"] == sid].sort_values("RMSE").iloc[0]

        # simple 6-month growth: first month avg vs last month avg
        g2 = g.copy()
        g2["date"] = pd.to_datetime(g2["date"])
        first_month = g2[g2["date"] < g2["date"].min() + pd.Timedelta(days=30)]["revenue"].mean()
        last_month = g2[g2["date"] > g2["date"].max() - pd.Timedelta(days=30)]["revenue"].mean()
        growth_pct = 100 * (last_month - first_month) / first_month if first_month else np.nan

        kpi_rows.append({
            "store_id": sid,
            "store_location": sloc,
            "total_revenue": total_rev,
            "avg_daily_revenue": avg_daily_rev,
            "total_transactions": total_txn,
            "growth_pct_first_vs_last_month": growth_pct,
            "best_model": best_model_row["model"],
            "best_model_rmse": best_model_row["RMSE"],
            "best_model_mape": best_model_row["MAPE"],
        })

    kpi_df = pd.DataFrame(kpi_rows)
    kpi_df.to_parquet(ART / "kpi_summary.parquet", index=False)
    print(f"[save] kpi_summary.parquet\n{kpi_df}")

    print("\n[done] Dashboard artifacts built.")


if __name__ == "__main__":
    main()
