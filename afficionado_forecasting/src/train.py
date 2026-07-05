"""
End-to-end pipeline entry point.

Run with:
    conda activate acr-forecasting
    python -m src.train

What it does (mirrors the project methodology document):
    1. Loads raw transactions & reconstructs dates      (data_preprocessing.py)
    2. Builds daily-per-store revenue series
    3. Time-based train/test split (last N days held out, no shuffling)
    4. Trains baseline + statistical models per store    (models.py)
    5. Trains one Gradient Boosting model on pooled, feature-engineered data
    6. Evaluates every model with MAE / RMSE / MAPE / Peak Error Rate
    7. Saves a model-comparison CSV + plots to outputs/
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config
from . import data_preprocessing as prep
from . import feature_engineering as feats
from . import models as mdl
from . import evaluate as ev


def time_split(df: pd.DataFrame, date_col: str, test_days: int):
    cutoff = df[date_col].max() - pd.Timedelta(days=test_days - 1)
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    return train, test


def run_series_models_per_store(daily_store: pd.DataFrame, test_days: int) -> pd.DataFrame:
    """Fits every model in models.SERIES_MODELS on each store's revenue series."""
    rows = []
    predictions_for_plot = {}

    for store_id, g in daily_store.groupby("store_id"):
        g = g.sort_values("date")
        train, test = time_split(g, "date", test_days)
        train_series = train.set_index("date")["revenue"]
        y_true = test["revenue"].values
        horizon = len(test)

        for name, fn in mdl.SERIES_MODELS.items():
            try:
                y_pred = fn(train_series, horizon)
            except Exception as exc:  # pragma: no cover - defensive, e.g. SARIMA convergence issues
                print(f"  [warn] {name} failed for store {store_id}: {exc}")
                continue

            row = ev.summarize(y_true, y_pred, label=name)
            row["store_id"] = store_id
            rows.append(row)
            predictions_for_plot[(store_id, name)] = (test["date"].values, y_true, y_pred)

    return pd.DataFrame(rows), predictions_for_plot


def run_gradient_boosting(daily_store: pd.DataFrame, test_days: int):
    """Single pooled model across all stores, using engineered features."""
    feat_df = feats.add_daily_features(daily_store).dropna()
    exclude = {"date", "store_location", "revenue", "transaction_qty", "n_transactions"}
    feature_cols = [
        c for c in feat_df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(feat_df[c])
    ]

    train, test = time_split(feat_df, "date", test_days)
    y_pred = mdl.gradient_boosting_forecast(train[feature_cols], train["revenue"], test[feature_cols])
    y_true = test["revenue"].values

    result = ev.summarize(y_true, y_pred, label="Gradient Boosting")
    result["store_id"] = "ALL"

    per_store_mape = {}
    test = test.assign(_pred=y_pred)
    for store_id, g in test.groupby("store_id"):
        per_store_mape[store_id] = ev.mape(g["revenue"], g["_pred"])

    return pd.DataFrame([result]), per_store_mape, test


def main():
    print("=" * 70)
    print("Afficionado Coffee Roasters -- Forecasting Pipeline")
    print("=" * 70)

    print("\n[1/5] Preprocessing raw transactions ...")
    _, hourly_store, daily_store, daily_category = prep.run_pipeline()
    print(f"  Reconstructed date range: {daily_store['date'].min().date()} -> {daily_store['date'].max().date()}")
    print(f"  Stores: {sorted(daily_store['store_id'].unique().tolist())}")

    test_days = config.TEST_HOLDOUT_DAYS
    print(f"\n[2/5] Time-based split: last {test_days} days held out per store")

    print("\n[3/5] Training baseline + statistical models per store ...")
    series_results, series_preds = run_series_models_per_store(daily_store, test_days)

    print("\n[4/5] Training Gradient Boosting (pooled, feature-engineered) ...")
    gbm_results, gbm_per_store_mape, gbm_test = run_gradient_boosting(daily_store, test_days)

    print("\n[5/5] Building comparison report & plots ...")
    all_results = pd.concat([series_results, gbm_results], ignore_index=True)
    all_results.to_csv(config.OUTPUTS_DIR / "model_comparison.csv", index=False)

    # KPI: Store Forecast Stability for the GBM model
    stability = ev.store_forecast_stability(gbm_per_store_mape)
    with open(config.OUTPUTS_DIR / "kpi_summary.txt", "w") as f:
        f.write("KPI Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Store Forecast Stability (GBM, std of per-store MAPE): {stability:.2f}\n")
        f.write(f"Per-store MAPE (GBM): {gbm_per_store_mape}\n")
        f.write("\nFull model comparison saved to model_comparison.csv\n")

    # Best model per store by RMSE (series models only, for a quick chart)
    best_per_store = (
        series_results.sort_values("RMSE")
        .groupby("store_id")
        .first()
        .reset_index()
    )
    print("\nBest series model per store (by RMSE):")
    print(best_per_store[["store_id", "model", "MAE", "RMSE", "MAPE (%)"]].to_string(index=False))

    # Plot: actual vs. best-model forecast, per store
    fig, axes = plt.subplots(len(best_per_store), 1, figsize=(10, 4 * len(best_per_store)), squeeze=False)
    for i, row in best_per_store.iterrows():
        store_id, best_model = row["store_id"], row["model"]
        dates, y_true, y_pred = series_preds[(store_id, best_model)]
        ax = axes[i, 0]
        ax.plot(dates, y_true, label="Actual", marker="o")
        ax.plot(dates, y_pred, label=f"Forecast ({best_model})", marker="x")
        ax.set_title(f"Store {store_id} -- Daily Revenue: Actual vs Forecast")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(config.OUTPUTS_DIR / "forecast_vs_actual.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved: {config.OUTPUTS_DIR / 'model_comparison.csv'}")
    print(f"Saved: {config.OUTPUTS_DIR / 'kpi_summary.txt'}")
    print(f"Saved: {config.OUTPUTS_DIR / 'forecast_vs_actual.png'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
