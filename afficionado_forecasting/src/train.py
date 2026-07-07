from src.models import best_forecast
import pandas as pd

def run_training(df):
    results = []

    for store_id, group in df.groupby("store_id"):
        preds, _, model_name = best_forecast(group, store_id)

        results.append({
            "store_id": store_id,
            "model": model_name,
            "forecast_mean": preds.mean()
        })

    return pd.DataFrame(results)
