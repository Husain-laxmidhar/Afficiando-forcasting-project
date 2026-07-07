import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.models import best_forecast

st.set_page_config(page_title="Coffee Forecast Dashboard", layout="wide")

st.title("☕ Demand Forecast Dashboard")

# Load data
df = pd.read_parquet("data/processed/daily_store.parquet")

store_ids = df["store_id"].unique()
store_id = st.selectbox("Select Store", store_ids)

horizon = st.slider("Forecast Days", 7, 30, 14)

store_df = df[df["store_id"] == store_id].copy()

preds, ci, model_name = best_forecast(store_df, store_id, horizon)

st.subheader(f"📊 Model Used: {model_name}")

# Plot
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=store_df["date"],
    y=store_df["revenue"],
    name="Actual"
))

future_dates = pd.date_range(
    start=store_df["date"].iloc[-1],
    periods=horizon+1
)[1:]

fig.add_trace(go.Scatter(
    x=future_dates,
    y=preds,
    name="Forecast"
))

# Confidence band
fig.add_trace(go.Scatter(
    x=future_dates,
    y=ci["upper"],
    line=dict(width=0),
    showlegend=False
))

fig.add_trace(go.Scatter(
    x=future_dates,
    y=ci["lower"],
    fill='tonexty',
    name="Confidence Interval",
    line=dict(width=0)
))

st.plotly_chart(fig, use_container_width=True)
