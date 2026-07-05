"""
Streamlit dashboard for Afficionado Coffee Roasters demand forecasting.

Run with:
    conda activate acr-forecasting
    streamlit run app.py

Core modules implemented (per the project spec):
    - Store-wise sales forecast charts
    - Hourly demand heatmap (average, by hour x day-of-week -- "future" demand pattern)
    - Model selection & comparison
    - Confidence interval visualization
User capabilities:
    - Store selector
    - Forecast horizon slider
    - Revenue vs quantity toggle
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src import config
from src import data_preprocessing as prep
from src import feature_engineering as feats
from src import models as mdl
from src import evaluate as ev

st.set_page_config(page_title="Afficionado Coffee Roasters -- Demand Forecasting",
                    layout="wide", page_icon="☕")


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading & processing transactions ...")
def load_data():
    if config.DAILY_STORE_PATH.exists() and config.HOURLY_STORE_PATH.exists():
        daily_store = pd.read_parquet(config.DAILY_STORE_PATH)
        hourly_store = pd.read_parquet(config.HOURLY_STORE_PATH)
        daily_category = pd.read_parquet(config.DAILY_CATEGORY_PATH)
    else:
        _, hourly_store, daily_store, daily_category = prep.run_pipeline()
    return daily_store, hourly_store, daily_category


@st.cache_data(show_spinner=False)
def get_forecast(daily_store: pd.DataFrame, store_id: int, target: str,
                  model_name: str, horizon: int):
    g = daily_store[daily_store["store_id"] == store_id].sort_values("date")
    train_series = g.set_index("date")[target]

    fn = mdl.SERIES_MODELS[model_name]
    y_pred = fn(train_series, horizon)

    last_date = g["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Simple +/-1.28 std residual-based interval (~80% band) using in-sample residual std
    in_sample_pred = fn(train_series.iloc[:-min(30, len(train_series) - 1)],
                         min(30, len(train_series) - 1))
    resid = train_series.iloc[-len(in_sample_pred):].values - in_sample_pred
    resid_std = np.std(resid) if len(resid) > 1 else train_series.std()

    return future_dates, y_pred, resid_std


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
daily_store, hourly_store, daily_category = load_data()

st.sidebar.header("Controls")
store_options = (
    daily_store[["store_id", "store_location"]].drop_duplicates()
    .sort_values("store_id")
)
store_labels = {row.store_id: f"Store {row.store_id} -- {row.store_location}"
                for row in store_options.itertuples()}
selected_store = st.sidebar.selectbox(
    "Store", options=list(store_labels.keys()), format_func=lambda x: store_labels[x]
)

target_toggle = st.sidebar.radio("Metric", ["Revenue", "Quantity"], horizontal=True)
target_col = "revenue" if target_toggle == "Revenue" else "transaction_qty"

horizon = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=30, value=7, step=1)

model_name = st.sidebar.selectbox("Model", options=list(mdl.SERIES_MODELS.keys()), index=2)

st.sidebar.caption(
    "Note: the raw export has no calendar date column, only a time-of-day stamp. "
    "Dates shown here are reconstructed by treating each time-reset in the file as a "
    "new trading day, starting from an arbitrary base date -- day-of-week patterns are "
    "real, absolute calendar dates are not."
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("☕ Afficionado Coffee Roasters -- Demand Forecasting")
st.caption("Data-driven forecasting & peak demand prediction dashboard")

store_df = daily_store[daily_store["store_id"] == selected_store].sort_values("date")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total historical revenue", f"${store_df['revenue'].sum():,.0f}")
kpi2.metric("Total transactions", f"{int(store_df['n_transactions'].sum()):,}")
kpi3.metric("Avg daily revenue", f"${store_df['revenue'].mean():,.0f}")
kpi4.metric("History span (days)", f"{store_df['date'].nunique()}")

st.divider()

# ---------------------------------------------------------------------------
# Forecast chart with confidence interval
# ---------------------------------------------------------------------------
st.subheader(f"Forecast -- {store_labels[selected_store]}")

future_dates, y_pred, resid_std = get_forecast(daily_store, selected_store, target_col, model_name, horizon)
lower = y_pred - 1.28 * resid_std
upper = y_pred + 1.28 * resid_std

fig = go.Figure()
fig.add_trace(go.Scatter(x=store_df["date"], y=store_df[target_col],
                          mode="lines", name="Historical", line=dict(color="#6f4e37")))
fig.add_trace(go.Scatter(x=future_dates, y=y_pred, mode="lines+markers",
                          name=f"Forecast ({model_name})", line=dict(color="#d97706")))
fig.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(upper) + list(lower[::-1]),
    fill="toself", fillcolor="rgba(217,119,6,0.15)", line=dict(color="rgba(0,0,0,0)"),
    name="~80% confidence interval", showlegend=True,
))
fig.update_layout(height=450, xaxis_title="Date",
                   yaxis_title=target_toggle, legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------
st.subheader("Model comparison (backtest on last 30 days)")

with st.spinner("Backtesting all models ..."):
    test_days = config.TEST_HOLDOUT_DAYS
    g = store_df.copy()
    cutoff = g["date"].max() - pd.Timedelta(days=test_days - 1)
    train = g[g["date"] < cutoff]
    test = g[g["date"] >= cutoff]

    rows = []
    for name, fn in mdl.SERIES_MODELS.items():
        try:
            pred = fn(train.set_index("date")[target_col], len(test))
            rows.append(ev.summarize(test[target_col].values, pred, label=name))
        except Exception as exc:
            st.warning(f"{name} failed: {exc}")
    comparison_df = pd.DataFrame(rows).sort_values("RMSE")

col_a, col_b = st.columns([1.3, 1])
with col_a:
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
with col_b:
    bar_fig = px.bar(comparison_df, x="model", y="RMSE", color="model",
                      title="RMSE by model (lower is better)")
    bar_fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(bar_fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Hourly demand heatmap
# ---------------------------------------------------------------------------
st.subheader("Hourly demand heatmap (avg transaction volume, hour x day-of-week)")

hourly_g = hourly_store[hourly_store["store_id"] == selected_store].copy()
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
hourly_g["dow_name"] = hourly_g["day_of_week"].map(dict(enumerate(dow_names)))

pivot = (
    hourly_g.groupby(["dow_name", "hour_of_day"])["transaction_qty"]
    .mean()
    .reset_index()
    .pivot(index="dow_name", columns="hour_of_day", values="transaction_qty")
    .reindex(dow_names)
)

heat_fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Oranges",
                      labels=dict(x="Hour of day", y="Day of week", color="Avg qty"))
heat_fig.update_layout(height=350)
st.plotly_chart(heat_fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Category demand
# ---------------------------------------------------------------------------
st.subheader("Category-level daily demand (all stores)")
cat_fig = px.line(daily_category, x="date", y="transaction_qty", color="product_category")
cat_fig.update_layout(height=400, legend=dict(orientation="h", y=-0.3))
st.plotly_chart(cat_fig, use_container_width=True)
