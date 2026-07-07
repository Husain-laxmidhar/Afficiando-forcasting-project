"""
Afficionado Coffee Roasters — Forecasting & Peak Demand Dashboard
===================================================================
A Streamlit app for store-level sales forecasting and peak-demand analysis.

DESIGN NOTE (important for deployment):
This app performs NO model training at runtime. Every forecast, metric, and
aggregate it displays was pre-computed offline (see /scripts/01-04_*.py) and
saved as small parquet files in /artifacts. This keeps the deployed app's
dependency footprint tiny (pandas + plotly + streamlit only) and avoids the
prophet / pmdarima / conda-vs-pip build failures that heavy forecasting
libraries cause on Streamlit Community Cloud.

To regenerate the artifacts with new data, run the scripts in /scripts in
order (01 -> 02 -> 03 -> 04), then redeploy — the app itself never changes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------
# Page config & constants
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Afficionado Coffee Roasters | Demand Forecasting",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

ART = Path(__file__).parent / "artifacts"
MODEL_COLORS = {
    "Naive": "#B0B0B0",
    "Moving Average": "#8C8C8C",
    "SARIMA": "#C97B3D",
    "Prophet": "#6B4226",
    "Gradient Boosting": "#2E7D32",
}
BRAND_BROWN = "#4A2E1F"
BRAND_CREAM = "#F5E9DA"


# --------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------
@st.cache_data
def load_all():
    data = {}
    data["daily_store"] = pd.read_parquet(ART / "daily_store.parquet")
    data["model_forecasts"] = pd.read_parquet(ART / "model_forecasts.parquet")
    data["model_metrics"] = pd.read_parquet(ART / "model_metrics.parquet")
    data["future_forecasts"] = pd.read_parquet(ART / "future_forecasts.parquet")
    data["hourly_heatmap"] = pd.read_parquet(ART / "hourly_heatmap.parquet")
    data["category_summary"] = pd.read_parquet(ART / "category_summary.parquet")
    data["kpi_summary"] = pd.read_parquet(ART / "kpi_summary.parquet")
    data["peak_hours"] = pd.read_parquet(ART / "peak_hours.parquet")
    with open(ART / "best_model_per_store.json") as f:
        data["best_model"] = json.load(f)

    for key in ["daily_store", "model_forecasts", "future_forecasts"]:
        data[key]["date"] = pd.to_datetime(data[key]["date"])
    return data


DATA = load_all()

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MODEL_LIST = ["Naive", "Moving Average", "SARIMA", "Prophet", "Gradient Boosting"]


# --------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.title("☕ Afficionado Coffee Roasters")
st.sidebar.caption("Data-Driven Forecasting & Peak Demand Prediction")

stores = DATA["kpi_summary"][["store_id", "store_location"]].drop_duplicates()
store_options = dict(zip(stores["store_location"], stores["store_id"]))
selected_store_name = st.sidebar.selectbox("Store", list(store_options.keys()))
selected_store_id = store_options[selected_store_name]

metric_toggle = st.sidebar.radio("Metric", ["Revenue ($)", "Transaction Quantity"], index=0)
metric_col = "revenue" if metric_toggle == "Revenue ($)" else "transaction_qty"
forecast_val_col = "forecast"  # future_forecasts always stores revenue-scale forecasts

horizon = st.sidebar.slider("Forecast horizon (days)", min_value=7, max_value=30, value=14, step=1)

st.sidebar.markdown("---")
selected_models = st.sidebar.multiselect(
    "Models to compare", MODEL_LIST, default=["Prophet", "Gradient Boosting", "SARIMA"]
)

st.sidebar.markdown("---")
best = DATA["best_model"][str(selected_store_id)]
st.sidebar.success(f"**Best model for {selected_store_name}:**\n\n{best['best_model']} (RMSE {best['rmse']:.0f})")
st.sidebar.caption(
    "Note: Forecasts are pre-computed offline from historical data (Jan–Jun 2025). "
    "The 'future' period shown extends beyond the last recorded transaction."
)


# --------------------------------------------------------------------------
# Header + KPI row
# --------------------------------------------------------------------------
st.title("Data-Driven Forecasting & Peak Demand Prediction")
st.caption("Afficionado Coffee Roasters — Store-Level Sales Intelligence Dashboard")

kpi = DATA["kpi_summary"][DATA["kpi_summary"]["store_id"] == selected_store_id].iloc[0]
peak = DATA["peak_hours"][DATA["peak_hours"]["store_id"] == selected_store_id].iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Revenue (6 mo.)", f"${kpi['total_revenue']:,.0f}")
c2.metric("Avg Daily Revenue", f"${kpi['avg_daily_revenue']:,.0f}")
c3.metric("Growth (1st vs last mo.)", f"{kpi['growth_pct_first_vs_last_month']:.1f}%")
c4.metric("Busiest Hour", f"{int(peak['hour']):02d}:00")
c5.metric("Best Model (by RMSE)", kpi["best_model"], f"MAPE {kpi['best_model_mape']:.1f}%")

st.markdown("---")

tab_forecast, tab_compare, tab_peak, tab_category, tab_about = st.tabs(
    ["📈 Forecast", "🧪 Model Comparison", "🔥 Peak Demand Heatmap", "🏷️ Category Mix", "ℹ️ About"]
)


# --------------------------------------------------------------------------
# TAB 1 — Forecast (history + future projection with confidence interval)
# --------------------------------------------------------------------------
with tab_forecast:
    st.subheader(f"{selected_store_name} — {metric_toggle} Forecast")

    hist = DATA["daily_store"][DATA["daily_store"]["store_id"] == selected_store_id].sort_values("date")
    fut = DATA["future_forecasts"][DATA["future_forecasts"]["store_id"] == selected_store_id]

    default_model = best["best_model"]
    model_for_chart = st.selectbox(
        "Model shown on chart", MODEL_LIST, index=MODEL_LIST.index(default_model)
    )

    fut_model = fut[fut["model"] == model_for_chart].sort_values("date").head(horizon)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist[metric_col], name="Historical", mode="lines",
        line=dict(color=BRAND_BROWN, width=1.5),
    ))

    if metric_col == "revenue":
        fig.add_trace(go.Scatter(
            x=fut_model["date"], y=fut_model["forecast"], name=f"{model_for_chart} Forecast",
            mode="lines", line=dict(color=MODEL_COLORS.get(model_for_chart, "#2E7D32"), width=2.5, dash="dash"),
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([fut_model["date"], fut_model["date"][::-1]]),
            y=pd.concat([fut_model["upper_80"], fut_model["lower_80"][::-1]]),
            fill="toself", fillcolor="rgba(46,125,50,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="80% Confidence Interval", showlegend=True,
        ))
    else:
        st.info(
            "Confidence-interval forecasts were trained on revenue. "
            "Transaction-quantity history is shown; switch to Revenue to see the forward forecast."
        )

    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title=None, yaxis_title=metric_toggle,
        plot_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View forecast data table"):
        st.dataframe(
            fut_model[["date", "forecast", "lower_80", "upper_80"]].rename(
                columns={"forecast": "Forecast", "lower_80": "Lower 80% CI", "upper_80": "Upper 80% CI"}
            ).set_index("date"),
            use_container_width=True,
        )


# --------------------------------------------------------------------------
# TAB 2 — Model comparison (test-period accuracy)
# --------------------------------------------------------------------------
with tab_compare:
    st.subheader(f"{selected_store_name} — Model Accuracy on Held-Out Test Period (last 30 days)")

    metrics_store = DATA["model_metrics"][DATA["model_metrics"]["store_id"] == selected_store_id]
    metrics_store = metrics_store.set_index("model").loc[
        [m for m in MODEL_LIST if m in metrics_store["model"].values]
    ].reset_index()

    mcol1, mcol2 = st.columns([1, 1])
    with mcol1:
        fig_bar = px.bar(
            metrics_store, x="model", y="RMSE", color="model",
            color_discrete_map=MODEL_COLORS, title="RMSE by Model (lower is better)",
        )
        fig_bar.update_layout(showlegend=False, height=380, xaxis_title=None)
        st.plotly_chart(fig_bar, use_container_width=True)
    with mcol2:
        fig_bar2 = px.bar(
            metrics_store, x="model", y="MAPE", color="model",
            color_discrete_map=MODEL_COLORS, title="MAPE % by Model (lower is better)",
        )
        fig_bar2.update_layout(showlegend=False, height=380, xaxis_title=None)
        st.plotly_chart(fig_bar2, use_container_width=True)

    st.dataframe(
        metrics_store.set_index("model")[["MAE", "RMSE", "MAPE", "Peak_Error_Rate"]].style.format(
            {"MAE": "{:.1f}", "RMSE": "{:.1f}", "MAPE": "{:.1f}%", "Peak_Error_Rate": "{:.1f}%"}
        ),
        use_container_width=True,
    )

    st.markdown("##### Test-Period Predictions vs Actuals")
    fc = DATA["model_forecasts"][
        (DATA["model_forecasts"]["store_id"] == selected_store_id)
        & (DATA["model_forecasts"]["model"].isin(selected_models))
    ]
    fig2 = go.Figure()
    actual = fc[fc["model"] == fc["model"].iloc[0]][["date", "actual"]].drop_duplicates()
    fig2.add_trace(go.Scatter(x=actual["date"], y=actual["actual"], name="Actual", mode="lines+markers",
                               line=dict(color="black", width=2)))
    for m in selected_models:
        sub = fc[fc["model"] == m]
        fig2.add_trace(go.Scatter(x=sub["date"], y=sub["predicted"], name=m, mode="lines",
                                   line=dict(color=MODEL_COLORS.get(m), dash="dot")))
    fig2.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Peak Error Rate = share of actual top-decile demand days the model failed to also "
        "rank in its own top-decile of predictions. It measures how well a model flags rush periods, "
        "not just average accuracy."
    )


# --------------------------------------------------------------------------
# TAB 3 — Peak demand heatmap
# --------------------------------------------------------------------------
with tab_peak:
    st.subheader(f"{selected_store_name} — Hourly Demand Heatmap")
    hm = DATA["hourly_heatmap"][DATA["hourly_heatmap"]["store_id"] == selected_store_id]
    pivot = hm.pivot(index="dow_name", columns="hour", values="avg_transactions").reindex(DOW_ORDER)

    fig3 = px.imshow(
        pivot, color_continuous_scale="YlOrBr", aspect="auto",
        labels=dict(x="Hour of Day", y="Day of Week", color="Avg Transactions"),
    )
    fig3.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        f"Busiest hour overall: **{int(peak['hour']):02d}:00** "
        f"(~{peak['avg_transactions']:.0f} avg. transactions). "
        "Use this view for staff scheduling and inventory prep timing."
    )


# --------------------------------------------------------------------------
# TAB 4 — Category mix
# --------------------------------------------------------------------------
with tab_category:
    st.subheader(f"{selected_store_name} — Revenue Mix by Product Category")
    cat = DATA["category_summary"][DATA["category_summary"]["store_id"] == selected_store_id].sort_values(
        "total_revenue", ascending=False
    )
    fig4 = px.bar(
        cat, x="total_revenue", y="product_category", orientation="h",
        color="revenue_share_pct", color_continuous_scale="YlOrBr",
        labels={"total_revenue": "Total Revenue ($)", "product_category": ""},
    )
    fig4.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig4, use_container_width=True)


# --------------------------------------------------------------------------
# TAB 5 — About / methodology
# --------------------------------------------------------------------------
with tab_about:
    st.markdown("""
### About this dashboard

This dashboard supports **short-term (1–7 day)** and **medium-term (14–30 day)**
demand forecasting across Afficionado Coffee Roasters' three stores: Astoria,
Hell's Kitchen, and Lower Manhattan.

**Data note:** the source POS export contains a time-of-day per transaction but
no explicit date column. A continuous 181-day calendar index (Jan 1 – Jun 30,
2025) was reconstructed per store from the sequential transaction order and
daily time resets, then used as the forecasting timeline throughout this app.

**Models evaluated:** Naive (last-value), 7-day Moving Average, SARIMA(1,1,1)(1,1,1)_7,
Facebook Prophet (weekly seasonality), and Gradient Boosting Regression on
lag/rolling/calendar features. The best model per store is selected by lowest
RMSE on a held-out final 30-day test window (time-based split, no shuffling).

**Metrics:** MAE, RMSE, MAPE, and Peak Error Rate (missed top-decile demand
days) — see the Model Comparison tab.

All forecasts shown are pre-computed offline; this app performs no live
model training.
""")
