import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dashboard.data_loader import load_predictions
from dashboard.charts import (
    build_real_vs_predicted_chart,
    build_real_vs_synthetic_error_chart,
    build_feature_importance_chart,
    build_city_heatmap,
    build_rolling_rmse_gauge,
    build_anomaly_timeline,
    build_delay_violin,
    build_fare_boxplot,
)

st.set_page_config(layout="wide", page_title="Transit Regressor Analytics", page_icon="🚆")

st.title("🚆 Transit Dynamic Fare & Occupancy Regressor — Analytics Dashboard")
st.markdown("Real-time monitoring and analytics for distributed PySpark MLlib inference.")

# Sidebar Configuration
st.sidebar.header("Configuration & Filters")
db_path = st.sidebar.text_input("Database Path", value="api/predictions.db")
max_rows = st.sidebar.slider("Max Events to Load", min_value=50, max_value=2000, value=500, step=50)
refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", min_value=1, max_value=30, value=5)
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh", value=False)

# Load Data
df = load_predictions(db_path=db_path, n=max_rows)

if not df.empty and "city" in df.columns:
    all_cities = sorted(df["city"].dropna().unique().tolist())
    selected_cities = st.sidebar.multiselect("Filter Cities", options=all_cities, default=all_cities)
    if selected_cities:
        df = df[df["city"].isin(selected_cities)]

if not df.empty and "feed_type" in df.columns:
    feed_options = ["All"] + sorted(df["feed_type"].dropna().unique().tolist())
    selected_feed = st.sidebar.selectbox("Filter Feed Source", options=feed_options, index=0)
    if selected_feed != "All":
        df = df[df["feed_type"] == selected_feed]

# Metric KPI Header
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
total_count = len(df)
real_count = int((df["feed_type"] == "real").sum()) if "feed_type" in df.columns else 0
synth_count = int((df["feed_type"] == "synthetic").sum()) if "feed_type" in df.columns else 0
avg_fare = f"${df['fare_pred'].mean():.2f}" if "fare_pred" in df.columns and not df.empty else "N/A"
avg_delay = f"{df['delay_pred'].mean():.1f}m" if "delay_pred" in df.columns and not df.empty else "N/A"

kpi1.metric("Total Events", total_count)
kpi2.metric("Real Feed", real_count)
kpi3.metric("Synthetic Feed", synth_count)
kpi4.metric("Avg Predicted Fare", avg_fare)
kpi5.metric("Avg Predicted Delay", avg_delay)

st.markdown("---")

# 2-Column Chart Grid (8 Interactive Plotly Visualizations)
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(build_real_vs_predicted_chart(df, target="fare_pred"), use_container_width=True)
    st.plotly_chart(build_city_heatmap(df), use_container_width=True)
    st.plotly_chart(build_rolling_rmse_gauge(df, target="fare_pred"), use_container_width=True)
    st.plotly_chart(build_delay_violin(df), use_container_width=True)

with col2:
    st.plotly_chart(build_real_vs_synthetic_error_chart(df), use_container_width=True)
    st.plotly_chart(build_feature_importance_chart(), use_container_width=True)
    st.plotly_chart(build_anomaly_timeline(df), use_container_width=True)
    st.plotly_chart(build_fare_boxplot(df), use_container_width=True)

# Data Explorer
with st.expander("Explore Raw Prediction Logs"):
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No predictions recorded yet. Run the streaming job or synthetic generator to populate data.")

# Auto-refresh loop
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

