from typing import Dict, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_FEATURE_IMPORTANCES = {
    "passenger_count": 0.28,
    "route_length_km": 0.22,
    "occupancy_rate": 0.18,
    "boarding_count": 0.12,
    "hour": 0.08,
    "temperature_c": 0.06,
    "delay_minutes": 0.06,
}


def build_real_vs_predicted_chart(df: pd.DataFrame, target: str = "fare_pred") -> go.Figure:
    """Builds a time-series line chart comparing predictions across feeds."""
    if df.empty or target not in df.columns:
        fig = go.Figure()
        fig.update_layout(title=f"Real vs Predicted: {target} (Awaiting Data)")
        return fig

    sorted_df = df.sort_values(by="id" if "id" in df.columns else "ts")
    x_col = "id" if "id" in sorted_df.columns else "ts"
    color_col = "feed_type" if "feed_type" in sorted_df.columns else None

    fig = px.line(
        sorted_df,
        x=x_col,
        y=target,
        color=color_col,
        title=f"Dynamic Stream Predictions — {target}",
        labels={x_col: "Event Sequence (ID)", target: "Predicted Value"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def build_real_vs_synthetic_error_chart(df: pd.DataFrame) -> go.Figure:
    """Builds a grouped bar chart comparing mean metrics by feed_type."""
    if df.empty or "feed_type" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Feed Comparison: Real vs Synthetic (Awaiting Data)")
        return fig

    metrics = [c for c in ["fare_pred", "occupancy_pred", "delay_pred"] if c in df.columns]
    if not metrics:
        fig = go.Figure()
        fig.update_layout(title="Feed Comparison: Real vs Synthetic (Awaiting Data)")
        return fig

    agg = df.groupby("feed_type")[metrics].mean().reset_index()
    melted = agg.melt(id_vars="feed_type", value_vars=metrics, var_name="Metric", value_name="Average")

    fig = px.bar(
        melted,
        x="feed_type",
        y="Average",
        color="Metric",
        barmode="group",
        title="Average Predictions by Feed Type (Real vs Synthetic)",
        labels={"feed_type": "Feed Source", "Average": "Mean Prediction"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def build_feature_importance_chart(importances: Optional[Dict[str, float]] = None) -> go.Figure:
    """Builds a horizontal bar chart displaying model feature importances."""
    imp_dict = importances or DEFAULT_FEATURE_IMPORTANCES
    df_imp = (
        pd.DataFrame(list(imp_dict.items()), columns=["Feature", "Importance"])
        .sort_values("Importance", ascending=True)
    )

    fig = px.bar(
        df_imp,
        x="Importance",
        y="Feature",
        orientation="h",
        title="MLlib Model Feature Importances",
        labels={"Importance": "Relative Weight", "Feature": "Input Feature"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def build_city_heatmap(df: pd.DataFrame) -> go.Figure:
    """Builds a 2D density heatmap of predicted occupancy by hour and city."""
    if df.empty or "city" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Occupancy Density Heatmap by Hour and City (Awaiting Data)")
        return fig

    plot_df = df.copy()
    if "hour" not in plot_df.columns:
        if "ts" in plot_df.columns:
            plot_df["hour"] = pd.to_datetime(plot_df["ts"], errors="coerce").dt.hour.fillna(12).astype(int)
        else:
            plot_df["hour"] = 12

    z_col = "occupancy_pred" if "occupancy_pred" in plot_df.columns else None

    fig = px.density_heatmap(
        plot_df,
        x="hour",
        y="city",
        z=z_col,
        histfunc="avg",
        nbinsx=24,
        title="Average Predicted Occupancy by City and Hour",
        labels={"hour": "Hour of Day (0-23)", "city": "City", z_col: "Occupancy Rate"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def build_rolling_rmse_gauge(df: pd.DataFrame, target: str = "fare_pred") -> go.Figure:
    """Builds an indicator gauge chart for rolling prediction variance / deviation."""
    if df.empty or target not in df.columns or len(df[target].dropna()) == 0:
        val = 0.0
    else:
        series = df[target].dropna()
        if len(series) > 1:
            diffs = np.diff(series.tail(50))
            val = float(np.sqrt(np.mean(diffs**2)))
        else:
            val = float(series.iloc[0])

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=round(val, 3),
            title={"text": f"Rolling Volatility Index ({target})"},
            gauge={
                "axis": {"range": [0, max(5.0, val * 1.5)]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 1.5], "color": "#e0f3db"},
                    {"range": [1.5, 3.5], "color": "#fee8c8"},
                    {"range": [3.5, max(5.0, val * 1.5)], "color": "#fdbb84"},
                ],
            },
        )
    )
    fig.update_layout(template="plotly_white", margin=dict(t=50, b=20, l=20, r=20))
    return fig


def build_anomaly_timeline(df: pd.DataFrame) -> go.Figure:
    """Builds a scatter plot tracking delays and potential anomalies across time."""
    if df.empty or "delay_pred" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Delay & Anomaly Timeline (Awaiting Data)")
        return fig

    sorted_df = df.sort_values(by="id" if "id" in df.columns else "ts")
    x_col = "ts" if "ts" in sorted_df.columns else "id"
    color_col = "feed_type" if "feed_type" in sorted_df.columns else None
    size_col = "fare_pred" if "fare_pred" in sorted_df.columns and (sorted_df["fare_pred"] > 0).all() else None

    fig = px.scatter(
        sorted_df,
        x=x_col,
        y="delay_pred",
        color=color_col,
        size=size_col,
        hover_data=[c for c in ["city", "route", "occupancy_pred"] if c in sorted_df.columns],
        title="Anomaly Timeline (Predicted Delay & Fare Magnitude)",
        labels={x_col: "Timestamp / Sequence", "delay_pred": "Delay (Minutes)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def build_delay_violin(df: pd.DataFrame) -> go.Figure:
    """Builds a violin plot showing the distribution of delays per city."""
    if df.empty or "delay_pred" not in df.columns or "city" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Delay Distribution by City (Awaiting Data)")
        return fig

    color_col = "feed_type" if "feed_type" in df.columns else None

    fig = px.violin(
        df,
        x="city",
        y="delay_pred",
        color=color_col,
        box=True,
        points="all",
        title="Delay Distributions Across Cities",
        labels={"city": "City", "delay_pred": "Delay (Minutes)"},
    )
    fig.update_layout(template="plotly_white")
    return fig


def build_fare_boxplot(df: pd.DataFrame) -> go.Figure:
    """Builds a box plot showing the spread of predicted fares per city."""
    if df.empty or "fare_pred" not in df.columns or "city" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="Fare Distribution by City (Awaiting Data)")
        return fig

    color_col = "feed_type" if "feed_type" in df.columns else None

    fig = px.box(
        df,
        x="city",
        y="fare_pred",
        color=color_col,
        title="Fare Prediction Spread by City",
        labels={"city": "City", "fare_pred": "Fare ($)"},
    )
    fig.update_layout(template="plotly_white")
    return fig

