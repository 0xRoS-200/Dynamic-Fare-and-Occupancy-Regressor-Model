import sqlite3
import pandas as pd
import pytest

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


@pytest.fixture
def tmp_db(tmp_path):
    db_file = tmp_path / "test_predictions.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            city TEXT,
            route TEXT,
            fare_pred REAL,
            occupancy_pred REAL,
            delay_pred REAL,
            feed_type TEXT
        )
    """)
    for i in range(10):
        feed = "real" if i % 2 == 0 else "synthetic"
        cursor.execute(
            """
            INSERT INTO predictions (ts, city, route, fare_pred, occupancy_pred, delay_pred, feed_type)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"2026-10-15T0{i}:00:00Z",
                "London" if i < 5 else "New_York",
                f"route_{i}",
                5.0 + i * 0.5,
                0.2 + (i % 5) * 0.1,
                float(i),
                feed,
            ),
        )
    conn.commit()
    conn.close()
    return db_file


@pytest.fixture
def empty_db(tmp_path):
    db_file = tmp_path / "empty_predictions.db"
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            city TEXT,
            route TEXT,
            fare_pred REAL,
            occupancy_pred REAL,
            delay_pred REAL,
            feed_type TEXT
        )
    """)
    conn.commit()
    conn.close()
    return db_file


@pytest.fixture
def sample_pred_df():
    data = []
    for i in range(20):
        data.append({
            "id": i + 1,
            "ts": f"2026-10-15T{i % 24:02d}:30:00Z",
            "city": "Paris" if i % 2 == 0 else "Tokyo",
            "route": f"RT_{i}",
            "fare_pred": 6.5 + (i % 7) * 0.8,
            "occupancy_rate": 0.3 + (i % 6) * 0.1,
            "occupancy_pred": 0.32 + (i % 6) * 0.1,
            "delay_minutes": float(i % 10),
            "delay_pred": float((i % 10) + 0.5),
            "feed_type": "real" if i % 3 != 0 else "synthetic",
        })
    return pd.DataFrame(data)


def test_load_predictions_returns_dataframe(tmp_db):
    df = load_predictions(db_path=str(tmp_db), n=500)
    assert isinstance(df, pd.DataFrame)
    assert "fare_pred" in df.columns
    assert "feed_type" in df.columns
    assert len(df) == 10


def test_load_predictions_empty_db_returns_empty_df(empty_db):
    df = load_predictions(db_path=str(empty_db), n=500)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "fare_pred" in df.columns
    assert "feed_type" in df.columns


def test_build_real_vs_predicted_chart_returns_figure(sample_pred_df):
    fig = build_real_vs_predicted_chart(sample_pred_df, target="fare_pred")
    assert fig is not None


def test_all_8_charts_return_figures(sample_pred_df):
    f1 = build_real_vs_predicted_chart(sample_pred_df, target="fare_pred")
    f2 = build_real_vs_synthetic_error_chart(sample_pred_df)
    f3 = build_feature_importance_chart({"boarding_count": 0.45, "route_length_km": 0.30, "hour": 0.25})
    f4 = build_city_heatmap(sample_pred_df)
    f5 = build_rolling_rmse_gauge(sample_pred_df, target="fare_pred")
    f6 = build_anomaly_timeline(sample_pred_df)
    f7 = build_delay_violin(sample_pred_df)
    f8 = build_fare_boxplot(sample_pred_df)

    assert f1 is not None
    assert f2 is not None
    assert f3 is not None
    assert f4 is not None
    assert f5 is not None
    assert f6 is not None
    assert f7 is not None
    assert f8 is not None


def test_charts_handle_empty_dataframe():
    empty_df = pd.DataFrame(columns=[
        "id", "ts", "city", "route", "fare_pred", "occupancy_pred", "delay_pred", "feed_type"
    ])
    assert build_real_vs_predicted_chart(empty_df) is not None
    assert build_real_vs_synthetic_error_chart(empty_df) is not None
    assert build_feature_importance_chart(None) is not None
    assert build_city_heatmap(empty_df) is not None
    assert build_rolling_rmse_gauge(empty_df) is not None
    assert build_anomaly_timeline(empty_df) is not None
    assert build_delay_violin(empty_df) is not None
    assert build_fare_boxplot(empty_df) is not None

