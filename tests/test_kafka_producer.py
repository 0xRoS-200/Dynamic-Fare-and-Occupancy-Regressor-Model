import json
import os
import sqlite3
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from stream.kafka_producer import emit_row, run_producer
from stream.spark_streaming_predict import STREAMING_SCHEMA, process_batch
from spark.ingest import get_or_create_spark


@pytest.fixture
def mock_kafka_producer():
    return MagicMock()


@pytest.fixture
def one_row_df():
    csv_path = "archive/data/splits/transit_test.csv/test.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, nrows=1)
    return pd.DataFrame([
        {
            "trip_id": "TRIP_001",
            "timestamp": "2024-01-01 08:00:00",
            "route_id": "route_1",
            "city": "New_York",
            "route_type": "bus",
            "fare_type": "standard",
            "route_length_km": 10.0,
            "occupancy_rate": 0.6,
            "delay_minutes": 5,
        }
    ])


def test_producer_emits_valid_json(mock_kafka_producer, one_row_df):
    emit_row(mock_kafka_producer, one_row_df.iloc[0], topic="transit_live")
    call_args = mock_kafka_producer.send.call_args
    assert call_args is not None
    payload = json.loads(call_args[0][1])
    assert "fare_amount" in payload
    assert "event_ts" in payload
    assert payload["city"] == one_row_df.iloc[0]["city"]


def test_streaming_schema_matches_producer_schema():
    field_names = [f.name for f in STREAMING_SCHEMA.fields]
    assert "fare_amount" in field_names
    assert "event_ts" in field_names
    assert "trip_id" in field_names
    assert "city" in field_names
    assert "route_id" in field_names
    assert "occupancy_rate" in field_names
    assert "delay_minutes" in field_names


def test_process_batch_calls_api_and_inserts_sqlite(tmp_path):
    spark = get_or_create_spark("test_streaming_batch")
    test_db = str(tmp_path / "test_stream_predictions.db")

    sample_data = [
        {
            "trip_id": "TRIP_STREAM_1",
            "city": "London",
            "route_id": "route_5",
            "fare_amount": 3.5,
            "occupancy_rate": 0.75,
            "delay_minutes": 2.0,
            "event_ts": "2026-10-15T00:00:00Z",
        }
    ]
    df = spark.createDataFrame(sample_data)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "fare": 3.75,
        "occupancy": 0.72,
        "delay": 2.1,
        "timestamp": "2026-10-15T00:00:00Z",
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        process_batch(df, batch_id=1, api_url="http://mock-server:8000", db_path=test_db)
        assert mock_post.called

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT city, route, fare_pred, occupancy_pred, delay_pred, feed_type FROM predictions")
    rows = cursor.fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][0] == "London"
    assert rows[0][1] == "route_5"
    assert rows[0][2] == 3.75
    assert rows[0][5] == "real"


def test_run_producer_limit(mock_kafka_producer, tmp_path):
    mini_csv = tmp_path / "mini_test.csv"
    df = pd.DataFrame([
        {"trip_id": f"T_{i}", "city": "Paris", "route_id": f"R_{i}", "fare_type": "standard"}
        for i in range(5)
    ])
    df.to_csv(mini_csv, index=False)

    total_sent = run_producer(
        csv_path=str(mini_csv),
        producer=mock_kafka_producer,
        topic="transit_live",
        delay_sec=0.001,
        limit=3,
    )

    assert total_sent == 3
    assert mock_kafka_producer.send.call_count == 3

