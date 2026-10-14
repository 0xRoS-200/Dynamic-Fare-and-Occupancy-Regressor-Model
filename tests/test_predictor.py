import os
import sqlite3
import pytest
from fastapi.testclient import TestClient

from api.predictor import Predictor
from api.server import app, get_db_path


@pytest.fixture(scope="module")
def sample_event():
    return {
        "city": "New_York",
        "route_id": "route_1",
        "boarding_count": 15.0,
        "alighting_count": 10.0,
        "passenger_count": 25.0,
        "route_length_km": 8.5,
        "headway_min": 10.0,
        "temperature_c": 22.0,
        "vehicle_capacity": 50.0,
        "hour_sin": 0.5,
        "hour_cos": 0.86,
        "dow_sin": 0.0,
        "month_sin": 0.5,
        "rolling_mean_1h_boarding": 14.0,
        "rolling_std_1h_boarding": 2.0,
        "feed_type": "test",
    }


@pytest.fixture(scope="module")
def loaded_predictor():
    model_dir = os.environ.get("MODEL_DIR", "./hdfs_local/transit/models/")
    return Predictor(model_dir=model_dir)


@pytest.fixture(scope="module")
def test_client():
    return TestClient(app)


def test_predictor_returns_three_floats(loaded_predictor, sample_event):
    result = loaded_predictor.predict(sample_event)
    assert isinstance(result["fare"], float)
    assert isinstance(result["occupancy"], float)
    assert isinstance(result["delay"], float)
    assert "timestamp" in result


def test_post_predict_endpoint(test_client, sample_event):
    r = test_client.post("/predict", json=sample_event)
    assert r.status_code == 200
    data = r.json()
    assert "fare" in data
    assert "occupancy" in data
    assert "delay" in data
    assert "timestamp" in data
    assert isinstance(data["fare"], float)


def test_health_endpoint(test_client):
    r = test_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data.get("models_loaded") is True


def test_predictions_persisted_to_sqlite(test_client, sample_event):
    db_path = get_db_path()
    r = test_client.post("/predict", json=sample_event)
    assert r.status_code == 200

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, ts, city, route, fare_pred, occupancy_pred, delay_pred, feed_type FROM predictions ORDER BY id DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[2] == sample_event["city"]
    assert row[3] == sample_event["route_id"]
    assert isinstance(row[4], float)
    assert row[7] == sample_event["feed_type"]


def test_websocket_stream_endpoint(test_client, sample_event):
    with test_client.websocket_connect("/ws/stream") as websocket:
        websocket.send_json(sample_event)
        data = websocket.receive_json()
        assert "fare" in data
        assert "occupancy" in data
        assert "delay" in data

