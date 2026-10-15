import json
import os
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from stream.synthetic_generator import SyntheticGenerator, run_generator


@pytest.fixture
def training_stats():
    stats_path = "stream/training_stats.json"
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            return json.load(f)
    return {
        "fare_amount": {"mean": 8.0, "std": 2.5, "min": 2.0, "max": 16.0},
        "occupancy_rate": {"mean": 0.35, "std": 0.20, "min": 0.0, "max": 1.0},
        "delay_minutes": {"mean": 2.5, "std": 3.0, "min": 0.0, "max": 60.0},
    }


def test_synthetic_stats_within_3sigma_of_training(training_stats, n=1000):
    gen = SyntheticGenerator(stats=training_stats, seed=42)
    batch = [gen.generate_event() for _ in range(n)]
    df = pd.DataFrame(batch)
    for col in ["fare_amount", "occupancy_rate", "delay_minutes"]:
        z = abs(df[col].mean() - training_stats[col]["mean"]) / training_stats[col]["std"]
        assert z < 3, f"{col} mean z-score {z:.2f} > 3"


def test_synthetic_event_has_required_keys(training_stats):
    gen = SyntheticGenerator(stats=training_stats)
    event = gen.generate_event()
    for key in ["city", "route_id", "fare_amount", "occupancy_rate", "delay_minutes"]:
        assert key in event
    assert event.get("feed_type") == "synthetic"
    assert "event_ts" in event


def test_synthetic_generator_default_stats_loaded():
    gen = SyntheticGenerator()
    assert gen.stats is not None
    assert "fare_amount" in gen.stats
    event = gen.generate_event()
    assert "fare_amount" in event
    assert "occupancy_rate" in event
    assert "delay_minutes" in event


def test_anomaly_injection_rate(training_stats):
    gen = SyntheticGenerator(stats=training_stats, seed=123)
    n = 2000
    events = [gen.generate_event() for _ in range(n)]
    anomalies = sum(1 for e in events if e.get("anomaly_flag", 0) == 1)
    rate = anomalies / n
    assert 0.02 <= rate <= 0.08, f"Expected anomaly rate ~0.05, got {rate:.4f}"


def test_run_generator_posts_to_api():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"fare": 8.5, "occupancy": 0.4, "delay": 1.2, "timestamp": "2026-10-15T00:00:00Z"}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        total = run_generator(api_url="http://mock-api:8000", limit=5, delay_sec=0.001)
        assert total == 5
        assert mock_post.call_count == 5
        # Verify feed_type was passed as synthetic
        posted_payload = mock_post.call_args[1]["json"]
        assert posted_payload["feed_type"] == "synthetic"

