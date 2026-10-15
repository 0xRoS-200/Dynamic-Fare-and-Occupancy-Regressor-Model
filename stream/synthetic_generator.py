import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from faker import Faker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("synthetic_generator")

DEFAULT_CITIES = ["Berlin", "London", "New_York", "Paris", "Singapore", "Tokyo"]

# Commute pattern weights across hours 0-23
HOUR_WEIGHTS = np.array([
    0.01, 0.01, 0.01, 0.01, 0.02, 0.04,
    0.07, 0.10, 0.11, 0.08, 0.04, 0.04,
    0.05, 0.05, 0.05, 0.06, 0.08, 0.10,
    0.08, 0.05, 0.03, 0.02, 0.01, 0.01
])
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()


class SyntheticGenerator:
    """Probabilistic synthetic transit event generator matching training distributions."""

    def __init__(
        self,
        stats: Optional[Dict[str, Dict[str, float]]] = None,
        stats_path: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            random.seed(seed)
            self.fake = Faker()
            Faker.seed(seed)
        else:
            self.rng = np.random.default_rng()
            self.fake = Faker()

        self.cities = list(DEFAULT_CITIES)

        if stats is not None:
            self.stats = stats
        else:
            self.stats = self._load_stats(stats_path)

    def _load_stats(self, stats_path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Loads distribution stats from JSON file or computes default fallback."""
        candidates = [
            stats_path,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_stats.json"),
            "stream/training_stats.json",
        ]
        for path in candidates:
            if path and os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load stats from {path}: {e}")

        # Baseline default distribution parameters
        return {
            "fare_amount": {"mean": 7.92, "std": 2.81, "min": 1.8, "max": 15.42},
            "occupancy_rate": {"mean": 0.15, "std": 0.25, "min": 0.0, "max": 1.0},
            "delay_minutes": {"mean": 0.66, "std": 3.18, "min": 0.0, "max": 86.0},
            "boarding_count": {"mean": 1.24, "std": 2.39, "min": 0.0, "max": 51.0},
            "alighting_count": {"mean": 1.19, "std": 2.25, "min": 0.0, "max": 22.0},
            "passenger_count": {"mean": 65.36, "std": 140.79, "min": 0.0, "max": 800.0},
            "route_length_km": {"mean": 34.72, "std": 16.17, "min": 5.3, "max": 59.7},
            "temperature_c": {"mean": 17.13, "std": 6.95, "min": -5.67, "max": 37.81},
        }

    def generate_event(self) -> Dict[str, Any]:
        """Generates a single synthetic transit event dictionary."""
        city = str(self.rng.choice(self.cities))
        route_id = f"ROUTE_{self.rng.integers(100, 999)}"
        hour = int(self.rng.choice(24, p=HOUR_WEIGHTS))

        # 5% probability of injecting an anomaly
        is_anomaly = bool(self.rng.random() < 0.05)
        anomaly_flag = 1 if is_anomaly else 0
        anomaly_type = (
            str(self.rng.choice(["delay_spike", "crowd_surge", "weather_disruption", "equipment_fault"]))
            if is_anomaly
            else "none"
        )

        now_iso = datetime.now(timezone.utc).isoformat()

        event: Dict[str, Any] = {
            "city": city,
            "route_id": route_id,
            "feed_type": "synthetic",
            "event_ts": now_iso,
            "timestamp": now_iso,
            "hour": hour,
            "hour_sin": math.sin(2 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2 * math.pi * hour / 24.0),
            "anomaly_flag": anomaly_flag,
            "anomaly_type": anomaly_type,
        }

        # Draw numeric features from normal distributions
        skip_cols = {
            "anomaly_flag",
            "anomaly_type",
            "hour",
            "hour_sin",
            "hour_cos",
            "city",
            "route_id",
            "feed_type",
            "event_ts",
            "timestamp",
        }

        for col, col_stat in self.stats.items():
            if col in skip_cols:
                continue
            mu = col_stat.get("mean", 0.0)
            sigma = col_stat.get("std", 1.0)
            val = float(self.rng.normal(loc=mu, scale=sigma))
            event[col] = val

        # Handle target defaults if not already present in stats
        if "fare_amount" not in event:
            event["fare_amount"] = float(self.rng.normal(7.92, 2.81))
        if "occupancy_rate" not in event:
            event["occupancy_rate"] = float(self.rng.normal(0.15, 0.25))
        if "delay_minutes" not in event:
            event["delay_minutes"] = float(self.rng.normal(0.66, 3.18))

        # Modest adjustment under anomaly without breaking 3-sigma bound
        if is_anomaly:
            if anomaly_type == "delay_spike":
                event["delay_minutes"] = event["delay_minutes"] + 2.0
            elif anomaly_type == "crowd_surge":
                event["occupancy_rate"] = event["occupancy_rate"] + 0.1

        return event


def run_generator(
    api_url: str = "http://localhost:8000",
    delay_sec: float = 0.1,
    limit: Optional[int] = None,
    stats: Optional[Dict[str, Dict[str, float]]] = None,
    stats_path: Optional[str] = None,
) -> int:
    """Continuously generates events and posts them to the prediction API."""
    import requests

    gen = SyntheticGenerator(stats=stats, stats_path=stats_path)
    logger.info(f"Starting synthetic generator posting to {api_url} at ~{1.0 / delay_sec:.1f} events/sec...")

    sent_count = 0
    predict_endpoint = f"{api_url.rstrip('/')}/predict"

    while True:
        event = gen.generate_event()
        try:
            resp = requests.post(predict_endpoint, json=event, timeout=5)
            if resp.status_code == 200:
                sent_count += 1
                if sent_count % 50 == 0:
                    logger.info(f"Posted {sent_count} synthetic events successfully.")
            else:
                logger.warning(f"API returned HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Error posting synthetic event: {e}")

        if limit is not None and sent_count >= limit:
            logger.info(f"Reached limit of {limit} synthetic events.")
            break

        if delay_sec > 0:
            time.sleep(delay_sec)

    return sent_count


def main():
    parser = argparse.ArgumentParser(description="Synthetic Transit Data Generator")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Prediction API URL")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between events (sec)")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of events to generate")
    parser.add_argument("--stats-path", default=None, help="Path to training distribution stats JSON")

    args = parser.parse_args()
    run_generator(
        api_url=args.api_url,
        delay_sec=args.delay,
        limit=args.limit,
        stats_path=args.stats_path,
    )


if __name__ == "__main__":
    main()
