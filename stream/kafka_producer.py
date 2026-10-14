import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

import pandas as pd

# Compatibility bridge for kafka-python on Python 3.12+
try:
    from kafka import KafkaProducer
except ImportError:
    try:
        import six
        sys.modules.setdefault("kafka.vendor.six", six)
        sys.modules.setdefault("kafka.vendor.six.moves", six.moves)
        from kafka import KafkaProducer
    except Exception:
        KafkaProducer = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("kafka_producer")


def compute_fare_amount(row: Dict[str, Any]) -> float:
    """Calculates synthetic dynamic fare based on fare_type, distance, and occupancy."""
    fare_map = {
        "standard": 2.50,
        "concession": 1.50,
        "child": 1.00,
        "group": 6.00,
        "tourist_pass": 5.00,
        "monthly_pass": 3.00,
    }
    fare_type = str(row.get("fare_type", "standard")).lower()
    base_fare = fare_map.get(fare_type, 2.50)

    try:
        dist_val = row.get("route_length_km")
        dist = float(dist_val) if dist_val is not None and not pd.isna(dist_val) else 5.0
    except (ValueError, TypeError):
        dist = 5.0

    try:
        occ_val = row.get("occupancy_rate")
        occ = float(occ_val) if occ_val is not None and not pd.isna(occ_val) else 0.5
    except (ValueError, TypeError):
        occ = 0.5

    return round(base_fare + 0.15 * dist + 0.50 * occ, 2)


def sanitize_value(val: Any) -> Any:
    """Sanitizes numpy and pandas scalars to native JSON serializable types."""
    if val is None or pd.isna(val):
        return None
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return None
    return val


def emit_row(producer: Any, row: Union[pd.Series, Dict[str, Any]], topic: str = "transit_live") -> Dict[str, Any]:
    """Emits a single transit row to Kafka serialized as JSON.

    Args:
        producer: KafkaProducer or mock producer instance
        row: Row data as pandas Series or dict
        topic: Destination Kafka topic name

    Returns:
        Dict of the emitted payload
    """
    if hasattr(row, "to_dict"):
        raw_dict = row.to_dict()
    else:
        raw_dict = dict(row)

    payload = {k: sanitize_value(v) for k, v in raw_dict.items()}

    # Ensure fare_amount is present
    if payload.get("fare_amount") is None:
        payload["fare_amount"] = compute_fare_amount(payload)

    # Inject event_ts in ISO format UTC
    if not payload.get("event_ts"):
        payload["event_ts"] = datetime.now(timezone.utc).isoformat()

    json_payload = json.dumps(payload)
    producer.send(topic, json_payload)
    return payload


def get_default_producer(broker: str = "localhost:9092") -> Optional[Any]:
    """Instantiates a standard KafkaProducer if available."""
    if KafkaProducer is None:
        raise RuntimeError("kafka-python is not installed or failed to import.")
    return KafkaProducer(
        bootstrap_servers=broker,
        value_serializer=lambda v: str(v).encode("utf-8") if not isinstance(v, bytes) else v,
    )


def run_producer(
    csv_path: str = "archive/data/splits/transit_test.csv/test.csv",
    broker: str = "localhost:9092",
    topic: str = "transit_live",
    delay_sec: float = 0.1,
    limit: Optional[int] = None,
    producer: Optional[Any] = None,
) -> int:
    """Streams rows from CSV to Kafka at ~1/delay_sec events per second."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found at {csv_path}")

    if producer is None:
        producer = get_default_producer(broker=broker)

    logger.info(f"Streaming data from {csv_path} to topic '{topic}'...")

    sent_count = 0
    df_iter = pd.read_csv(csv_path, chunksize=1000)

    for chunk in df_iter:
        for _, row in chunk.iterrows():
            emit_row(producer, row, topic=topic)
            sent_count += 1

            if limit is not None and sent_count >= limit:
                logger.info(f"Reached limit of {limit} events sent.")
                if hasattr(producer, "flush"):
                    producer.flush()
                return sent_count

            if delay_sec > 0:
                time.sleep(delay_sec)

    if hasattr(producer, "flush"):
        producer.flush()

    logger.info(f"Finished streaming {sent_count} events to topic '{topic}'.")
    return sent_count


def main():
    parser = argparse.ArgumentParser(description="Real-time Transit Kafka Producer")
    parser.add_argument(
        "--csv-path",
        default="archive/data/splits/transit_test.csv/test.csv",
        help="Path to transit events CSV",
    )
    parser.add_argument("--broker", default="localhost:9092", help="Kafka broker address")
    parser.add_argument("--topic", default="transit_live", help="Kafka topic name")
    parser.add_argument("--delay", type=float, default=0.1, help="Delay between messages in seconds")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of events to emit")

    args = parser.parse_args()
    run_producer(
        csv_path=args.csv_path,
        broker=args.broker,
        topic=args.topic,
        delay_sec=args.delay,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

