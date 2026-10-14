import argparse
import logging
import os
import sys
from typing import Optional

# Windows winutils / HADOOP_HOME setup if needed
if sys.platform == "win32" and "HADOOP_HOME" not in os.environ:
    for candidate in [
        r"C:\temp\winutils\hadoop-3.3.6",
        r"C:\temp\winutils\hadoop-3.3.5",
        r"C:\temp\winutils\hadoop-3.1.0",
        r"C:\temp\winutils\hadoop-3.0.0",
    ]:
        if os.path.exists(candidate):
            os.environ["HADOOP_HOME"] = candidate
            bin_path = os.path.join(candidate, "bin")
            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
            break

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import requests
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from api.predictor import insert_prediction
from spark.ingest import get_or_create_spark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spark_streaming_predict")

STRING_COLUMNS = [
    "trip_id",
    "timestamp",
    "route_id",
    "city",
    "route_type",
    "origin_station",
    "dest_station",
    "service_status",
    "fare_type",
    "weather_condition",
    "anomaly_type",
    "season",
    "event_ts",
]

NUMERIC_COLUMNS = [
    "boarding_count",
    "alighting_count",
    "passenger_count",
    "occupancy_rate",
    "delay_minutes",
    "actual_travel_min",
    "scheduled_travel_min",
    "cancellation_flag",
    "temperature_c",
    "route_length_km",
    "headway_min",
    "vehicle_capacity",
    "operates_night",
    "is_electric",
    "is_peak_hour",
    "is_weekend",
    "is_holiday",
    "is_school_hours",
    "is_night_service",
    "anomaly_flag",
    "hour",
    "day_of_week",
    "month",
    "year",
    "day_of_year",
    "quarter",
    "week_of_year",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "board_6h_mean",
    "board_6h_std",
    "board_6h_max",
    "occ_6h_mean",
    "delay_6h_mean",
    "board_24h_mean",
    "board_24h_std",
    "board_24h_max",
    "occ_24h_mean",
    "delay_24h_mean",
    "board_7d_mean",
    "board_7d_std",
    "board_7d_max",
    "occ_7d_mean",
    "delay_7d_mean",
    "board_lag_1",
    "occ_lag_1",
    "board_lag_24",
    "occ_lag_24",
    "board_lag_168",
    "occ_lag_168",
    "flow_imbalance",
    "board_roc_1h",
    "board_roc_24h",
    "board_acc",
    "occ_volatility",
    "delay_cumsum_4h",
    "route_popularity",
    "efficiency_pax_km",
    "reliability_score",
    "weather_adj_demand",
    "is_crowded",
    "is_overcrowded",
    "rush_intensity",
    "transfer_prob",
    "on_time_flag",
    "consistency_score",
    "board_zscore",
    "is_board_anomaly",
    "city_hour_mean_board",
    "seasonal_deviation",
    "fare_amount",
]

STREAMING_SCHEMA = StructType(
    [StructField(c, StringType(), True) for c in STRING_COLUMNS]
    + [StructField(c, DoubleType(), True) for c in NUMERIC_COLUMNS]
)


def process_batch(
    batch_df: DataFrame,
    batch_id: int,
    api_url: str = "http://localhost:8000",
    db_path: Optional[str] = None,
):
    """Processes a micro-batch by posting events to prediction API and logging to SQLite."""
    rows = batch_df.collect()
    if not rows:
        return

    logger.info(f"Processing micro-batch {batch_id} with {len(rows)} events...")

    for row in rows:
        payload = row.asDict()
        # Clean null values
        cleaned_payload = {k: v for k, v in payload.items() if v is not None}
        cleaned_payload["feed_type"] = "real"

        try:
            resp = requests.post(f"{api_url.rstrip('/')}/predict", json=cleaned_payload, timeout=10)
            if resp.status_code == 200:
                result = resp.json()
                if db_path:
                    # Write prediction to explicit db_path if provided
                    record = {
                        "timestamp": result.get("timestamp", cleaned_payload.get("event_ts", "")),
                        "city": cleaned_payload.get("city", "Unknown"),
                        "route_id": cleaned_payload.get("route_id", "Unknown"),
                        "fare_pred": float(result.get("fare", 0.0)),
                        "occupancy_pred": float(result.get("occupancy", 0.0)),
                        "delay_pred": float(result.get("delay", 0.0)),
                        "feed_type": "real",
                    }
                    insert_prediction(result=record, db_path=db_path)
            else:
                logger.warning(f"Predict API returned HTTP {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Error calling predict API: {e}")


def create_streaming_query(
    spark,
    bootstrap_servers: str = "localhost:9092",
    topic: str = "transit_live",
    api_url: str = "http://localhost:8000",
    checkpoint_dir: Optional[str] = None,
    db_path: Optional[str] = None,
):
    """Creates a Structured Streaming query that consumes Kafka and runs batch prediction."""
    stream_df = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap_servers)
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed_df = stream_df.select(
        from_json(col("value").cast("string"), STREAMING_SCHEMA).alias("data")
    ).select("data.*")

    writer = parsed_df.writeStream.foreachBatch(
        lambda df, bid: process_batch(df, bid, api_url=api_url, db_path=db_path)
    )

    if checkpoint_dir:
        writer = writer.option("checkpointLocation", checkpoint_dir)

    return writer.start()


def main():
    parser = argparse.ArgumentParser(description="Spark Structured Streaming Prediction Job")
    parser.add_argument("--broker", default="localhost:9092", help="Kafka broker address")
    parser.add_argument("--topic", default="transit_live", help="Kafka topic name")
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPI predictor URL")
    parser.add_argument("--db-path", default=None, help="Explicit path to SQLite database")
    parser.add_argument("--checkpoint-dir", default="./hdfs_local/transit/checkpoints", help="Checkpoint directory")

    args = parser.parse_args()

    spark = get_or_create_spark("transit_structured_streaming")
    query = create_streaming_query(
        spark=spark,
        bootstrap_servers=args.broker,
        topic=args.topic,
        api_url=args.api_url,
        checkpoint_dir=args.checkpoint_dir,
        db_path=args.db_path,
    )
    logger.info("Structured Streaming query started. Awaiting termination...")
    query.awaitTermination()


if __name__ == "__main__":
    main()

