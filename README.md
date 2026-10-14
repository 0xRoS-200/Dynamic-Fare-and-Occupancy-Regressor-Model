# Dynamic Fare & Occupancy Regressor

The Dynamic Fare & Occupancy Regressor is an end-to-end distributed transit analytics and real-time regression platform built on Apache Spark, HDFS, and Kafka. It leverages Spark MLlib for training multi-target regression models on transit ridership and pricing patterns, exposes low-latency real-time prediction endpoints via FastAPI, and visualizes dynamic fare adjustments and occupancy metrics through an interactive Streamlit dashboard.

## Prerequisites

- Java 11+
- Hadoop 3.3 (`HADOOP_HOME`)
- Kafka 3.x (`KAFKA_HOME`)
- Python 3.11

## Architecture

```
+--------------------+        +-----------------------+        +----------------------+
|  Raw Transit Data  | -----> |  Apache Spark / HDFS  | -----> |  MLlib Models        |
|  (CSV / Archive)   |        |  Feature Pipeline     |        |  (Fare & Occupancy)  |
+--------------------+        +-----------------------+        +----------------------+
                                                                          |
                                                                          v
+--------------------+        +-----------------------+        +----------------------+
|  Streamlit         | <----- |  Kafka Stream         | <----- |  FastAPI Server      |
|  Dashboard (:8501) |        |  (transit_live)       |        |  (:8000)             |
+--------------------+        +-----------------------+        +----------------------+
```

## Quick Start

```bash
pip install -r requirements.txt
make setup      # HDFS + Kafka init (uses local fallback on Windows)
make ingest     # CSV → HDFS Parquet
make features   # Feature engineering
make train      # Train 3 ML models
make serve      # Start prediction API on :8000
make stream     # Start Kafka producer + synthetic generator
make dashboard  # Open Streamlit on :8501
```

