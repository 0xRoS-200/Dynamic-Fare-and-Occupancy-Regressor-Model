# Dynamic Fare & Occupancy Regressor

The **Dynamic Fare & Occupancy Regressor** is an enterprise-grade distributed transit analytics and real-time regression platform built on Apache Spark, HDFS, and Apache Kafka. It leverages Spark MLlib for training multi-target regression models on transit ridership, congestion, and pricing dynamics, exposes sub-10ms real-time prediction endpoints via FastAPI and WebSockets, and visualizes live telemetry and pricing adjustments through an interactive Streamlit dashboard.

---

## Architecture Overview

```
                                      +---------------------------------------------+
                                      |            RAW TRANSIT ARCHIVE              |
                                      |  (CSV Telemetry: 570k+ records, 6 Cities)   |
                                      +---------------------------------------------+
                                                             |
                                                             v [Map Phase]
                                      +---------------------------------------------+
                                      |        HDFS / LOCAL PARQUET INGEST          |
                                      |           (spark/ingest.py)                 |
                                      +---------------------------------------------+
                                                             |
                                                             v [Reduce Phase]
                                      +---------------------------------------------+
                                      |     DISTRIBUTED FEATURE ENGINEERING         |
                                      |   - No-leakage rolling windows (1h, 6h, 24h)|
                                      |   - Cyclic trigonometric encodings          |
                                      |   - City-level normalization baselines      |
                                      +---------------------------------------------+
                                                             |
                                                             v
                                      +---------------------------------------------+
                                      |         SPARK MLLIB MODEL TRAINING          |
                                      |   - Fare Regressor (GBTRegressor)           |
                                      |   - Occupancy Regressor (GBTRegressor)      |
                                      |   - Delay Regressor (RandomForestRegressor) |
                                      |   - Experiment Tracking via MLflow          |
                                      +---------------------------------------------+
                                                             |
                                                             v
       +-----------------------------------------------------------------------------------------------+
       |                                   REAL-TIME SERVING & STREAMING                               |
       +-----------------------------------------------------------------------------------------------+
                 |                                      ^                                      ^
                 | REST / WS                            | Micro-batch                          | Events
                 v                                      | (foreachBatch)                       | (10/sec)
     +-----------------------+              +-----------------------+              +-----------------------+
     |   FastAPI Predictor   | <----------- |    Spark Streaming    | <----------- |     Kafka Stream      |
     |   POST /predict       |              |  (spark_streaming_    |              |     (transit_live)    |
     |   WS   /ws/stream     |              |      predict.py)      |              |   (kafka_producer.py) |
     |   GET  /health        |              +-----------------------+              +-----------------------+
     +-----------------------+                                                                 ^
                 |                                                                             |
                 v                                                                 +-----------------------+
     +-----------------------+                                                     |  Synthetic Generator  |
     |  SQLite Audit Store   | <-------------------------------------------------- | (synthetic_generator) |
     |  (api/predictions.db) |                                                     +-----------------------+
     +-----------------------+
                 |
                 v
     +------------------------------------------------------+
     |             STREAMLIT ANALYTICS DASHBOARD            |
     |       8 Live Plotly Visualizations (:8501)           |
     +------------------------------------------------------+
```

---

## Key Features

1. **Distributed MapReduce Ingestion & Feature Engineering**:
   - Zero-data-leakage temporal feature extraction using PySpark Window specifications (`rowsBetween(-(window_hours * 60), -1)`).
   - Trigonometric time encodings ($\sin / \cos$) for hour, day of week, day of year, and month.
   - City-level z-score standardization and demand norm features.
2. **Multi-Target Regressors**:
   - Dynamic Fare: Gradient-Boosted Trees (`GBTRegressor`, test RMSE $< 1.40$).
   - Transit Occupancy: Gradient-Boosted Trees (`GBTRegressor`, test $R^2 > 0.99$).
   - Travel Delay: Random Forest Regressor (`RandomForestRegressor`).
   - Unified MLflow experiment logging with complete metric tracking.
3. **Sub-10ms Inference Serving**:
   - FastAPI server with `POST /predict`, `GET /health`, and bidirectional `WS /ws/stream`.
   - Lazy SparkSession singleton with vector feature alignment to trained MLlib pipeline stages.
   - SQLite persistence layer with feed segmentation (`real` vs. `synthetic`).
4. **Real-time Kafka Feeds & Structured Streaming**:
   - Kafka producer streaming historical test telemetry with dynamic fare calculations and UTC timestamp injection.
   - Spark Structured Streaming consumer parsing 88-column event schemas and delegating micro-batch inference via HTTP.
5. **Probabilistic Synthetic Telemetry Generator**:
   - Marginal distribution matching ($N(\mu, \sigma)$) across all 88 feature columns derived from training datasets.
   - 5% realistic anomaly injection (delay spikes, crowd surges, weather events) bounded within $3\sigma$.
6. **Live Analytics Dashboard**:
   - Interactive Streamlit dashboard with real-time KPI metrics and 8 Plotly charts.

---

## Dataset & Data Sources

This platform models urban transit dynamics using the **Public Transportation Usage Dataset**, a large-scale multimodal transit dataset capturing hourly mobility patterns, demand cycles, delay propagation, and occupancy rates across 6 international metropolitan systems.

- **Primary Dataset Link (Kaggle)**: [Public Transportation Usage Dataset on Kaggle](https://www.kaggle.com/datasets/amith1707/public-transportation-usage-dataset)
- **Dataset Author**: Amith Nair
- **License**: [Creative Commons Zero (CC0 1.0) Public Domain](https://creativecommons.org/publicdomain/zero/1.0/)
- **Total Records**: 569,400 hourly observations (2020-01-01 to 2020-12-30)
- **Feature Space**: 88 attributes (demand counts, delays, occupancy, network graph metrics, cyclic temporal encodings)
- **Metropolitan Cities**: London, New York, Berlin, Tokyo, Paris, Singapore
- **Transit Modes**: Metro, Bus, Rail, Tram, Light Rail (LRT), Ferry

### Dataset Download & Acquisition

To download the dataset using the Kaggle API CLI:

```bash
# Ensure Kaggle CLI is installed and configured with your API credentials (~/.kaggle/kaggle.json)
kaggle datasets download -d amith1707/public-transportation-usage-dataset
unzip public-transportation-usage-dataset.zip -d archive/
```

Alternatively, download manually from the [Kaggle Dataset Page](https://www.kaggle.com/datasets/amith1707/public-transportation-usage-dataset) and unpack into the `archive/` directory.

### Local Archive Layout

The repository expects data under the local `archive/data/` directory (excluded from git tracking):

```
archive/
├── data/
│   ├── splits/
│   │   ├── transit_train.csv/train.csv    # 80% Chronological Train split (455,520 records)
│   │   ├── transit_val.csv/val.csv        # 10% Chronological Validation split (56,940 records)
│   │   └── transit_test.csv/test.csv      # 10% Chronological Test split (56,940 records)
│   ├── full/
│   │   └── transit_full.csv/processed.csv # Full consolidated transit telemetry
│   ├── features/
│   │   └── transit_features.csv/transit_fe.csv # Baseline feature matrix
│   └── subsets/
│       ├── transit_forecasting.csv
│       ├── transit_anomaly_detection.csv
│       └── Cities/ (Berlin, London, New York, Paris, Singapore, Tokyo)
├── metadata/
│   └── metadata.json                      # Dataset schema and specification metadata
└── README.md                              # Upstream dataset documentation
```

---

## Prerequisites

- **Java**: OpenJDK 11 or 17 (verified with JVM 64-bit)
- **Hadoop**: Apache Hadoop 3.3.x (`HADOOP_HOME` set; local fallback supported on Windows with winutils)
- **Kafka**: Apache Kafka 3.x (`KAFKA_HOME` optional; local broker on `localhost:9092`)
- **Python**: Python 3.11 or 3.12 with `pip`

---

## Project Structure

```
Dynamic Fare and Occupancy Regressor/
│
├── api/
│   ├── __init__.py
│   ├── predictor.py             # Lazy MLlib predictor wrapper and SQLite persistence
│   ├── server.py                # FastAPI REST & WebSocket streaming server
│   └── predictions.db           # SQLite inference audit log (git-ignored)
│
├── dashboard/
│   ├── __init__.py
│   ├── app.py                   # Streamlit multi-filter analytics application
│   ├── charts.py                # 8 interactive Plotly visualization builders
│   └── data_loader.py           # Database query interface and resilient loader
│
├── infra/
│   ├── check_infra.py           # Pre-flight infrastructure validation
│   ├── setup_hadoop.sh          # HDFS directory bootstrap script
│   └── setup_kafka.sh           # Kafka topic bootstrap script
│
├── spark/
│   ├── ingest.py                # Map Phase: Raw CSV to partitioned Parquet
│   ├── feature_engineering.py   # Reduce Phase: Rolling aggregations & cyclic encodings
│   ├── train_pipeline.py        # Spark MLlib GBT & RF model training with MLflow
│   └── batch_evaluate.py        # Standalone model test split evaluator
│
├── stream/
│   ├── __init__.py
│   ├── kafka_producer.py        # Real-time event serialization & Kafka emission
│   ├── spark_streaming_predict.py # Spark Structured Streaming foreachBatch consumer
│   ├── synthetic_generator.py   # Probabilistic telemetry generator with 3-sigma bounds
│   └── training_stats.json      # Pre-computed per-feature statistical baselines
│
├── tests/
│   ├── test_check_infra.py
│   ├── test_ingest.py
│   ├── test_feature_engineering.py
│   ├── test_train_pipeline.py
│   ├── test_predictor.py
│   ├── test_kafka_producer.py
│   ├── test_synthetic_generator.py
│   ├── test_dashboard_data_loader.py
│   └── test_integration.py      # End-to-end multi-stage pipeline integration test
│
├── Makefile                     # Developer lifecycle commands
├── requirements.txt             # Locked Python dependencies
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository and install dependencies
git clone https://github.com/0xRoS-200/Dynamic-Fare-and-Occupancy-Regressor-Model.git
cd "Dynamic-Fare-and-Occupancy-Regressor-Model"
pip install -r requirements.txt
```

### 🚀 1-Command All-In-One Run (Fastest)

To launch the entire end-to-end platform with a single command:

```bash
python run.py
```
*(or run `make run` or execute `.\run.bat` on Windows)*

This unified command automatically:
1. Validates pre-trained models exist in `hdfs_local/transit/models/` (trains them if missing).
2. Starts the **FastAPI prediction server** on `http://localhost:8000`.
3. Polls `/health` until the inference engine is ready.
4. Launches the **Streamlit analytics dashboard** on `http://localhost:8501`.
5. Starts the **synthetic telemetry generator** to feed live predictions.
6. Opens the dashboard in your default browser.
7. Gracefully cleans up all background processes on `Ctrl+C`.

### 2. Infrastructure Validation

```bash
# Validate local HDFS fallback and Kafka environment
make setup
# Or execute directly:
python infra/check_infra.py --local-fallback
```

### 3. Ingestion (Map Phase)

```bash
# Ingest raw split CSVs to Parquet (drops null target rows, casts schema)
make ingest
# Or directly:
python spark/ingest.py --local-fallback
```

### 4. Feature Engineering (Reduce Phase)

```bash
# Generate 100+ numeric features with rolling windows and cyclic encodings
make features
# Or directly:
python spark/feature_engineering.py --local-fallback
```

### 5. Model Training & Evaluation

```bash
# Train GBT fare/occupancy models and Random Forest delay model
make train
# Or directly:
python spark/train_pipeline.py --local-fallback --max-iter 50

# Evaluate models on test dataset:
python spark/batch_evaluate.py --local-fallback --report evaluation_report.json
```

### 6. Start Prediction Server

```bash
# Launch FastAPI on port 8000
make serve
# Or directly:
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### 7. Run Real-time Feeds

```bash
# Option A: Stream real dataset to Kafka
make stream

# Option B: Run probabilistic synthetic generator posting to the API
python stream/synthetic_generator.py --api-url http://localhost:8000 --delay 0.1
```

### 8. Launch Live Analytics Dashboard

```bash
# Start Streamlit dashboard at http://localhost:8501
make dashboard
# Or directly:
streamlit run dashboard/app.py
```

---

## Dashboard Visualizations

The Streamlit analytics dashboard renders 8 interactive Plotly charts:
1. **Real vs. Predicted Stream**: Dynamic line chart tracking predicted fare, occupancy, or delay over event sequence.
2. **Feed Comparison (Real vs. Synthetic)**: Grouped bar chart comparing metric averages across feed origins.
3. **MLlib Feature Importances**: Horizontal bar chart presenting relative predictor weights.
4. **Occupancy Density Heatmap**: 2D diurnal heatmap displaying passenger density across hours (0–23) and metropolitan cities.
5. **Rolling Volatility Index**: Dynamic gauge indicator highlighting prediction stability and deviation.
6. **Anomaly & Delay Timeline**: Multi-dimensional scatter plot correlating delay severity with fare pricing.
7. **City Delay Distributions**: Comparative violin plots capturing delay variance across cities.
8. **Fare Spread Boxplot**: Box-and-whisker plot displaying fare dispersion and median tickets per metropolitan area.

---

## Verification & Testing

The repository includes a comprehensive test suite covering all units, regression checks, and an end-to-end pipeline test:

```bash
# Run all 31 unit, regression, and integration tests
pytest -v
```

Test breakdown:
- `tests/test_check_infra.py`: HDFS and Kafka connectivity checks
- `tests/test_ingest.py`: Parquet conversion, schema casting, and null target row filtering
- `tests/test_feature_engineering.py`: Trigonometric encoding, rolling window aggregation, and zero-leakage assertions
- `tests/test_train_pipeline.py`: Model fitting, persistence, and fare RMSE threshold verification ($< 5.0$)
- `tests/test_predictor.py`: REST and WebSocket prediction endpoints and SQLite persistence
- `tests/test_kafka_producer.py`: Kafka serialization, rate limiting, and structured streaming batch logic
- `tests/test_synthetic_generator.py`: Empirical distribution matching ($< 3\sigma$) and anomaly injection rate
- `tests/test_dashboard_data_loader.py`: Database query interface and all 8 Plotly chart renderings
- `tests/test_integration.py`: Complete multi-stage end-to-end pipeline execution
