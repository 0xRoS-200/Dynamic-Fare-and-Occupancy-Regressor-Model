# Dynamic Fare & Occupancy Regressor — Design Specification

> **Status:** Draft · 2026-10-14

---

## Intent & Why

Public transit systems generate massive volumes of time-stamped, route-level observations covering fares, occupancy, delays, and passenger flow. Operators and planners need a **live, continuously-updating regressor** that can:

1. Learn from the full historical dataset (569,400 rows × 88 columns across 6 cities) using distributed big-data techniques.
2. Accept **real-time streaming records** and produce predictions of fare, occupancy rate, and delay for any incoming trip event.
3. Compare model predictions against **synthetic baseline data** to expose distribution drift and model error visually.
4. Deliver an **analytics dashboard** with live-updating charts so planners can act on insights without touching code.

---

## Goals

- Train distributed regression models (fare, occupancy, delay) on the full dataset via Apache Spark + HDFS.
- Serve real-time predictions over a REST/WebSocket endpoint.
- Ingest a synthetic real-time data stream (simulated sensor feed) and compare it to model predictions.
- Visualise analytics in a web dashboard: feature importances, error distributions, per-city/mode breakdowns, rolling real vs. predicted charts.

## Non-Goals

- Graph Neural Network (GNN) training (dataset supports it, but out of scope here).
- Production Kubernetes deployment or multi-node HDFS cluster (single-node pseudo-distributed is the target).
- Fare pricing optimisation / RL — prediction only.

## Anti-Goals

- A dashboard that requires manual refresh — all charts must update live.
- A monolithic notebook that cannot be run in parts — pipeline must be modular.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Distributed processing | Apache Spark 3.5 (PySpark), HDFS (Hadoop 3.3 pseudo-distributed) |
| Big-data ingestion | PySpark DataFrames + MapReduce-style transforms |
| ML training | Spark MLlib (GBTRegressor, RandomForestRegressor), MLflow tracking |
| Real-time streaming | Apache Kafka + Spark Structured Streaming |
| Synthetic data generator | Python `faker` + NumPy probabilistic model |
| Prediction API | FastAPI (WebSocket + REST) |
| Dashboard | Streamlit + Plotly (auto-refreshing) |
| Storage | HDFS for raw/processed parquet; local SQLite for live predictions log |
| Orchestration | Python scripts + Makefile; optional Airflow DAG |
| Language | Python 3.11 |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  DATA LAYER (HDFS)                                                   │
│  archive/data/  ──► hdfs ingest ──► /transit/raw/*.parquet           │
│                                ──► /transit/features/*.parquet       │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ Spark read
┌────────────────────────────▼─────────────────────────────────────────┐
│  TRAINING PIPELINE  (spark/train_pipeline.py)                        │
│  MapReduce pre-processing → Feature engineering → MLlib GBT train   │
│  → model saved to HDFS /transit/models/  + MLflow registry          │
└────────────────────────────┬─────────────────────────────────────────┘
                             │ model artefact
┌────────────────────────────▼─────────────────────────────────────────┐
│  PREDICTION SERVER  (api/server.py  — FastAPI)                       │
│  REST  POST /predict   ← single event JSON                           │
│  WS    /ws/stream      ← live prediction stream                      │
│  stores results → predictions.db (SQLite)                            │
└───────────┬────────────────────────────────┬─────────────────────────┘
            │ real events                    │ synthetic events
┌───────────▼──────────────┐    ┌────────────▼──────────────────────── ┐
│ REAL-TIME FEED           │    │ SYNTHETIC GENERATOR                   │
│ Kafka producer           │    │ stream/synthetic_generator.py         │
│ (replays archive data    │    │ probabilistic model seeded from       │
│  with live timestamps)   │    │ dataset statistics                    │
└──────────────────────────┘    └───────────────────────────────────────┘
                             │ WebSocket frames (real + synthetic)
┌────────────────────────────▼─────────────────────────────────────────┐
│  ANALYTICS DASHBOARD  (dashboard/app.py  — Streamlit)                │
│  • Live: Real vs Predicted line chart (fare, occupancy, delay)       │
│  • Comparison: Real-feed vs Synthetic-feed prediction error          │
│  • Feature importances bar chart                                     │
│  • City / mode heatmap (occupancy by hour-of-week)                  │
│  • Rolling MAE / RMSE gauges                                         │
│  • Anomaly flag timeline                                             │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Details

### Historical Ingestion (MapReduce pattern)
1. **Map phase** — partition raw CSVs by city + date; cast types; drop nulls.
2. **Reduce phase** — aggregate rolling stats (1h/6h/24h mean/std per route); encode cyclic temporal features (sin/cos of hour, day-of-week).
3. **Output** — feature-engineered Parquet files on HDFS partitioned by `city/date`.

### Real-Time Feed
- A Kafka producer replays `archive/data/splits/transit_test.csv/test.csv` row-by-row at ~10 records/sec with live UTC timestamps injected.
- Spark Structured Streaming consumes the topic, applies the saved feature transformer, calls the MLlib model, and writes predictions to SQLite.

### Synthetic Generator
- Mirrors the archive's statistical distribution (mean/std per city, mode, hour) with added Gaussian noise and injected anomalies.
- Runs concurrently with the real feed so the dashboard can show side-by-side drift.

---

## Regression Targets

| Target | Model | Metric |
|---|---|---|
| `fare_amount` | GBTRegressor | RMSE, R² |
| `occupancy_rate` | GBTRegressor | RMSE, MAE |
| `delay_minutes` | RandomForestRegressor | RMSE, MAE |

---

## Analytics Charts (Dashboard)

| Chart | Description |
|---|---|
| Live Real vs Predicted | Line chart — actual vs model output per minute |
| Real vs Synthetic Error | Side-by-side bar chart of prediction error for real vs synthetic feed |
| Feature Importances | Horizontal bar — top 20 features per model |
| City Heatmap | Occupancy by hour-of-week, per city |
| Rolling RMSE/MAE | Gauge charts updating every 30 s |
| Anomaly Timeline | Scatter plot — anomaly_flag events over time |
| Delay Distribution | Violin plot — delay_minutes by transit mode |
| Fare Box Plot | Box plot — fare by city and fare_type |

---

## File Structure

```
Dynamic Fare and Occupancy Regressor/
├── archive/                        # existing dataset (read-only)
├── docs/
│   └── superpowers/
│       ├── specs/2026-10-14-dynamic-fare-regressor-design.md
│       └── plans/2026-10-14-dynamic-fare-regressor-plan.md
├── infra/
│   ├── setup_hadoop.sh             # HDFS pseudo-distributed setup script
│   └── setup_kafka.sh              # Kafka single-broker setup script
├── spark/
│   ├── ingest.py                   # CSV → HDFS parquet (MapReduce)
│   ├── feature_engineering.py      # PySpark transforms (map/reduce stages)
│   ├── train_pipeline.py           # MLlib model training + MLflow logging
│   └── batch_evaluate.py           # batch evaluation on test split
├── stream/
│   ├── kafka_producer.py           # replays test.csv → Kafka topic
│   ├── spark_streaming_predict.py  # Structured Streaming consumer
│   └── synthetic_generator.py     # probabilistic synthetic feed
├── api/
│   ├── server.py                   # FastAPI REST + WebSocket server
│   ├── predictor.py                # model loading + single-event inference
│   └── predictions.db              # SQLite log (auto-created)
├── dashboard/
│   ├── app.py                      # Streamlit dashboard entry point
│   ├── charts.py                   # Plotly chart builders
│   └── data_loader.py              # reads SQLite + live WebSocket
├── tests/
│   ├── test_feature_engineering.py
│   ├── test_predictor.py
│   └── test_synthetic_generator.py
├── Makefile                        # one-command bootstrap
├── requirements.txt
└── README.md
```

---

## Constraints

- Python 3.11; PySpark 3.5; Spark MLlib only (no scikit-learn for training).
- HDFS pseudo-distributed on localhost (single JVM, no true cluster needed).
- Kafka 3.x single broker, topic `transit_live`.
- MLflow tracking server runs locally (sqlite backend).
- Dashboard auto-refreshes every 5 seconds via Streamlit `st.rerun()`.
- All monetary values in dataset currency units (not normalised to USD).
- No PII — dataset is synthetic/anonymised.

