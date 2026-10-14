# Dynamic Fare & Occupancy Regressor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a distributed big-data regression system that trains on 569K-row transit data via Apache Spark + HDFS, serves real-time predictions via FastAPI, ingests both a live Kafka feed and a synthetic generator concurrently, and displays all analytics in a live-updating Streamlit dashboard.

**Architecture:** PySpark reads the archive CSVs into HDFS as Parquet, applies MapReduce-style feature engineering, and trains GBT/RF regressors via Spark MLlib. A FastAPI server loads the saved models and exposes REST + WebSocket prediction endpoints. Two concurrent producers (Kafka replay + synthetic generator) feed the server; results are logged to SQLite. A Streamlit dashboard polls SQLite and renders live Plotly charts.

**Tech Stack:** Python 3.11, PySpark 3.5, Hadoop HDFS 3.3 (pseudo-distributed), Kafka 3.x, Spark MLlib, MLflow, FastAPI, Streamlit, Plotly, SQLite, pytest

**Spec:** `docs/superpowers/specs/2026-10-14-dynamic-fare-regressor-design.md`

---

## Global Constraints

- Python 3.11; PySpark 3.5.x; Kafka 3.x; Streamlit ≥ 1.35
- HDFS pseudo-distributed on localhost:9000; NameNode UI on port 9870
- Kafka broker on localhost:9092; topic name `transit_live`; consumer group `regressor`
- MLflow tracking URI `sqlite:///mlflow.db`; experiment name `transit_regressor`
- Dashboard auto-refresh interval: 5 seconds (`st.rerun()`)
- Prediction log: `api/predictions.db` (SQLite, table `predictions`)
- Three regression targets: `fare_amount`, `occupancy_rate`, `delay_minutes`
- Archive data root: `archive/data/`; train split: `archive/data/splits/transit_train.csv/train.csv`
- All paths use POSIX-style forward slashes inside PySpark/HDFS URIs
- Commits: one per task using `feat: <task>` convention

---

## Review Focus

1. **Streaming schema mismatch** — Kafka producer emits JSON with snake_case keys; Structured Streaming consumer must parse the same schema or it silently drops records. Covered in Task 6 test.
2. **Rolling feature leakage** — rolling mean/std must be computed over a window anchored at `(current_row - 1)` to avoid using the current label. Covered in Task 3 test.
3. **Synthetic vs real distribution drift** — if the synthetic generator's `mean`/`std` parameters diverge from actual training distribution, dashboard comparison is misleading. Task 8 test asserts synthetic stats are within 3σ of training stats.
4. **Model cold-start on empty DB** — dashboard must handle zero rows in `predictions` table without crashing. Covered in Task 9 test.
5. **HDFS not available on Windows without WSL** — `infra/setup_hadoop.sh` must print a clear error and offer a local-filesystem fallback path so the pipeline still runs. Covered in Task 1 test.

---

## Task 1: Infrastructure Bootstrap

**Files:**
- Create: `infra/setup_hadoop.sh`
- Create: `infra/setup_kafka.sh`
- Create: `infra/check_infra.py`
- Create: `Makefile`
- Create: `requirements.txt`
- Create: `README.md`

**Interfaces:**
- Produces: `check_infra.py` exits 0 when HDFS + Kafka are reachable, 1 otherwise with clear error messages; used by every subsequent task as a pre-flight check.

---

- [ ] **Step 1: Write `requirements.txt`**

```
pyspark==3.5.3
kafka-python==2.0.2
mlflow==2.15.1
fastapi==0.115.0
uvicorn[standard]==0.30.6
streamlit==1.38.0
plotly==5.24.0
pandas==2.2.3
numpy==1.26.4
faker==28.0.0
pytest==8.3.3
httpx==0.27.2
websockets==13.0
```

- [ ] **Step 2: Run `pip install -r requirements.txt`** and verify no conflicts.

- [ ] **Step 3: Write `infra/setup_hadoop.sh`**

  Script checks if `HADOOP_HOME` is set and `hdfs` is on `PATH`; if not, prints install URL and exits 1. If yes: formats namenode (if first run), starts HDFS daemons, creates `/transit/{raw,features,models}` directories on HDFS, and prints "HDFS ready". Includes a `--local-fallback` flag that writes to `./hdfs_local/` instead (for Windows without WSL).

- [ ] **Step 4: Write `infra/setup_kafka.sh`**

  Checks if `KAFKA_HOME` is set; if not prints install URL and exits 1. Starts ZooKeeper + Kafka broker, creates topic `transit_live` with 3 partitions / 1 replica, prints "Kafka ready".

- [ ] **Step 5: Write `infra/check_infra.py`**

```python
def check_hdfs(local_fallback: bool) -> bool: ...  # returns True if path accessible
def check_kafka(broker: str, topic: str) -> bool: ...
if __name__ == "__main__":
    # parse --local-fallback flag; exit 0 or 1
```

- [ ] **Step 6: Write failing test `tests/test_check_infra.py`**

```python
def test_check_hdfs_local_fallback_always_true():
    assert check_hdfs(local_fallback=True)

def test_check_kafka_bad_broker_returns_false():
    assert not check_kafka("localhost:1", "transit_live")
```

- [ ] **Step 7: Run `pytest tests/test_check_infra.py -v`** — expect FAIL (functions not defined).
- [ ] **Step 8: Implement functions in `infra/check_infra.py`**
- [ ] **Step 9: Run `pytest tests/test_check_infra.py -v`** — expect PASS.
- [ ] **Step 10: Write `Makefile`**

```makefile
setup:
    bash infra/setup_hadoop.sh --local-fallback
    bash infra/setup_kafka.sh
    python infra/check_infra.py --local-fallback

ingest:
    python spark/ingest.py --local-fallback

train:
    python spark/train_pipeline.py --local-fallback

serve:
    uvicorn api.server:app --reload --port 8000

stream:
    python stream/kafka_producer.py &
    python stream/synthetic_generator.py &
    python stream/spark_streaming_predict.py

dashboard:
    streamlit run dashboard/app.py
```

- [ ] **Step 11: Write `README.md`** with quick-start section referencing Makefile targets.
- [ ] **Step 12: Commit**

```bash
git add infra/ Makefile requirements.txt README.md tests/test_check_infra.py
git commit -m "feat: infrastructure bootstrap and pre-flight checks"
```

---

## Task 2: HDFS Data Ingestion (MapReduce — Map Phase)

**Files:**
- Create: `spark/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `archive/data/splits/transit_train.csv/train.csv`, `transit_val.csv/val.csv`, `transit_test.csv/test.csv`, `archive/data/full/transit_full.csv/processed.csv`
- Produces: Parquet files at `hdfs://localhost:9000/transit/raw/{train,val,test,full}.parquet` (or `./hdfs_local/transit/raw/` in fallback); `ingest_report.json` with row counts and schema summary.

---

- [ ] **Step 1: Write failing test**

```python
def test_ingest_produces_parquet_with_expected_schema(spark, tmp_path):
    # calls ingest.run(src="archive/data/splits/transit_train.csv/train.csv",
    #                  dest=str(tmp_path), split="train")
    df = spark.read.parquet(str(tmp_path / "train.parquet"))
    assert "fare_amount" in df.columns
    assert "occupancy_rate" in df.columns
    assert "delay_minutes" in df.columns
    assert df.count() > 0
```

- [ ] **Step 2: Run test → FAIL**
- [ ] **Step 3: Implement `ingest.run(src, dest, split)` in `spark/ingest.py`**

  Uses `SparkSession.builder.appName("transit_ingest")`, reads CSV with `inferSchema=True, header=True`. **Map phase:** `select` + `cast` all numeric columns to `DoubleType`, parse `timestamp` to `TimestampType`. Drop rows where any target column is null. Write as Parquet (overwrite mode) to `dest/{split}.parquet`.

- [ ] **Step 4: Add CLI entry point** — `argparse` with `--src`, `--dest`, `--split`, `--local-fallback`; if `--local-fallback`, dest is `./hdfs_local/transit/raw/`.
- [ ] **Step 5: Run test → PASS**
- [ ] **Step 6: Run manual ingest for train/val/test/full** and print `ingest_report.json`.
- [ ] **Step 7: Commit**

```bash
git add spark/ingest.py tests/test_ingest.py
git commit -m "feat: HDFS parquet ingestion (map phase)"
```

---

## Task 3: Feature Engineering (MapReduce — Reduce Phase)

**Files:**
- Create: `spark/feature_engineering.py`
- Test: `tests/test_feature_engineering.py`

**Interfaces:**
- Consumes: Parquet from Task 2 (`/transit/raw/{split}.parquet`)
- Produces: Feature-engineered Parquet at `/transit/features/{split}.parquet`; `FeatureTransformer` class that can be serialised via `save(path)` / `load(path)`.

---

- [ ] **Step 1: Write failing tests**

```python
def test_cyclic_hour_encoding_range(spark, sample_df):
    df = add_cyclic_features(sample_df)
    assert df.filter((col("hour_sin") < -1) | (col("hour_sin") > 1)).count() == 0

def test_rolling_mean_no_leakage(spark, sample_df):
    # row at index i must not use row i's own value
    df = add_rolling_features(sample_df, window_hours=1)
    # rolling mean at second row ≠ current row value when there's variance
    row1, row2 = df.orderBy("timestamp").take(2)
    assert row1["rolling_mean_1h_boarding"] != row2["boarding_count"]

def test_feature_transformer_roundtrip(spark, tmp_path, sample_df):
    t = FeatureTransformer()
    t.fit(sample_df)
    t.save(str(tmp_path))
    t2 = FeatureTransformer.load(str(tmp_path))
    out = t2.transform(sample_df)
    assert "fare_amount" in out.columns
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `add_cyclic_features(df)` in `spark/feature_engineering.py`**

  Uses `pyspark.sql.functions`: `sin(col("hour") * 2π/24)`, `cos(...)`, `sin(col("day_of_week") * 2π/7)`, `cos(...)`, `sin(col("month") * 2π/12)`, `cos(...)`.

- [ ] **Step 4: Implement `add_rolling_features(df, window_hours)` in same file**

  Uses `Window.partitionBy("route_id").orderBy("timestamp").rowsBetween(-(window_hours*60), -1)`. Computes `rolling_mean_{1,6,24}h_boarding`, `rolling_std_{1,6,24}h_boarding`.

- [ ] **Step 5: Implement `FeatureTransformer` class**

  `fit(df)` computes per-city mean/std for normalisation; `transform(df)` applies all transforms; `save(path)` writes stats as JSON + PySpark `Pipeline.save`; `load(path)` is classmethod.

- [ ] **Step 6: Run tests → PASS**
- [ ] **Step 7: Add CLI** — reads from `/transit/raw/{split}.parquet`, writes to `/transit/features/{split}.parquet`, saves transformer to `/transit/models/transformer/`.
- [ ] **Step 8: Commit**

```bash
git add spark/feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat: MapReduce feature engineering with no-leakage rolling windows"
```

---

## Task 4: Spark MLlib Model Training

**Files:**
- Create: `spark/train_pipeline.py`
- Create: `spark/batch_evaluate.py`
- Test: `tests/test_train_pipeline.py`

**Interfaces:**
- Consumes: `/transit/features/train.parquet` (Task 3 output)
- Produces: Three saved Spark ML models at `/transit/models/{fare,occupancy,delay}_model/`; MLflow run with RMSE/MAE/R² metrics; `evaluation_report.json`.

---

- [ ] **Step 1: Write failing tests**

```python
def test_models_are_saved(tmp_path, spark, small_feature_df):
    train_all(small_feature_df, model_dir=str(tmp_path))
    assert (tmp_path / "fare_model").exists()
    assert (tmp_path / "occupancy_model").exists()
    assert (tmp_path / "delay_model").exists()

def test_fare_model_rmse_below_threshold(tmp_path, spark, feature_df):
    metrics = evaluate_model(spark, str(tmp_path / "fare_model"),
                             target="fare_amount", test_path=...)
    assert metrics["rmse"] < 5.0  # currency units
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `train_all(df, model_dir)` in `spark/train_pipeline.py`**

  For each target (`fare_amount`, `occupancy_rate`, `delay_minutes`):
  - Assemble features with `VectorAssembler` (all numeric columns minus targets).
  - `GBTRegressor(featuresCol="features", labelCol=target, maxIter=50)` for fare/occupancy; `RandomForestRegressor(maxIter=50)` for delay.
  - Wrap in `Pipeline([assembler, regressor])`.
  - `.fit(train_df)` then `.save(model_dir/<target>_model)`.
  - Log to MLflow: params (`maxIter`, `maxDepth`), metrics (RMSE, MAE, R² on val set).

- [ ] **Step 4: Implement `evaluate_model(spark, model_path, target, test_path)` in `spark/batch_evaluate.py`**

  Loads saved model, transforms test Parquet, computes `RegressionEvaluator` metrics, returns dict `{"rmse": float, "mae": float, "r2": float}`.

- [ ] **Step 5: Run tests → PASS**
- [ ] **Step 6: Train full models** via CLI: `python spark/train_pipeline.py --local-fallback`; record MLflow run IDs.
- [ ] **Step 7: Commit**

```bash
git add spark/train_pipeline.py spark/batch_evaluate.py tests/test_train_pipeline.py
git commit -m "feat: Spark MLlib GBT/RF training with MLflow tracking"
```

---

## Task 5: Prediction API (FastAPI)

**Files:**
- Create: `api/predictor.py`
- Create: `api/server.py`
- Test: `tests/test_predictor.py`

**Interfaces:**
- Consumes: Saved models from Task 4 (`/transit/models/`); single-event dict with all feature columns.
- Produces:
  - `POST /predict` → `{"fare": float, "occupancy": float, "delay": float, "timestamp": str}`
  - `GET /health` → `{"status": "ok", "models_loaded": true}`
  - `WS /ws/stream` → sends same prediction JSON for every event pushed to server
  - Writes each prediction row to `api/predictions.db` table `predictions`.

---

- [ ] **Step 1: Write failing tests**

```python
def test_predictor_returns_three_floats(loaded_predictor, sample_event):
    result = loaded_predictor.predict(sample_event)
    assert isinstance(result["fare"], float)
    assert isinstance(result["occupancy"], float)
    assert isinstance(result["delay"], float)

def test_post_predict_endpoint(test_client, sample_event):
    r = test_client.post("/predict", json=sample_event)
    assert r.status_code == 200
    assert "fare" in r.json()

def test_health_endpoint(test_client):
    r = test_client.get("/health")
    assert r.json()["status"] == "ok"
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `Predictor` class in `api/predictor.py`**

  `__init__(model_dir)`: loads three `PipelineModel.load(...)`. `predict(event: dict) -> dict`: converts event to single-row Spark DataFrame (lazy SparkSession singleton), runs each model, extracts `.prediction` scalar, returns dict.

  > Use a singleton SparkSession with `getOrCreate()` to avoid JVM startup cost on every request.

- [ ] **Step 4: Implement `api/server.py` (FastAPI)**

  `POST /predict`: calls `predictor.predict(body)`, inserts row into SQLite, returns JSON.
  `GET /health`: returns status.
  `WS /ws/stream`: accepts connection; inner loop waits for client to send an event JSON, calls predictor, pushes result back; keeps connection alive.

- [ ] **Step 5: Implement SQLite persistence** — on startup create table `predictions(id, ts, city, route, fare_pred, occupancy_pred, delay_pred, feed_type)`.
- [ ] **Step 6: Run tests → PASS**
- [ ] **Step 7: Commit**

```bash
git add api/ tests/test_predictor.py
git commit -m "feat: FastAPI prediction server with SQLite logging"
```

---

## Task 6: Real-Time Kafka Feed + Structured Streaming

**Files:**
- Create: `stream/kafka_producer.py`
- Create: `stream/spark_streaming_predict.py`
- Test: `tests/test_kafka_producer.py`

**Interfaces:**
- Consumes: `archive/data/splits/transit_test.csv/test.csv`; Kafka topic `transit_live`
- Produces: Kafka messages (JSON, snake_case keys, with injected `event_ts` UTC); Structured Streaming job reads topic and calls prediction API via HTTP for each micro-batch; results written to `api/predictions.db` with `feed_type="real"`.

---

- [ ] **Step 1: Write failing tests**

```python
def test_producer_emits_valid_json(mock_kafka_producer, one_row_df):
    emit_row(mock_kafka_producer, one_row_df.iloc[0], topic="transit_live")
    call_args = mock_kafka_producer.send.call_args
    payload = json.loads(call_args[0][1])
    assert "fare_amount" in payload
    assert "event_ts" in payload

def test_streaming_schema_matches_producer_schema():
    from stream.spark_streaming_predict import STREAMING_SCHEMA
    assert "fare_amount" in [f.name for f in STREAMING_SCHEMA.fields]
    assert "event_ts" in [f.name for f in STREAMING_SCHEMA.fields]
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `kafka_producer.py`**

  Reads test.csv with pandas; iterates rows at ~10/sec (`time.sleep(0.1)`); serialises each row to JSON with `event_ts=datetime.utcnow().isoformat()`; sends to `kafka-python` KafkaProducer.

- [ ] **Step 4: Implement `spark_streaming_predict.py`**

  `STREAMING_SCHEMA` = `StructType` matching all 88 columns + `event_ts StringType`. Reads Kafka topic with `spark.readStream.format("kafka")`, parses JSON value, calls prediction API batch endpoint per micro-batch using `foreachBatch`. Writes results to SQLite with `feed_type="real"`.

- [ ] **Step 5: Run tests → PASS**
- [ ] **Step 6: Commit**

```bash
git add stream/kafka_producer.py stream/spark_streaming_predict.py tests/test_kafka_producer.py
git commit -m "feat: Kafka producer and Structured Streaming prediction consumer"
```

---

## Task 7: Synthetic Data Generator

**Files:**
- Create: `stream/synthetic_generator.py`
- Test: `tests/test_synthetic_generator.py`

**Interfaces:**
- Consumes: `archive/data/features/transit_features.csv/transit_fe.csv` for distribution parameters.
- Produces: Continuous stream of synthetic event dicts posted to `POST /predict`; written to `api/predictions.db` with `feed_type="synthetic"`.

---

- [ ] **Step 1: Write failing tests**

```python
def test_synthetic_stats_within_3sigma_of_training(training_stats, n=1000):
    gen = SyntheticGenerator(stats=training_stats)
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
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `SyntheticGenerator` class**

  Constructor loads training stats JSON (pre-computed from feature CSV: per-column mean/std/min/max). `generate_event()` samples from Normal distributions seeded per-city; adds Faker-generated `route_id`, random `city` choice, random `hour` weighted by commute pattern. Injects anomaly with p=0.05.

- [ ] **Step 4: Implement main loop** — runs at 10 events/sec, posts to `http://localhost:8000/predict`, logs response.
- [ ] **Step 5: Run tests → PASS**
- [ ] **Step 6: Commit**

```bash
git add stream/synthetic_generator.py tests/test_synthetic_generator.py
git commit -m "feat: probabilistic synthetic data generator with distribution matching"
```

---

## Task 8: Analytics Dashboard

**Files:**
- Create: `dashboard/data_loader.py`
- Create: `dashboard/charts.py`
- Create: `dashboard/app.py`
- Test: `tests/test_dashboard_data_loader.py`

**Interfaces:**
- Consumes: `api/predictions.db` (SQLite); live prediction data via `data_loader.load_predictions(n=500)`.
- Produces: Streamlit app at `http://localhost:8501` with 8 live-updating Plotly charts.

---

- [ ] **Step 1: Write failing tests**

```python
def test_load_predictions_returns_dataframe(tmp_db):
    # tmp_db is a pre-seeded SQLite with 10 rows
    df = load_predictions(db_path=str(tmp_db), n=500)
    assert isinstance(df, pd.DataFrame)
    assert "fare_pred" in df.columns
    assert "feed_type" in df.columns

def test_load_predictions_empty_db_returns_empty_df(empty_db):
    df = load_predictions(db_path=str(empty_db), n=500)
    assert len(df) == 0  # must not crash

def test_build_real_vs_predicted_chart_returns_figure(sample_pred_df):
    fig = build_real_vs_predicted_chart(sample_pred_df, target="fare_pred")
    assert fig is not None
```

- [ ] **Step 2: Run tests → FAIL**
- [ ] **Step 3: Implement `load_predictions(db_path, n)` in `dashboard/data_loader.py`**

  `sqlite3.connect(db_path)` → `pd.read_sql("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", con, params=(n,))`. Returns empty DataFrame with correct columns on empty table.

- [ ] **Step 4: Implement chart functions in `dashboard/charts.py`**

  | Function | Chart | Library call |
  |---|---|---|
  | `build_real_vs_predicted_chart(df, target)` | Line chart | `px.line` |
  | `build_real_vs_synthetic_error_chart(df)` | Side-by-side bar (MAE by feed_type) | `px.bar` |
  | `build_feature_importance_chart(importances)` | Horizontal bar | `px.bar(..., orientation='h')` |
  | `build_city_heatmap(df)` | Heatmap occupancy by hour/city | `px.density_heatmap` |
  | `build_rolling_rmse_gauge(df, target)` | Gauge | `go.Indicator` |
  | `build_anomaly_timeline(df)` | Scatter | `px.scatter` |
  | `build_delay_violin(df)` | Violin | `px.violin` |
  | `build_fare_boxplot(df)` | Box | `px.box` |

- [ ] **Step 5: Implement `dashboard/app.py`**

  ```python
  st.set_page_config(layout="wide", page_title="Transit Regressor Analytics")
  # sidebar: db_path, refresh_interval, city filter, feed_type filter
  df = load_predictions(...)
  # 2-column layout for charts
  col1, col2 = st.columns(2)
  with col1: st.plotly_chart(build_real_vs_predicted_chart(df, "fare_pred"))
  # ... all 8 charts
  time.sleep(refresh_interval)
  st.rerun()
  ```

- [ ] **Step 6: Run tests → PASS**
- [ ] **Step 7: Manual smoke test** — start server, generate 50 synthetic events, open dashboard, verify charts render.
- [ ] **Step 8: Commit**

```bash
git add dashboard/ tests/test_dashboard_data_loader.py
git commit -m "feat: live Streamlit analytics dashboard with 8 Plotly charts"
```

---

## Task 9: Integration Test & Documentation

**Files:**
- Create: `tests/test_integration.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: All components from Tasks 1–8.
- Produces: End-to-end test that ingests → engineers → trains (mini dataset) → serves → generates → checks DB has rows; updated README with architecture diagram and full quick-start guide.

---

- [ ] **Step 1: Write integration test**

```python
def test_end_to_end_pipeline(tmp_path, spark):
    # 1. Ingest 100-row sample → tmp_path/raw/
    ingest.run(src=SAMPLE_CSV, dest=str(tmp_path/"raw"), split="train")
    # 2. Feature engineer
    fe.run(src=str(tmp_path/"raw/train.parquet"),
           dest=str(tmp_path/"features"), split="train")
    # 3. Train (maxIter=2 for speed)
    train.train_all(spark.read.parquet(str(tmp_path/"features/train.parquet")),
                    model_dir=str(tmp_path/"models"), max_iter=2)
    # 4. Load predictor
    pred = Predictor(str(tmp_path/"models"))
    result = pred.predict(SAMPLE_EVENT)
    assert "fare" in result
    # 5. Predictions DB gets row
    db = str(tmp_path/"predictions.db")
    insert_prediction(result, db_path=db)
    df = load_predictions(db_path=db, n=10)
    assert len(df) == 1
```

- [ ] **Step 2: Run → FAIL**
- [ ] **Step 3: Fix any wiring issues found**
- [ ] **Step 4: Run → PASS**
- [ ] **Step 5: Update README** with:
  - Architecture ASCII diagram (from spec)
  - Prerequisites section (Java 11+, Hadoop 3.3, Kafka 3.x, Python 3.11)
  - `make setup && make ingest && make train && make serve` quick-start
  - Screenshot placeholder for dashboard
- [ ] **Step 6: Final commit**

```bash
git add tests/test_integration.py README.md
git commit -m "feat: integration test and full documentation"
```

---

## Execution Handoff

Plan saved. Please review and choose an execution approach:

- **Subagent-driven** — Fresh subagent per task with reviewer gate. Most thorough; best for this plan given the 9 independent-ish tasks and the risk of infrastructure misconfiguration rippling downstream.
- **Native** — I implement every task in this session, then a final review pass. Faster; good if you want to iterate quickly and have HDFS/Kafka already installed.

**I recommend Subagent-driven**, because Tasks 1–3 set up infrastructure and data contracts that every later task depends on, and catching a schema mismatch early (especially Kafka ↔ Structured Streaming) is worth the independent reviewer gate.

Does the plan capture what you want, and which approach should we use?

