import os
import pandas as pd
import pytest

from spark import ingest
from spark import feature_engineering as fe
from spark import train_pipeline as train
from api.predictor import Predictor, insert_prediction
from dashboard.data_loader import load_predictions
from spark.ingest import get_or_create_spark

SAMPLE_EVENT = {
    "city": "London",
    "route_id": "route_1",
    "boarding_count": 12.0,
    "alighting_count": 8.0,
    "passenger_count": 20.0,
    "route_length_km": 15.0,
    "headway_min": 8.0,
    "temperature_c": 18.0,
    "vehicle_capacity": 60.0,
    "hour_sin": 0.5,
    "hour_cos": 0.86,
    "dow_sin": 0.0,
    "month_sin": 0.5,
    "feed_type": "real",
}


@pytest.fixture(scope="module")
def spark():
    return get_or_create_spark("test_integration")


def test_end_to_end_pipeline(tmp_path, spark):
    """End-to-end integration test validating the entire pipeline:

    Ingest -> Feature Engineering -> Model Training -> Serving/Prediction -> Database Logging -> Dashboard Loading
    """
    # 1. Ingest 100-row sample -> tmp_path/raw/
    sample_csv = tmp_path / "sample_train.csv"
    src_csv = "archive/data/splits/transit_train.csv/train.csv"
    if not os.path.exists(src_csv):
        src_csv = "archive/data/splits/transit_test.csv/test.csv"

    df_sample = pd.read_csv(src_csv, nrows=100)
    df_sample.to_csv(sample_csv, index=False)

    raw_dest = str(tmp_path / "raw")
    ingest.run(src=str(sample_csv), dest=raw_dest, split="train")
    raw_parquet = os.path.join(raw_dest, "train.parquet")
    assert os.path.exists(raw_parquet)

    # 2. Feature engineer
    feat_dest = str(tmp_path / "features")
    fe.run(src=raw_parquet, dest=feat_dest, split="train")
    feat_parquet = os.path.join(feat_dest, "train.parquet")
    assert os.path.exists(feat_parquet)

    # 3. Train models (max_iter=2 for speed, log_mlflow=False)
    features_df = spark.read.parquet(feat_parquet)
    models_dir = str(tmp_path / "models")
    train.train_all(
        df=features_df,
        model_dir=models_dir,
        max_iter=2,
        log_mlflow=False,
    )
    assert os.path.exists(os.path.join(models_dir, "fare_model"))
    assert os.path.exists(os.path.join(models_dir, "occupancy_model"))
    assert os.path.exists(os.path.join(models_dir, "delay_model"))

    # 4. Load predictor & run inference
    pred = Predictor(model_dir=models_dir)
    result = pred.predict(SAMPLE_EVENT)
    assert "fare" in result
    assert "occupancy" in result
    assert "delay" in result
    assert isinstance(result["fare"], float)
    assert isinstance(result["occupancy"], float)
    assert isinstance(result["delay"], float)

    # 5. Persist prediction to SQLite & verify dashboard loader
    db = str(tmp_path / "predictions.db")
    insert_prediction(result, db_path=db)
    df_loaded = load_predictions(db_path=db, n=10)

    assert len(df_loaded) == 1
    assert "fare_pred" in df_loaded.columns
    assert "occupancy_pred" in df_loaded.columns
    assert "delay_pred" in df_loaded.columns
    assert df_loaded.iloc[0]["fare_pred"] == pytest.approx(result["fare"])

