import os
import pytest
from pyspark.sql.types import (
    DoubleType,
    StringType,
    StructField,
    StructType,
)
from spark.ingest import get_or_create_spark
from spark.train_pipeline import train_all
from spark.batch_evaluate import evaluate_model, evaluate_all_models


@pytest.fixture(scope="module")
def spark():
    return get_or_create_spark("test_train_pipeline")


@pytest.fixture
def small_feature_df(spark):
    schema = StructType([
        StructField("trip_id", StringType(), False),
        StructField("route_length_km", DoubleType(), False),
        StructField("boarding_count", DoubleType(), False),
        StructField("hour_sin", DoubleType(), False),
        StructField("hour_cos", DoubleType(), False),
        StructField("dow_sin", DoubleType(), False),
        StructField("month_sin", DoubleType(), False),
        StructField("rolling_mean_1h_boarding", DoubleType(), False),
        StructField("rolling_std_1h_boarding", DoubleType(), False),
        StructField("city_norm_boarding", DoubleType(), False),
        StructField("fare_amount", DoubleType(), False),
        StructField("occupancy_rate", DoubleType(), False),
        StructField("delay_minutes", DoubleType(), False),
    ])

    # 15 synthetic rows with predictable target relationships
    data = []
    for i in range(15):
        dist = 2.0 + i * 0.5
        board = 10.0 + i * 2.0
        h_sin = 0.5
        h_cos = 0.8
        d_sin = 0.1
        m_sin = 0.3
        rmean = board - 1.0
        rstd = 1.5
        cnorm = (board - 20.0) / 10.0
        # Target relationships matching our domain:
        fare = 2.50 + 0.15 * dist + 0.50 * 0.4
        occ = min(1.0, 0.2 + 0.03 * i)
        delay = float(i % 5)
        data.append((
            f"trip_{i}",
            dist,
            board,
            h_sin,
            h_cos,
            d_sin,
            m_sin,
            rmean,
            rstd,
            cnorm,
            fare,
            occ,
            delay,
        ))

    return spark.createDataFrame(data, schema)


def test_models_are_saved(tmp_path, spark, small_feature_df):
    model_dir = str(tmp_path / "models")
    train_all(small_feature_df, model_dir=model_dir, max_iter=2, log_mlflow=False)
    assert (tmp_path / "models" / "fare_model").exists()
    assert (tmp_path / "models" / "occupancy_model").exists()
    assert (tmp_path / "models" / "delay_model").exists()


def test_fare_model_rmse_below_threshold(tmp_path, spark, small_feature_df):
    model_dir = str(tmp_path / "models")
    train_all(small_feature_df, model_dir=model_dir, max_iter=5, log_mlflow=False)
    fare_model_path = os.path.join(model_dir, "fare_model")
    metrics = evaluate_model(
        spark=spark,
        model_path=fare_model_path,
        target="fare_amount",
        test_data=small_feature_df,
    )
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert metrics["rmse"] < 5.0


def test_evaluate_all_models_report(tmp_path, spark, small_feature_df):
    model_dir = str(tmp_path / "models")
    train_all(small_feature_df, model_dir=model_dir, max_iter=2, log_mlflow=False)
    report_path = str(tmp_path / "evaluation_report.json")
    report = evaluate_all_models(
        spark=spark,
        model_dir=model_dir,
        test_data=small_feature_df,
        report_path=report_path,
    )
    assert os.path.exists(report_path)
    assert "fare_amount" in report
    assert "occupancy_rate" in report
    assert "delay_minutes" in report
    assert report["fare_amount"]["rmse"] < 5.0

