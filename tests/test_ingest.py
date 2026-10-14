import pytest
import os
from pyspark.sql import SparkSession
from spark.ingest import run, get_or_create_spark

@pytest.fixture(scope="module")
def spark():
    return get_or_create_spark("test_ingest")

SAMPLE_CSV = "archive/data/splits/transit_train.csv/train.csv"

def test_ingest_produces_parquet_with_expected_schema(spark, tmp_path):
    result = run(src=SAMPLE_CSV, dest=str(tmp_path), split="train")
    df = spark.read.parquet(str(tmp_path / "train.parquet"))
    assert "fare_amount" in df.columns
    assert "occupancy_rate" in df.columns
    assert "delay_minutes" in df.columns
    assert df.count() > 0
    assert result["split"] == "train"
    assert result["rows"] > 0

def test_ingest_drops_null_target_rows(spark, tmp_path):
    result = run(src=SAMPLE_CSV, dest=str(tmp_path / "null_test"), split="train")
    df = spark.read.parquet(str(tmp_path / "null_test" / "train.parquet"))
    null_count = df.filter(
        df.fare_amount.isNull() | df.occupancy_rate.isNull() | df.delay_minutes.isNull()
    ).count()
    assert null_count == 0

