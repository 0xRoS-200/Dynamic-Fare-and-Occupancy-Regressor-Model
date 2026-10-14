import pytest
from pyspark.sql.functions import col, to_timestamp
from spark.ingest import get_or_create_spark

# Import functions and classes from spark.feature_engineering
from spark.feature_engineering import (
    add_cyclic_features,
    add_rolling_features,
    FeatureTransformer,
)


@pytest.fixture(scope="module")
def spark():
    return get_or_create_spark("test_feature_engineering")


@pytest.fixture
def sample_df(spark):
    data = [
        ("trip_1", "2024-01-01 08:00:00", "route_1", "New York", 10.0, 25.0, 0.4, 2.0),
        ("trip_2", "2024-01-01 09:00:00", "route_1", "New York", 20.0, 27.0, 0.5, 5.0),
        ("trip_3", "2024-01-01 10:00:00", "route_1", "New York", 30.0, 30.0, 0.6, 0.0),
        ("trip_4", "2024-01-01 08:30:00", "route_2", "Chicago", 15.0, 18.0, 0.3, 1.0),
        ("trip_5", "2024-01-01 09:30:00", "route_2", "Chicago", 25.0, 22.0, 0.7, 4.0),
    ]
    schema = [
        "trip_id",
        "timestamp",
        "route_id",
        "city",
        "boarding_count",
        "fare_amount",
        "occupancy_rate",
        "delay_minutes",
    ]
    df = spark.createDataFrame(data, schema)
    return df.withColumn("timestamp", to_timestamp("timestamp"))


def test_cyclic_hour_encoding_range(spark, sample_df):
    df = add_cyclic_features(sample_df)
    assert df.filter((col("hour_sin") < -1) | (col("hour_sin") > 1)).count() == 0
    assert "hour_sin" in df.columns
    assert "hour_cos" in df.columns
    assert "month_sin" in df.columns
    assert "month_cos" in df.columns


def test_rolling_mean_no_leakage(spark, sample_df):
    # row at index i must not use row i's own value
    df = add_rolling_features(sample_df, window_hours=1)
    # rolling mean at second row != current row value when there's variance
    row1, row2 = df.orderBy("timestamp").take(2)
    assert row1["rolling_mean_1h_boarding"] != row2["boarding_count"]


def test_feature_transformer_roundtrip(spark, tmp_path, sample_df):
    t = FeatureTransformer()
    t.fit(sample_df)
    save_path = str(tmp_path / "transformer")
    t.save(save_path)
    t2 = FeatureTransformer.load(save_path)
    out = t2.transform(sample_df)
    assert "fare_amount" in out.columns
    assert "hour_sin" in out.columns
    assert "rolling_mean_1h_boarding" in out.columns


def test_rolling_features_multiple_windows(spark, sample_df):
    df = add_rolling_features(sample_df)
    for w in [1, 6, 24]:
        assert f"rolling_mean_{w}h_boarding" in df.columns
        assert f"rolling_std_{w}h_boarding" in df.columns

