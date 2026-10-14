import argparse
import json
import math
import os
import sys

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

from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window
from spark.ingest import get_or_create_spark


def add_cyclic_features(df: DataFrame) -> DataFrame:
    """Computes cyclic trigonometric features for hour, day of week, and month.

    Uses:
    - hour_sin = sin(hour * 2pi / 24), hour_cos = cos(hour * 2pi / 24)
    - day_of_week_sin = sin(day_of_week * 2pi / 7), day_of_week_cos = cos(day_of_week * 2pi / 7)
    - month_sin = sin(month * 2pi / 12), month_cos = cos(month * 2pi / 12)
    """
    pi = math.pi
    res = df

    # Derive hour, day_of_week, month from timestamp if missing
    if "timestamp" in res.columns:
        if "hour" not in res.columns:
            res = res.withColumn("hour", F.hour(F.col("timestamp")).cast(DoubleType()))
        if "day_of_week" not in res.columns:
            res = res.withColumn("day_of_week", F.dayofweek(F.col("timestamp")).cast(DoubleType()))
        if "month" not in res.columns:
            res = res.withColumn("month", F.month(F.col("timestamp")).cast(DoubleType()))

    # Hour cyclic encoding
    if "hour" in res.columns:
        hour_rad = F.col("hour").cast(DoubleType()) * (2.0 * pi / 24.0)
        res = res.withColumn("hour_sin", F.sin(hour_rad).cast(DoubleType()))
        res = res.withColumn("hour_cos", F.cos(hour_rad).cast(DoubleType()))

    # Day of week cyclic encoding
    if "day_of_week" in res.columns:
        dow_rad = F.col("day_of_week").cast(DoubleType()) * (2.0 * pi / 7.0)
        dow_sin_col = F.sin(dow_rad).cast(DoubleType())
        dow_cos_col = F.cos(dow_rad).cast(DoubleType())
        res = res.withColumn("day_of_week_sin", dow_sin_col)
        res = res.withColumn("day_of_week_cos", dow_cos_col)
        res = res.withColumn("dow_sin", dow_sin_col)
        res = res.withColumn("dow_cos", dow_cos_col)

    # Month cyclic encoding
    if "month" in res.columns:
        month_rad = F.col("month").cast(DoubleType()) * (2.0 * pi / 12.0)
        res = res.withColumn("month_sin", F.sin(month_rad).cast(DoubleType()))
        res = res.withColumn("month_cos", F.cos(month_rad).cast(DoubleType()))

    return res


def add_rolling_features(df: DataFrame, window_hours=None) -> DataFrame:
    """Computes rolling mean and standard deviation for boarding count without data leakage.

    Uses Window.partitionBy("route_id").orderBy("timestamp").rowsBetween(-(window_hours * 60), -1).
    Since rowsBetween ends at -1, the current row's own value is strictly excluded.
    """
    if window_hours is None:
        windows = [1, 6, 24]
    elif isinstance(window_hours, (int, float)):
        windows = [int(window_hours)]
    else:
        windows = list(window_hours)

    res = df
    partition_cols = ["route_id"] if "route_id" in res.columns else []
    order_col = F.col("timestamp") if "timestamp" in res.columns else F.lit(1)

    for w in windows:
        window_size = int(w * 60)
        if partition_cols:
            w_spec = (
                Window.partitionBy(*partition_cols)
                .orderBy(order_col)
                .rowsBetween(-window_size, -1)
            )
        else:
            w_spec = Window.orderBy(order_col).rowsBetween(-window_size, -1)

        mean_name = f"rolling_mean_{w}h_boarding"
        std_name = f"rolling_std_{w}h_boarding"

        res = res.withColumn(
            mean_name, F.avg(F.col("boarding_count")).over(w_spec).cast(DoubleType())
        )
        res = res.withColumn(
            std_name, F.stddev(F.col("boarding_count")).over(w_spec).cast(DoubleType())
        )

    return res


class FeatureTransformer:
    """Feature transformer for MapReduce Reduce Phase.

    Fits per-city normalization stats, saves/loads transformer artifacts and Spark Pipeline,
    and applies cyclic, rolling, and normalization transforms.
    """

    def __init__(self, city_stats=None, global_stats=None, pipeline_model=None):
        self.city_stats = city_stats or {}
        self.global_stats = global_stats or {"mean": 0.0, "std": 1.0}
        self.pipeline_model = pipeline_model

    def fit(self, df: DataFrame):
        """Computes per-city mean/std for normalization and fits an MLlib Pipeline."""
        if "city" in df.columns and "boarding_count" in df.columns:
            stats_rows = (
                df.groupBy("city")
                .agg(
                    F.avg("boarding_count").alias("mean_board"),
                    F.stddev("boarding_count").alias("std_board"),
                )
                .collect()
            )
            self.city_stats = {}
            for r in stats_rows:
                mean_val = float(r["mean_board"]) if r["mean_board"] is not None else 0.0
                std_val = float(r["std_board"]) if r["std_board"] is not None and r["std_board"] > 0 else 1.0
                self.city_stats[r["city"]] = {"mean": mean_val, "std": std_val}

            glob_row = (
                df.agg(
                    F.avg("boarding_count").alias("mean_board"),
                    F.stddev("boarding_count").alias("std_board"),
                )
                .collect()[0]
            )
            glob_mean = float(glob_row["mean_board"]) if glob_row["mean_board"] is not None else 0.0
            glob_std = float(glob_row["std_board"]) if glob_row["std_board"] is not None and glob_row["std_board"] > 0 else 1.0
            self.global_stats = {"mean": glob_mean, "std": glob_std}

        # Initialize and fit Spark ML Pipeline
        pipeline = Pipeline(stages=[])
        self.pipeline_model = pipeline.fit(df)
        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """Applies cyclic features, rolling window aggregates, and city normalizations."""
        res = add_cyclic_features(df)
        res = add_rolling_features(res, window_hours=[1, 6, 24])

        # Apply city-level normalization if fitted
        if self.city_stats and "city" in res.columns and "boarding_count" in res.columns:
            # Build when-otherwise expressions for mean and std
            mean_expr = None
            std_expr = None
            for city, stats in self.city_stats.items():
                if mean_expr is None:
                    mean_expr = F.when(F.col("city") == city, F.lit(stats["mean"]))
                    std_expr = F.when(F.col("city") == city, F.lit(stats["std"]))
                else:
                    mean_expr = mean_expr.when(F.col("city") == city, F.lit(stats["mean"]))
                    std_expr = std_expr.when(F.col("city") == city, F.lit(stats["std"]))

            default_mean = self.global_stats.get("mean", 0.0)
            default_std = self.global_stats.get("std", 1.0)
            city_mean_col = mean_expr.otherwise(F.lit(default_mean))
            city_std_col = std_expr.otherwise(F.lit(default_std))

            res = res.withColumn("city_mean_boarding", city_mean_col)
            res = res.withColumn("city_std_boarding", city_std_col)
            res = res.withColumn(
                "city_norm_boarding",
                ((F.col("boarding_count") - city_mean_col) / city_std_col).cast(DoubleType()),
            )

        if self.pipeline_model is not None:
            res = self.pipeline_model.transform(res)

        return res

    def save(self, path: str):
        """Writes stats as JSON + PySpark Pipeline.save."""
        os.makedirs(path, exist_ok=True)
        stats_file = os.path.join(path, "stats.json")
        stats_data = {
            "city_stats": self.city_stats,
            "global_stats": self.global_stats,
        }
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2)

        if self.pipeline_model is not None:
            pipeline_dir = os.path.join(path, "pipeline")
            self.pipeline_model.write().overwrite().save(pipeline_dir)

    @classmethod
    def load(cls, path: str):
        """Loads stats JSON and PipelineModel from path."""
        stats_file = os.path.join(path, "stats.json")
        city_stats = {}
        global_stats = {"mean": 0.0, "std": 1.0}
        if os.path.exists(stats_file):
            with open(stats_file, "r", encoding="utf-8") as f:
                stats_data = json.load(f)
                city_stats = stats_data.get("city_stats", {})
                global_stats = stats_data.get("global_stats", global_stats)

        pipeline_dir = os.path.join(path, "pipeline")
        pipeline_model = None
        if os.path.exists(pipeline_dir):
            pipeline_model = PipelineModel.load(pipeline_dir)

        return cls(
            city_stats=city_stats,
            global_stats=global_stats,
            pipeline_model=pipeline_model,
        )


def run_pipeline(
    src: str,
    dest: str,
    model_dest: str,
    split: str = "train",
    transformer: FeatureTransformer = None,
) -> dict:
    """Processes raw parquet at src/{split}.parquet, fits/loads transformer, writes features."""
    spark = get_or_create_spark("feature_engineering")

    src_clean = src.rstrip("/\\")
    dest_clean = dest.rstrip("/\\")

    if src_clean.endswith(".parquet"):
        src_path = src_clean
    else:
        src_path = f"{src_clean}/{split}.parquet"

    dest_path = f"{dest_clean}/{split}.parquet"

    df = spark.read.parquet(src_path)

    if transformer is None:
        if split == "train":
            transformer = FeatureTransformer()
            transformer.fit(df)
            transformer.save(model_dest)
        else:
            transformer = FeatureTransformer.load(model_dest)

    features_df = transformer.transform(df)
    features_df.write.mode("overwrite").parquet(dest_path)

    return {
        "split": split,
        "src": src_path,
        "dest": dest_path,
        "rows": features_df.count(),
        "columns": len(features_df.columns),
        "model_dest": model_dest,
    }


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering (MapReduce Reduce Phase)")
    parser.add_argument("--src", type=str, default=None, help="Source raw parquet directory")
    parser.add_argument("--dest", type=str, default=None, help="Destination features directory")
    parser.add_argument(
        "--model-dest",
        type=str,
        default=None,
        help="Destination for saved FeatureTransformer",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test", "full"],
        help="Specific split to process",
    )
    parser.add_argument(
        "--local-fallback",
        action="store_true",
        default=False,
        help="Use local directory fallback instead of HDFS",
    )

    args = parser.parse_args()

    if args.dest:
        dest = args.dest
    elif args.local_fallback:
        dest = "./hdfs_local/transit/features/"
    else:
        dest = "hdfs://localhost:9000/transit/features/"

    if args.src:
        src = args.src
    elif args.local_fallback:
        src = "./hdfs_local/transit/raw/"
    else:
        src = "hdfs://localhost:9000/transit/raw/"

    if args.model_dest:
        model_dest = args.model_dest
    elif args.local_fallback:
        model_dest = "./hdfs_local/transit/models/transformer/"
    else:
        model_dest = "hdfs://localhost:9000/transit/models/transformer/"

    if args.split:
        result = run_pipeline(src, dest, model_dest, split=args.split)
        print(json.dumps(result, indent=2))
    else:
        # Check available splits in src
        splits_to_run = []
        for s in ["train", "val", "test", "full"]:
            s_path = os.path.join(src, f"{s}.parquet") if not src.startswith("hdfs://") else f"{src}/{s}.parquet"
            if os.path.exists(s_path):
                splits_to_run.append(s)

        if not splits_to_run:
            splits_to_run = ["train"]

        # Ensure train runs first to fit and save transformer
        if "train" in splits_to_run:
            splits_to_run.remove("train")
            splits_to_run.insert(0, "train")

        results = []
        transformer = None
        for s in splits_to_run:
            res = run_pipeline(src, dest, model_dest, split=s, transformer=transformer)
            results.append(res)
            if s == "train":
                transformer = FeatureTransformer.load(model_dest)

        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

