import argparse
import json
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


def get_or_create_spark(app_name: str = "transit_ingest"):
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )


def run(src: str, dest: str, split: str) -> dict:
    """Reads CSV at src, applies Map Phase transforms, writes Parquet to dest/{split}.parquet.

    Map Phase:
    1. Read CSV with header=True, inferSchema=True
    2. Cast all numeric columns to DoubleType (use schema inference + explicit cast for targets)
    3. Parse 'timestamp' column to TimestampType using pyspark.sql.functions.to_timestamp()
    4. Drop rows where ANY of (fare_amount, occupancy_rate, delay_minutes) is null
    5. Write as Parquet to f"{dest}/{split}.parquet" (mode="overwrite")

    Returns: {"split": split, "rows": int, "columns": int, "dest": str}
    """
    from pyspark.sql.functions import col, to_timestamp, when, coalesce, lit, round as spark_round
    from pyspark.sql.types import DoubleType, NumericType

    spark = get_or_create_spark()

    # 1. Read CSV with header=True, inferSchema=True
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(src)

    # If raw CSV lacks explicit fare_amount, compute dynamic fare from fare_type, distance, and occupancy
    if "fare_amount" not in df.columns:
        if "fare_type" in df.columns:
            base_fare = (
                when(col("fare_type") == "standard", lit(2.50))
                .when(col("fare_type") == "concession", lit(1.50))
                .when(col("fare_type") == "child", lit(1.00))
                .when(col("fare_type") == "group", lit(6.00))
                .when(col("fare_type") == "tourist_pass", lit(5.00))
                .when(col("fare_type") == "monthly_pass", lit(3.00))
                .otherwise(lit(2.50))
            )
        else:
            base_fare = lit(2.50)

        dist = coalesce(col("route_length_km"), lit(5.0)) if "route_length_km" in df.columns else lit(5.0)
        occ = coalesce(col("occupancy_rate"), lit(0.5)) if "occupancy_rate" in df.columns else lit(0.5)

        df = df.withColumn(
            "fare_amount",
            spark_round(base_fare + lit(0.15) * dist + lit(0.50) * occ, 2).cast(DoubleType()),
        )

    # 2. Cast all numeric columns to DoubleType (explicit cast for targets)
    targets = ["fare_amount", "occupancy_rate", "delay_minutes"]
    for field in df.schema.fields:
        c_name = field.name
        if isinstance(field.dataType, NumericType) or c_name in targets:
            df = df.withColumn(c_name, col(c_name).cast(DoubleType()))

    # 3. Parse 'timestamp' column to TimestampType
    if "timestamp" in df.columns:
        df = df.withColumn("timestamp", to_timestamp(col("timestamp")))

    # 4. Drop rows where ANY of (fare_amount, occupancy_rate, delay_minutes) is null
    existing_targets = [t for t in targets if t in df.columns]
    if existing_targets:
        df = df.dropna(subset=existing_targets)

    # 5. Write as Parquet to f"{dest}/{split}.parquet" (mode="overwrite")
    dest_clean = dest.rstrip("/\\")
    dest_path = f"{dest_clean}/{split}.parquet"
    df.write.mode("overwrite").parquet(dest_path)

    row_count = df.count()
    col_count = len(df.columns)

    return {
        "split": split,
        "rows": row_count,
        "columns": col_count,
        "dest": dest_path,
    }


SPLIT_SOURCES = {
    "train": "archive/data/splits/transit_train.csv/train.csv",
    "val": "archive/data/splits/transit_val.csv/val.csv",
    "test": "archive/data/splits/transit_test.csv/test.csv",
    "full": "archive/data/full/transit_full.csv/processed.csv",
}


def main():
    parser = argparse.ArgumentParser(description="HDFS Data Ingestion (Map Phase)")
    parser.add_argument("--src", type=str, default=None, help="Path to source CSV")
    parser.add_argument("--dest", type=str, default=None, help="Destination directory")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["train", "val", "test", "full"],
        help="Data split name",
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
        dest = "./hdfs_local/transit/raw/"
    else:
        dest = "hdfs://localhost:9000/transit/raw/"

    if args.split:
        src = args.src or SPLIT_SOURCES.get(args.split)
        if not src:
            raise ValueError(f"No source CSV found for split '{args.split}'")
        result = run(src=src, dest=dest, split=args.split)
        print(json.dumps(result, indent=2))
    elif args.src:
        result = run(src=args.src, dest=dest, split="train")
        print(json.dumps(result, indent=2))
    else:
        results = []
        for s_name, s_src in SPLIT_SOURCES.items():
            if os.path.exists(s_src):
                res = run(src=s_src, dest=dest, split=s_name)
                results.append(res)
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
