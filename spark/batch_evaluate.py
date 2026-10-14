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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame
from spark.ingest import get_or_create_spark


TARGET_MODEL_MAP = {
    "fare_amount": "fare_model",
    "occupancy_rate": "occupancy_model",
    "delay_minutes": "delay_model",
}


def evaluate_model(
    spark,
    model_path: str,
    target: str,
    test_data,
) -> dict:
    """Evaluates a saved Spark ML PipelineModel on test data for the given target.

    Args:
        spark: Active SparkSession
        model_path: Path to saved PipelineModel
        target: Target label column name ('fare_amount', 'occupancy_rate', 'delay_minutes')
        test_data: DataFrame or string path to test Parquet dataset

    Returns:
        dict: {"rmse": float, "mae": float, "r2": float}
    """
    if isinstance(test_data, str):
        df = spark.read.parquet(test_data)
    else:
        df = test_data

    model = PipelineModel.load(model_path)
    pred_df = model.transform(df)

    evaluator_rmse = RegressionEvaluator(
        labelCol=target, predictionCol="prediction", metricName="rmse"
    )
    evaluator_mae = RegressionEvaluator(
        labelCol=target, predictionCol="prediction", metricName="mae"
    )
    evaluator_r2 = RegressionEvaluator(
        labelCol=target, predictionCol="prediction", metricName="r2"
    )

    rmse = float(evaluator_rmse.evaluate(pred_df))
    mae = float(evaluator_mae.evaluate(pred_df))
    r2 = float(evaluator_r2.evaluate(pred_df))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }


def evaluate_all_models(
    spark,
    model_dir: str,
    test_data,
    report_path: str = None,
) -> dict:
    """Evaluates all three saved regression models on test data and writes report.

    Args:
        spark: Active SparkSession
        model_dir: Directory containing {fare,occupancy,delay}_model/
        test_data: DataFrame or string path to test Parquet dataset
        report_path: Optional file path to write evaluation_report.json

    Returns:
        dict: Target-keyed dictionary of evaluation metrics
    """
    report = {}
    for target, model_name in TARGET_MODEL_MAP.items():
        m_path = os.path.join(model_dir, model_name)
        if os.path.exists(m_path) or m_path.startswith("hdfs://"):
            metrics = evaluate_model(
                spark=spark,
                model_path=m_path,
                target=target,
                test_data=test_data,
            )
            report[target] = metrics

    if report_path:
        os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def main():
    parser = argparse.ArgumentParser(description="Batch Model Evaluation")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory containing saved ML models",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to feature-engineered test dataset",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="evaluation_report.json",
        help="Path to output evaluation report JSON",
    )
    parser.add_argument(
        "--local-fallback",
        action="store_true",
        default=False,
        help="Use local directory fallback instead of HDFS",
    )

    args = parser.parse_args()

    spark = get_or_create_spark("batch_evaluate")

    if args.model_dir:
        model_dir = args.model_dir
    elif args.local_fallback:
        model_dir = "./hdfs_local/transit/models/"
    else:
        model_dir = "hdfs://localhost:9000/transit/models/"

    if args.test_path:
        test_path = args.test_path
    elif args.local_fallback:
        test_path = "./hdfs_local/transit/features/test.parquet"
    else:
        test_path = "hdfs://localhost:9000/transit/features/test.parquet"

    report = evaluate_all_models(
        spark=spark,
        model_dir=model_dir,
        test_data=test_path,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

