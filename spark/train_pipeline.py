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
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.sql import DataFrame
from pyspark.sql.types import NumericType
from spark.ingest import get_or_create_spark


TARGET_CONFIG = {
    "fare_amount": {
        "model_name": "fare_model",
        "regressor_class": GBTRegressor,
        "default_depth": 5,
    },
    "occupancy_rate": {
        "model_name": "occupancy_model",
        "regressor_class": GBTRegressor,
        "default_depth": 5,
    },
    "delay_minutes": {
        "model_name": "delay_model",
        "regressor_class": RandomForestRegressor,
        "default_depth": 5,
    },
}


def get_feature_columns(df: DataFrame, targets: list) -> list:
    """Extracts all numeric column names from df, excluding target columns."""
    return [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, NumericType) and field.name not in targets
    ]


def train_all(
    df: DataFrame,
    model_dir: str,
    val_df: DataFrame = None,
    max_iter: int = 50,
    log_mlflow: bool = True,
    experiment_name: str = "transit_regressor",
) -> dict:
    """Trains regression models for fare_amount, occupancy_rate, and delay_minutes.

    For each target:
    - Assembles features with VectorAssembler (all numeric columns minus targets)
    - GBTRegressor for fare and occupancy; RandomForestRegressor for delay
    - Wraps in Pipeline([assembler, regressor])
    - Fits on df and saves to model_dir/{fare,occupancy,delay}_model
    - Logs params and validation metrics (RMSE, MAE, R2) to MLflow

    Returns:
        dict: {"models": dict, "metrics": dict, "mlflow_run_ids": dict}
    """
    targets = list(TARGET_CONFIG.keys())
    feature_cols = get_feature_columns(df, targets)

    if not feature_cols:
        raise ValueError("No numeric feature columns found in dataset")

    # Clean missing values in feature columns
    train_clean = df.fillna(0.0, subset=feature_cols)
    val_clean = val_df.fillna(0.0, subset=feature_cols) if val_df is not None else train_clean

    os.makedirs(model_dir, exist_ok=True)

    saved_models = {}
    all_metrics = {}
    mlflow_run_ids = {}

    if log_mlflow:
        try:
            import mlflow

            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"Warning: MLflow setup failed ({e}), continuing without MLflow logging.")
            log_mlflow = False

    for target, config in TARGET_CONFIG.items():
        if target not in df.columns:
            print(f"Skipping {target}: target not found in DataFrame.")
            continue

        model_name = config["model_name"]
        reg_class = config["regressor_class"]
        max_depth = config["default_depth"]

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="keep",
        )

        reg_kwargs = {
            "featuresCol": "features",
            "labelCol": target,
            "maxDepth": max_depth,
            "seed": 42,
        }
        if reg_class == RandomForestRegressor:
            reg_kwargs["numTrees"] = max_iter
        else:
            reg_kwargs["maxIter"] = max_iter

        regressor = reg_class(**reg_kwargs)

        pipeline = Pipeline(stages=[assembler, regressor])
        fitted_model = pipeline.fit(train_clean)

        out_path = os.path.join(model_dir, model_name)
        fitted_model.write().overwrite().save(out_path)
        saved_models[target] = out_path

        # Compute validation metrics
        pred_df = fitted_model.transform(val_clean)

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

        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        all_metrics[target] = metrics

        # Log to MLflow
        if log_mlflow:
            import mlflow

            with mlflow.start_run(run_name=f"train_{model_name}") as run:
                mlflow.log_param("target", target)
                if reg_class == RandomForestRegressor:
                    mlflow.log_param("numTrees", max_iter)
                else:
                    mlflow.log_param("maxIter", max_iter)
                mlflow.log_param("maxDepth", max_depth)
                mlflow.log_param("num_features", len(feature_cols))
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                mlflow_run_ids[target] = run.info.run_id

    return {
        "models": saved_models,
        "metrics": all_metrics,
        "mlflow_run_ids": mlflow_run_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Spark MLlib Model Training Pipeline")
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to feature-engineered training Parquet",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Path to feature-engineered validation Parquet",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Maximum iterations for GBT and RF regressors",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        default=False,
        help="Disable MLflow metric logging",
    )
    parser.add_argument(
        "--local-fallback",
        action="store_true",
        default=False,
        help="Use local directory fallback instead of HDFS",
    )

    args = parser.parse_args()

    spark = get_or_create_spark("train_pipeline")

    if args.train_path:
        train_path = args.train_path
    elif args.local_fallback:
        train_path = "./hdfs_local/transit/features/train.parquet"
    else:
        train_path = "hdfs://localhost:9000/transit/features/train.parquet"

    if args.val_path:
        val_path = args.val_path
    elif args.local_fallback:
        val_path = "./hdfs_local/transit/features/val.parquet"
    else:
        val_path = "hdfs://localhost:9000/transit/features/val.parquet"

    if args.model_dir:
        model_dir = args.model_dir
    elif args.local_fallback:
        model_dir = "./hdfs_local/transit/models/"
    else:
        model_dir = "hdfs://localhost:9000/transit/models/"

    train_df = spark.read.parquet(train_path)
    val_df = spark.read.parquet(val_path) if os.path.exists(val_path) or val_path.startswith("hdfs://") else None

    result = train_all(
        df=train_df,
        model_dir=model_dir,
        val_df=val_df,
        max_iter=args.max_iter,
        log_mlflow=not args.no_mlflow,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

