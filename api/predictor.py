import math
import os
import sys
from datetime import datetime, timezone

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
from pyspark.sql.types import DoubleType, StructField, StructType
from spark.ingest import get_or_create_spark


class Predictor:
    """Predictor service wrapping trained Spark MLlib PipelineModels."""

    def __init__(self, model_dir: str = None):
        self.spark = get_or_create_spark("transit_predictor")

        if model_dir is None:
            model_dir = os.environ.get("MODEL_DIR", "./hdfs_local/transit/models/")

        self.model_dir = model_dir.rstrip("/\\")

        fare_path = f"{self.model_dir}/fare_model"
        occ_path = f"{self.model_dir}/occupancy_model"
        delay_path = f"{self.model_dir}/delay_model"

        self.fare_model = PipelineModel.load(fare_path)
        self.occupancy_model = PipelineModel.load(occ_path)
        self.delay_model = PipelineModel.load(delay_path)

        # Retrieve feature columns from VectorAssembler in the first stage
        self.input_cols = self.fare_model.stages[0].getInputCols()
        self.schema = StructType([
            StructField(col_name, DoubleType(), True) for col_name in self.input_cols
        ])

    def predict(self, event: dict) -> dict:
        """Runs predictions on single transit event dict.

        Args:
            event: Dictionary of event features

        Returns:
            dict: {"fare": float, "occupancy": float, "delay": float, "timestamp": str}
        """
        row_values = []
        for col_name in self.input_cols:
            val = event.get(col_name)
            try:
                fval = float(val) if val is not None else 0.0
                if math.isnan(fval) or math.isinf(fval):
                    fval = 0.0
                row_values.append(fval)
            except (ValueError, TypeError):
                row_values.append(0.0)

        df = self.spark.createDataFrame([row_values], schema=self.schema)

        fare_pred = float(self.fare_model.transform(df).select("prediction").head()["prediction"])
        occ_pred = float(self.occupancy_model.transform(df).select("prediction").head()["prediction"])
        delay_pred = float(self.delay_model.transform(df).select("prediction").head()["prediction"])

        if math.isnan(fare_pred) or math.isinf(fare_pred):
            fare_pred = 0.0
        if math.isnan(occ_pred) or math.isinf(occ_pred):
            occ_pred = 0.0
        if math.isnan(delay_pred) or math.isinf(delay_pred):
            delay_pred = 0.0

        ts = str(
            event.get("timestamp")
            or event.get("event_ts")
            or datetime.now(timezone.utc).isoformat()
        )

        return {
            "fare": fare_pred,
            "occupancy": occ_pred,
            "delay": delay_pred,
            "timestamp": ts,
        }


def insert_prediction(result: dict, db_path: str = None):
    """Convenience helper to insert a prediction result dictionary into SQLite."""
    from api.server import insert_prediction as server_insert_prediction, get_db_path

    path = db_path or get_db_path()
    server_insert_prediction(
        db_path=path,
        ts=str(result.get("timestamp", result.get("ts", datetime.now(timezone.utc).isoformat()))),
        city=str(result.get("city", "Unknown")),
        route=str(result.get("route_id", result.get("route", "Unknown"))),
        fare_pred=float(result.get("fare", result.get("fare_pred", 0.0))),
        occupancy_pred=float(result.get("occupancy", result.get("occupancy_pred", 0.0))),
        delay_pred=float(result.get("delay", result.get("delay_pred", 0.0))),
        feed_type=str(result.get("feed_type", "real")),
    )


