import os
import sqlite3
from typing import Optional
import pandas as pd

DEFAULT_COLUMNS = [
    "id",
    "ts",
    "city",
    "route",
    "fare_pred",
    "occupancy_pred",
    "delay_pred",
    "feed_type",
]


def load_predictions(db_path: str = "api/predictions.db", n: int = 500) -> pd.DataFrame:
    """Loads up to n recent prediction records from the SQLite database.

    Args:
        db_path: Path to the SQLite database file
        n: Maximum number of rows to retrieve (ordered by id DESC)

    Returns:
        pd.DataFrame containing prediction rows or an empty DataFrame with standard columns
    """
    empty_df = pd.DataFrame(columns=DEFAULT_COLUMNS)

    if not os.path.exists(db_path):
        return empty_df

    try:
        conn = sqlite3.connect(db_path)
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
        if not cursor.fetchone():
            conn.close()
            return empty_df

        query = "SELECT * FROM predictions ORDER BY id DESC LIMIT ?"
        df = pd.read_sql_query(query, conn, params=(n,))
        conn.close()

        if df.empty:
            return empty_df

        # Ensure all default columns exist
        for col in DEFAULT_COLUMNS:
            if col not in df.columns:
                df[col] = None

        return df
    except Exception:
        return empty_df

