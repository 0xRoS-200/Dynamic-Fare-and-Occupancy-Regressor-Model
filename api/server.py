import os
import sqlite3
import sys
from contextlib import asynccontextmanager
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from api.predictor import Predictor


DB_PATH = os.environ.get(
    "PREDICTIONS_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions.db"),
)


def get_db_path() -> str:
    return DB_PATH


def init_db(db_path: str = None):
    """Initializes SQLite database and predictions table."""
    path = db_path or get_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            city TEXT,
            route TEXT,
            fare_pred REAL,
            occupancy_pred REAL,
            delay_pred REAL,
            feed_type TEXT
        )
    """)
    conn.commit()
    conn.close()


# Ensure default database schema exists on import
init_db()


def insert_prediction(
    db_path: str,
    ts: str,
    city: str,
    route: str,
    fare_pred: float,
    occupancy_pred: float,
    delay_pred: float,
    feed_type: str = "real",
):
    """Inserts a single prediction record into the SQLite database."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO predictions (ts, city, route, fare_pred, occupancy_pred, delay_pred, feed_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (ts, city, route, fare_pred, occupancy_pred, delay_pred, feed_type),
    )
    conn.commit()
    conn.close()


# Lazy singleton Predictor
_predictor: Optional[Predictor] = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure SQLite DB table exists
    init_db()
    yield


app = FastAPI(
    title="Transit Dynamic Fare & Occupancy Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint confirming service status and models."""
    try:
        predictor = get_predictor()
        models_loaded = predictor is not None
    except Exception:
        models_loaded = False

    return {
        "status": "ok",
        "models_loaded": models_loaded,
    }


@app.post("/predict")
def predict(event: dict):
    """Predicts fare, occupancy, and delay for a single transit event and logs to SQLite."""
    predictor = get_predictor()
    result = predictor.predict(event)

    city = event.get("city", "Unknown")
    route = event.get("route_id", event.get("route", "Unknown"))
    feed_type = event.get("feed_type", "real")

    insert_prediction(
        db_path=get_db_path(),
        ts=result["timestamp"],
        city=city,
        route=route,
        fare_pred=result["fare"],
        occupancy_pred=result["occupancy"],
        delay_pred=result["delay"],
        feed_type=feed_type,
    )

    return result


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket stream accepting real-time events, running predictions, and returning JSON."""
    await websocket.accept()
    predictor = get_predictor()
    db_path = get_db_path()

    try:
        while True:
            event = await websocket.receive_json()
            result = predictor.predict(event)

            city = event.get("city", "Unknown")
            route = event.get("route_id", event.get("route", "Unknown"))
            feed_type = event.get("feed_type", "stream")

            insert_prediction(
                db_path=db_path,
                ts=result["timestamp"],
                city=city,
                route=route,
                fare_pred=result["fare"],
                occupancy_pred=result["occupancy"],
                delay_pred=result["delay"],
                feed_type=feed_type,
            )

            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn

    init_db()
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)

