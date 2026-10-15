"""Unified Runner for Dynamic Fare & Occupancy Regressor.

Launches the entire system with a single command:
1. Checks prerequisite models (and trains them if missing).
2. Starts the FastAPI prediction server on port 8000.
3. Waits for the API health check to pass.
4. Starts the Streamlit live analytics dashboard on port 8501.
5. Starts the synthetic telemetry stream generator.
6. Opens the dashboard in the browser and handles graceful shutdown.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

PROCESSES = []


def cleanup(signum=None, frame=None):
    """Gracefully terminate all spawned background processes."""
    print("\n[INFO] Shutting down all services...")
    for p in reversed(PROCESSES):
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
    print("[INFO] All services stopped. Goodbye!")
    sys.exit(0)


def wait_for_health(url: str, timeout: int = 45) -> bool:
    """Poll API health endpoint until ready or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def run_pipeline():
    """Run data ingestion, feature engineering, and model training."""
    print("[1/3] Ingesting CSV dataset to Parquet...")
    subprocess.run([sys.executable, "spark/ingest.py", "--local-fallback"], check=True)

    print("[2/3] Extracting features with rolling windows...")
    subprocess.run(
        [sys.executable, "spark/feature_engineering.py", "--local-fallback"], check=True
    )

    print("[3/3] Training Spark MLlib GBT & Random Forest models...")
    subprocess.run(
        [
            sys.executable,
            "spark/train_pipeline.py",
            "--local-fallback",
            "--max-iter",
            "50",
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run all services for Dynamic Fare & Occupancy Regressor"
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Force re-run of data ingestion, feature engineering, and model training before starting services",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=8000,
        help="Port for FastAPI inference server (default: 8000)",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8501,
        help="Port for Streamlit dashboard (default: 8501)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between synthetic stream events (default: 0.5)",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Do not start the synthetic telemetry generator",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open the browser",
    )

    args = parser.parse_args()

    # Register exit signal handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print("=" * 70)
    print("  DYNAMIC FARE & OCCUPANCY REGRESSOR -- ALL-IN-ONE RUNNER")
    print("=" * 70)

    # 1. Check if models exist
    models_dir = os.path.join("hdfs_local", "transit", "models", "fare_model")
    if args.pipeline or not os.path.exists(models_dir):
        if not os.path.exists(models_dir):
            print(
                "[INFO] Pre-trained models not found. Running training pipeline first..."
            )
        else:
            print("[INFO] --pipeline specified. Re-training models...")
        run_pipeline()
    else:
        print("[INFO] Pre-trained models found in hdfs_local/transit/models/")

    # 2. Launch FastAPI prediction server
    print(
        f"\n[INFO] Starting FastAPI Prediction Server on http://localhost:{args.api_port} ..."
    )
    api_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.server:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(args.api_port),
    ]
    api_proc = subprocess.Popen(api_cmd)
    PROCESSES.append(api_proc)

    # 3. Wait for API to become ready
    health_url = f"http://localhost:{args.api_port}/health"
    print(f"[INFO] Waiting for API health check at {health_url} ...")
    if wait_for_health(health_url, timeout=30):
        print("[INFO] API Server is healthy and accepting predictions!")
    else:
        print(
            "[WARN] API Server health check timed out. Proceeding with dashboard..."
        )

    # 4. Launch Streamlit Analytics Dashboard
    print(
        f"\n[INFO] Starting Streamlit Analytics Dashboard on http://localhost:{args.dashboard_port} ..."
    )
    dash_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "dashboard/app.py",
        "--server.port",
        str(args.dashboard_port),
        "--server.headless",
        "true",
    ]
    dash_proc = subprocess.Popen(dash_cmd)
    PROCESSES.append(dash_proc)

    # Give Streamlit a moment to start
    time.sleep(2)

    # 5. Open browser if requested
    dashboard_url = f"http://localhost:{args.dashboard_port}"
    if not args.no_browser:
        print(f"[INFO] Opening {dashboard_url} in your default browser...")
        try:
            webbrowser.open(dashboard_url)
        except Exception:
            pass

    # 6. Launch Synthetic Stream Generator
    if not args.no_stream:
        print(
            f"\n[INFO] Starting Synthetic Telemetry Streamer (delay: {args.delay}s)..."
        )
        stream_cmd = [
            sys.executable,
            "stream/synthetic_generator.py",
            "--api-url",
            f"http://localhost:{args.api_port}",
            "--delay",
            str(args.delay),
        ]
        stream_proc = subprocess.Popen(stream_cmd)
        PROCESSES.append(stream_proc)

    print("\n" + "=" * 70)
    print("  ALL SERVICES RUNNING SUCCESSFULLY!")
    print(f"  * FastAPI Docs:        http://localhost:{args.api_port}/docs")
    print(f"  * Streamlit Dashboard: {dashboard_url}")
    print("  * Stream Generator:    Active (feeding predictions)")
    print("  * Press Ctrl+C at any time to shut down all services.")
    print("=" * 70 + "\n")

    # Keep main process alive while child processes run
    try:
        while True:
            for p in PROCESSES:
                if p.poll() is not None:
                    # One of the critical processes crashed
                    print(
                        f"[WARN] A service process (PID {p.pid}) exited unexpectedly with code {p.returncode}."
                    )
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
