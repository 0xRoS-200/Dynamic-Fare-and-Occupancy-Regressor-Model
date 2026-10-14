import argparse
import os
import subprocess
import sys

# Compatibility bridge for kafka-python on Python 3.12+
try:
    from kafka import KafkaAdminClient
except ImportError:
    try:
        import six
        sys.modules.setdefault("kafka.vendor.six", six)
        sys.modules.setdefault("kafka.vendor.six.moves", six.moves)
        from kafka import KafkaAdminClient
    except Exception:
        KafkaAdminClient = None


def check_hdfs(local_fallback: bool = False) -> bool:
    """Returns True if HDFS path accessible, or if local_fallback=True and ./hdfs_local/ exists."""
    try:
        if local_fallback:
            return os.path.isdir(os.path.join(".", "hdfs_local", "transit", "raw"))
        res = subprocess.run(
            ["hdfs", "dfs", "-ls", "/transit"],
            capture_output=True,
            timeout=10,
        )
        return res.returncode == 0
    except Exception:
        return False


def check_kafka(broker: str = "localhost:9092", topic: str = "transit_live") -> bool:
    """Returns True if Kafka broker is reachable and topic exists. Returns False on any connection error."""
    try:
        if KafkaAdminClient is None:
            return False
        admin = KafkaAdminClient(bootstrap_servers=broker, request_timeout_ms=3000)
        try:
            topics = admin.list_topics()
            return topic in topics
        finally:
            admin.close()
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-flight infrastructure health check")
    parser.add_argument(
        "--local-fallback",
        action="store_true",
        default=False,
        help="Use local filesystem fallback instead of HDFS",
    )
    args = parser.parse_args()

    hdfs_ok = check_hdfs(local_fallback=args.local_fallback)
    kafka_ok = check_kafka()

    print(f"HDFS Status: {'READY' if hdfs_ok else 'UNAVAILABLE'} (local_fallback={args.local_fallback})")
    print(f"Kafka Status: {'READY' if kafka_ok else 'UNAVAILABLE'}")

    if hdfs_ok and kafka_ok:
        print("Infrastructure check: PASSED")
        return 0
    else:
        print("Infrastructure check: FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

