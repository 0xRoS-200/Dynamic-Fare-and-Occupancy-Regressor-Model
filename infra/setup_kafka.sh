#!/usr/bin/env bash
set -e

if [ -z "$KAFKA_HOME" ]; then
    echo "KAFKA_HOME environment variable is not set."
    echo "Please download and install Kafka from: https://kafka.apache.org/downloads"
    exit 1
fi

echo "Starting ZooKeeper..."
"$KAFKA_HOME/bin/zookeeper-server-start.sh" "$KAFKA_HOME/config/zookeeper.properties" &

echo "Waiting 5 seconds for ZooKeeper to start..."
sleep 5

echo "Starting Kafka server..."
"$KAFKA_HOME/bin/kafka-server-start.sh" "$KAFKA_HOME/config/server.properties" &

echo "Waiting 5 seconds for Kafka broker to start..."
sleep 5

echo "Creating transit_live topic..."
"$KAFKA_HOME/bin/kafka-topics.sh" --create --topic transit_live --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092 --if-not-exists

echo "Kafka ready"

