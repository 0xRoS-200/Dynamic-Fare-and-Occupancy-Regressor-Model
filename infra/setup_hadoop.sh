#!/usr/bin/env bash
set -e

if [ "$1" = "--local-fallback" ]; then
    mkdir -p ./hdfs_local/transit/raw ./hdfs_local/transit/features ./hdfs_local/transit/models
    echo "HDFS local fallback ready at ./hdfs_local/"
    exit 0
fi

if [ -z "$HADOOP_HOME" ] || ! command -v hdfs &> /dev/null; then
    echo "Hadoop is not configured or hdfs is not in PATH."
    echo "Please install Hadoop from: https://hadoop.apache.org/releases.html"
    exit 1
fi

if [ ! -d "$HADOOP_HOME/logs" ]; then
    echo "Formatting HDFS namenode..."
    hdfs namenode -format -force
fi

echo "Starting HDFS daemons..."
start-dfs.sh

echo "Creating HDFS transit directories..."
hdfs dfs -mkdir -p /transit/raw /transit/features /transit/models

echo "HDFS ready"

