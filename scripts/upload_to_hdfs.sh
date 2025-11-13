#!/usr/bin/env bash
# Upload preprocessed data to HDFS (placeholder script)
set -euo pipefail

LOCAL_PATH="$1"
HDFS_PATH="$2"

echo "Uploading $LOCAL_PATH to HDFS $HDFS_PATH"
hdfs dfs -mkdir -p "$HDFS_PATH"
hdfs dfs -put -f "$LOCAL_PATH"/* "$HDFS_PATH/"

echo "Upload complete"
