#!/bin/bash

# 1. Set Hadoop File System Path
FS="hdfs://localhost:9000"

# 2. Set Path to the JAR file
JAR_PATH="$(pwd)/target/retail-recommendation-1.0-SNAPSHOT.jar"

# 3. Input Path (Your /dunnhumby folder)
INPUT_BASE="$FS/dunnhumby"

echo "======================================================="
echo "Running ALS Recommender System (Local)"
echo "Input:  $INPUT_BASE"
echo "Output: $INPUT_BASE/predictions/final_recommendation.parquet"
echo "======================================================="

spark-submit \
  --class com.retail.ml.ProductRecommendation \
  --master "local[*]" \
  --conf spark.driver.memory=4g \
  --conf spark.hadoop.fs.defaultFS="$FS" \
  "$JAR_PATH" \
  "$INPUT_BASE"
