# Market Basket Analysis – Spark Job

This README explains how to run the **Market Basket Analysis** Spark job included in the `retail-ml-pipeline` project.

---

## 📂 Project Location

Before running anything, ensure you are inside the project directory:

```bash
cd ~/retail-ml-pipeline
```

---

## 🔧 Step 1 — Detect `fs.defaultFS`

Spark requires the correct HDFS filesystem URI. Use the following command to dynamically detect it from Hadoop's configuration:

```bash
FS="$(xmllint --xpath "string(//configuration/property[name='fs.defaultFS']/value)" /usr/local/hadoop/etc/hadoop/core-site.xml 2>/dev/null)"
FS=${FS:-hdfs://localhost:9000}
echo "Using fs.defaultFS = $FS"
```

If detection fails, the fallback value will be:

```
hdfs://localhost:9000
```

---

## 📦 Step 2 — Set JAR Path

The compiled application JAR is inside the `target/` directory:

```bash
ABSJAR="$(pwd)/target/retail-ml-pipeline-1.0-SNAPSHOT.jar"
```

---

## 📥 Step 3 — Input & Output Paths

Update these only if your dataset paths differ:

```bash
INPUT="hdfs://localhost:9000/user/hadoopusr/data/raw/Preprocessed_spark_data_unpacked/transaction_data.csv"
OUTPUT="hdfs://localhost:9000/user/hadoopusr/analysis_out/market_basket"
```

---

## 🚀 Step 4 — Run the Market Basket Analysis Job

Use the following full Spark command:

```bash
HADOOP_USER_NAME=hadoopusr \
  spark-submit \
  --class MarketBasketAnalysis \
  --master local[*] \
  --conf spark.hadoop.fs.defaultFS="$FS" \
  --conf spark.ui.showConsoleProgress=false \
  "$ABSJAR" \
  "$INPUT" \
  "$OUTPUT" \
  basket_id \
  product_id \
  0.01 \
  4
```

### Parameter Explanation

| Argument     | Meaning                                        |
| ------------ | ---------------------------------------------- |
| `basket_id`  | Column grouping items into a basket            |
| `product_id` | Column identifying the product per transaction |
| `0.01`       | Minimum support threshold                      |
| `4`          | Number of partitions for FP-Growth             |

---

## 📤 Output

The results will be stored in:

```
hdfs://localhost:9000/user/hadoopusr/analysis_out/market_basket
```

This directory contains frequent itemsets and association rules.

---

## ✅ Summary

This README provides:

- The correct directory to run the job
- How to detect HDFS filesystem
- How to locate your compiled JAR
- The complete Spark command to run Market Basket Analysis

You're ready to run FP-Growth at scale on the Dunnhumby dataset 🚀
