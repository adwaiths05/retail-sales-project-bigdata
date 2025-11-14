#!/usr/bin/env python3
from pyspark.sql import SparkSession, functions as F
import sys

if len(sys.argv) < 3:
    print("Usage: top_products_by_count.py <joined_hdfs_path> <out_hdfs_dir> [topN]")
    sys.exit(1)

joined = sys.argv[1]
outdir = sys.argv[2].rstrip('/') + '/top_products'
topN = int(sys.argv[3]) if len(sys.argv) > 3 else 100

spark = SparkSession.builder.appName("TopProducts").getOrCreate()
df = spark.read.option("header","true").csv(joined)

# group and count
res = df.groupBy("PRODUCT_ID","product_name") \
        .agg(F.count(F.lit(1)).alias("tx_count")) \
        .orderBy(F.desc("tx_count"))

res.coalesce(1).write.mode("overwrite").option("header","true").csv(outdir)
print("Wrote top products to:", outdir)
# optionally show topN to stdout
res.limit(topN).show(truncate=False)
spark.stop()


'''
In order to run this script, use the following command from the retail-sales-project-bigdata home directory:

HADOOP_USER_NAME=hadoopusr \
spark-submit \
  --master local[*] \
  --conf spark.hadoop.fs.defaultFS="$FS" \
  --conf spark.ui.showConsoleProgress=false \
  ~/retail-ml-pipeline/top_products_by_count.py \
  "$FS/user/hadoopusr/data/derived/product_join_output_*/joined_transactions_with_names" \
  "$FS/user/hadoopusr/data/derived/product_topcounts_$(date +%s)" 50
  
'''