# Product Recommendation System – Spark Job



This README explains how to run the **ALS Collaborative Filtering** Spark job included in the `retail-recommendation` project. This model generates personalized "Next Best Offer" product recommendations for every household using implicit feedback (sales value).



## 📂 Project Location



Before running anything, ensure you are inside the project directory:



```bash

cd ~/retail-recommendation
```


🔧 Step 1 — Detect fs.defaultFS

Spark requires the correct HDFS filesystem URI. Use the following command to dynamically detect it from Hadoop's configuration:


```
FS="$(xmllint --xpath "string(//configuration/property[name='fs.defaultFS']/value)" /usr/local/hadoop/etc/hadoop/core-site.xml 2>/dev/null)"

FS=${FS:-hdfs://localhost:9000}

echo "Using fs.defaultFS = $FS"

```


📦 Step 2 — Set JAR Path
The compiled application JAR is inside the target/ directory:�

```
ABSJAR="$(pwd)/target/retail-recommendation-1.0-SNAPSHOT.jar"
```

🚀 Step 4 — Run the Recommendation Job
Use the following full Spark command. This runs the ALS (Alternating Least Squares) algorithm, trains the matrix factorization model, and exports the results.


```
spark-submit \
  --class com.retail.ml.ProductRecommendation \
  --master local[*] \
  --conf spark.driver.memory=4g \
  --conf spark.hadoop.fs.defaultFS="$FS" \
  --conf spark.ui.showConsoleProgress=false \
  "$ABSJAR" \
  "$INPUT_BASE"
  ```

📤 Output
The results will be saved as a single Parquet file in:

```
<INPUT_BASE>/predictions/final_recommendation.parquet
```

Schema of the Output:

household_key: The customer ID.

PRODUCT_ID: The recommended product ID.

COMMODITY_DESC: The human-readable product name.

prediction_score: The confidence score (higher is better).

✅ Summary
This README provides:

The correct directory to run the job.

How to detect the HDFS filesystem.

How to locate your compiled JAR.

The complete Spark command to generate personalized recommendations.

You're ready to run ALS Matrix Factorization on the Dunnhumby dataset! 🚀
