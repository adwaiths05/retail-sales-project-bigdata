
package com.retail.ml;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import static org.apache.spark.sql.functions.col;

// Imports for File Renaming
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.conf.Configuration;

public class ProductRecommendation {

    public static void main(String[] args) throws Exception {
        
        // 1. Setup Paths
        // Default to localhost HDFS
        String basePath = (args.length > 0) ? args[0] : "hdfs://localhost:9000/dunnhumby";
        // The specific name you requested
        String finalFileName = "final_recommendation.parquet"; 
        
        // We will write to a temp folder first, then move the file
        String tempOutputPath = basePath + "/predictions/temp_recs";
        String finalOutputPath = basePath + "/predictions/" + finalFileName;

        SparkSession spark = SparkSession.builder()
                .appName("ProductRecommendation")
                .getOrCreate();

        spark.sparkContext().setLogLevel("WARN");

        System.out.println("Reading data from: " + basePath);

        // 2. Load Data (Standard Logic)
        Dataset<Row> tx = spark.read().option("header", "true").option("inferSchema", "true")
                .csv(basePath + "/transaction_data.csv")
                .select("household_key", "PRODUCT_ID", "SALES_VALUE");

        Dataset<Row> products = spark.read().option("header", "true").option("inferSchema", "true")
                .csv(basePath + "/product.csv")
                .select("PRODUCT_ID", "COMMODITY_DESC", "SUB_COMMODITY_DESC");

        Dataset<Row> ratings = tx.groupBy("household_key", "PRODUCT_ID")
                .agg(functions.sum("SALES_VALUE").alias("rating"));

        // 3. Train Model
        ALS als = new ALS()
                .setMaxIter(10)
                .setRegParam(0.1)
                .setUserCol("household_key")
                .setItemCol("PRODUCT_ID")
                .setRatingCol("rating")
                .setImplicitPrefs(true)
                .setColdStartStrategy("drop");

        ALSModel model = als.fit(ratings);

        // 4. Generate Recommendations
        Dataset<Row> userRecs = model.recommendForAllUsers(5);

        Dataset<Row> explodedRecs = userRecs
                .withColumn("rec", functions.explode(col("recommendations")))
                .select(col("household_key"), col("rec.PRODUCT_ID").alias("PRODUCT_ID"), col("rec.rating").alias("prediction_score"));

        Dataset<Row> finalRecs = explodedRecs.join(products, "PRODUCT_ID")
                .select(col("household_key"), col("PRODUCT_ID"), col("COMMODITY_DESC"), col("prediction_score"));

        // 5. Write and Rename (The "Clean Output" Logic)
        
        System.out.println("Writing temporary output to: " + tempOutputPath);
        
        // Write as a single part file to the temp folder
        finalRecs.coalesce(1)
                .write()
                .mode("overwrite")
                .parquet(tempOutputPath);

        // Rename logic
        Configuration hadoopConf = spark.sparkContext().hadoopConfiguration();
        FileSystem fs = FileSystem.get(hadoopConf);
        
        Path tempDir = new Path(tempOutputPath);
        Path finalFile = new Path(finalOutputPath);

        // Find the generated part file
        FileStatus[] parts = fs.globStatus(new Path(tempDir, "part-*.parquet"));

        if (parts.length > 0) {
            // Delete the old final file if it exists (so we can overwrite)
            if (fs.exists(finalFile)) {
                fs.delete(finalFile, false);
            }
            
            // Rename temp part file to "final_recommendation.parquet"
            fs.rename(parts[0].getPath(), finalFile);
            System.out.println("SUCCESS: Renamed output to: " + finalOutputPath);
        } else {
            System.err.println("ERROR: Could not find generated part file.");
        }

        // Clean up temp folder
        fs.delete(tempDir, true);

        spark.stop();
    }
}
