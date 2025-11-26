package com.retail.ml;

import org.apache.spark.sql.*;
import static org.apache.spark.sql.functions.*;   // <-- REQUIRED IMPORT

public class SegmentationFeatureEngineering {

    public static void main(String[] args) throws Exception {

        SparkSession spark = SparkSession.builder()
                .appName("SegmentationFeatureEngineering")
                .master("local[*]")
                .getOrCreate();

        String basePath = "hdfs://localhost:9000/user/ads/retail_project/raw/";

        Dataset<Row> transactions = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(basePath + "transaction_data.csv")
                .drop("TRANS_TIME");

        Dataset<Row> products = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(basePath + "product.csv");

        Dataset<Row> demo = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(basePath + "hh_demographic.csv");

        Dataset<Row> joined = transactions
                .join(products, "PRODUCT_ID")
                .join(demo, "household_key");

        Dataset<Row> features = joined.groupBy("household_key")
                .agg(
                        sum("SALES_VALUE").alias("total_spend"),
                        sum("QUANTITY").alias("total_qty"),
                        countDistinct("BASKET_ID").alias("num_visits"),
                        countDistinct("PRODUCT_ID").alias("unique_products"),
                        countDistinct("BRAND").alias("unique_brands"),
                        avg("SALES_VALUE").alias("avg_basket_value")
                );

        features.write()
                .mode(SaveMode.Overwrite)
                .parquet("hdfs://localhost:9000/user/ads/retail_project/processed/features/");

        spark.stop();
    }
}
