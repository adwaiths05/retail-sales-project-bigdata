package com.retail.ingestion;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class DataLoader {

    public static Dataset<Row> loadCsv(SparkSession spark, String path) {
        return spark.read().option("header", "true").option("inferSchema", "true").csv(path);
    }
}
