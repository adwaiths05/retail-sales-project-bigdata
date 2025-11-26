package com.retail.ml;

import org.apache.spark.sql.*;
import org.apache.spark.sql.types.*;
import static org.apache.spark.sql.functions.*;

import org.apache.spark.ml.PipelineModel;

public class SegmentationPrediction {

    public static void main(String[] args) throws Exception {

        SparkSession spark = SparkSession.builder()
                .appName("SegmentationPrediction")
                .master("local[*]")
                .getOrCreate();

        if (args.length == 0) {
            System.out.println("Usage:");
            System.out.println("  Predict for household:  household_key <id>");
            System.out.println("  Predict manually:       manual <total_spend> <total_qty> <num_visits> <unique_products> <unique_brands> <avg_basket_value>");
            System.exit(1);
        }

        // Load saved trained pipeline model
        String modelPath = "hdfs://localhost:9000/user/ads/retail_project/model/segmentation_model/";
        PipelineModel model = PipelineModel.load(modelPath);

        // =====================================================================
        // MODE 1 — Predict cluster for an existing household_key
        // =====================================================================
        if (args[0].equalsIgnoreCase("household_key")) {

            int key = Integer.parseInt(args[1]);

            Dataset<Row> features = spark.read()
                    .parquet("hdfs://localhost:9000/user/ads/retail_project/processed/features/")
                    .filter(col("household_key").equalTo(key));

            if (features.count() == 0) {
                System.out.println(" ERROR: Household key " + key + " not found.");
                spark.stop();
                return;
            }

            Dataset<Row> pred = model.transform(features);

            System.out.println("\n================ PREDICTION RESULT ================");
            pred.select("household_key", "cluster").show(false);
            System.out.println("===================================================\n");

            spark.stop();
            return;
        }

        // =====================================================================
        // MODE 2 — Manual numeric inputs for prediction
        // =====================================================================
        else if (args[0].equalsIgnoreCase("manual")) {

            if (args.length != 7) {
                System.out.println(" ERROR: Manual mode requires EXACTLY 6 numeric values.");
                System.exit(1);
            }

            double total_spend = Double.parseDouble(args[1]);
            double total_qty = Double.parseDouble(args[2]);
            double num_visits = Double.parseDouble(args[3]);
            double unique_products = Double.parseDouble(args[4]);
            double unique_brands = Double.parseDouble(args[5]);
            double avg_basket_value = Double.parseDouble(args[6]);

            // Schema for manual row
            StructType schema = new StructType()
                    .add("total_spend", DataTypes.DoubleType)
                    .add("total_qty", DataTypes.DoubleType)
                    .add("num_visits", DataTypes.DoubleType)
                    .add("unique_products", DataTypes.DoubleType)
                    .add("unique_brands", DataTypes.DoubleType)
                    .add("avg_basket_value", DataTypes.DoubleType);

            Row row = RowFactory.create(total_spend, total_qty, num_visits, unique_products, unique_brands, avg_basket_value);

            Dataset<Row> manualDf = spark.createDataFrame(
                    java.util.Collections.singletonList(row),
                    schema
            );

            Dataset<Row> pred = model.transform(manualDf);

            System.out.println("\n================ MANUAL INPUT RESULT ================");
            pred.select("features", "scaledFeatures", "cluster").show(false);
            System.out.println("=====================================================\n");

            spark.stop();
            return;
        }

        else {
            System.out.println(" ERROR: Unknown mode " + args[0]);
        }

        spark.stop();
    }
}
