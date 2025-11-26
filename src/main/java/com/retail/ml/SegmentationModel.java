package com.retail.ml;

import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;

import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;

import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;

import org.apache.spark.ml.evaluation.ClusteringEvaluator;

import org.apache.spark.sql.Encoders;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SegmentationModel {

    // Simple POJO for storing metrics so we can use Encoders.bean(...)
    public static class MetricRecord implements Serializable {
        private int k;
        private double silhouette;
        private double wssse;

        // No-arg constructor required for bean encoder
        public MetricRecord() {}

        public MetricRecord(int k, double silhouette, double wssse) {
            this.k = k;
            this.silhouette = silhouette;
            this.wssse = wssse;
        }

        public int getK() { return k; }
        public void setK(int k) { this.k = k; }

        public double getSilhouette() { return silhouette; }
        public void setSilhouette(double silhouette) { this.silhouette = silhouette; }

        public double getWssse() { return wssse; }
        public void setWssse(double wssse) { this.wssse = wssse; }
    }

    public static void main(String[] args) throws Exception {

        SparkSession spark = SparkSession.builder()
                .appName("SegmentationModel")
                .master("local[*]")
                .getOrCreate();

        System.out.println("\n========== LOADING FEATURES ==========\n");

        Dataset<Row> features = spark.read()
                .parquet("hdfs://localhost:9000/user/ads/retail_project/processed/features/");

        features = features.na().fill(0);

        System.out.println("Loaded rows: " + features.count());
        features.show(5, false);

        // ========================================
        // COMMON PIPELINE STAGES
        // ========================================
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{
                        "total_spend",
                        "total_qty",
                        "num_visits",
                        "unique_products",
                        "unique_brands",
                        "avg_basket_value"
                })
                .setOutputCol("features");

        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(false);

        // ========================================
        // AUTO K-TUNING
        // ========================================
        System.out.println("\n========== FINDING BEST K ==========\n");

        int[] kValues = {2, 3, 4, 5, 6, 7, 8};

        double bestSilhouette = Double.NEGATIVE_INFINITY;
        int bestK = kValues[0];
        double bestWSSSE = Double.MAX_VALUE;

        List<MetricRecord> metricsList = new ArrayList<>();

        for (int k : kValues) {

            KMeans kmeans = new KMeans()
                    .setK(k)
                    .setSeed(123)
                    .setFeaturesCol("scaledFeatures")
                    .setPredictionCol("cluster");

            Pipeline pipeline = new Pipeline()
                    .setStages(new PipelineStage[]{
                            assembler,
                            scaler,
                            kmeans
                    });

            PipelineModel model = pipeline.fit(features);

            Dataset<Row> prediction = model.transform(features);

            // ---- Silhouette ----
            ClusteringEvaluator evaluator = new ClusteringEvaluator()
                    .setFeaturesCol("scaledFeatures")
                    .setPredictionCol("cluster")
                    .setMetricName("silhouette");

            double silhouette = evaluator.evaluate(prediction);

            // ---- WSSSE ---- (Spark 3.x)
            KMeansModel kModel = (KMeansModel) model.stages()[2];
            double wssse = kModel.summary().trainingCost();

            System.out.println("K = " + k + " | Silhouette = " + silhouette + " | WSSSE = " + wssse);

            // Save metric to list (bean)
            metricsList.add(new MetricRecord(k, silhouette, wssse));

            // Choose best K by silhouette (break ties by lower WSSSE)
            if (silhouette > bestSilhouette || (silhouette == bestSilhouette && wssse < bestWSSSE)) {
                bestSilhouette = silhouette;
                bestK = k;
                bestWSSSE = wssse;
            }
        }

        System.out.println("\n========== BEST K FOUND ==========");
        System.out.println("Best K = " + bestK);
        System.out.println("Silhouette = " + bestSilhouette);
        System.out.println("WSSSE = " + bestWSSSE);
        System.out.println("=================================\n");

        // ============================================
        // SAVE METRICS TO HDFS (as JSON)
        // ============================================
        Dataset<Row> metricsDF = spark.createDataset(metricsList, Encoders.bean(MetricRecord.class)).toDF();

        String metricsPath = "hdfs://localhost:9000/user/ads/retail_project/model/segmentation_metrics/";
        metricsDF.write().mode("overwrite").json(metricsPath);

        System.out.println("Saved evaluation metrics to:");
        System.out.println(metricsPath);

        // Save Best K
        Dataset<Row> bestKDF = spark.createDataset(
                Arrays.asList(bestK),
                Encoders.INT()
        ).toDF("best_k");

        String bestKPath = "hdfs://localhost:9000/user/ads/retail_project/model/best_k/";
        bestKDF.write().mode("overwrite").json(bestKPath);

        System.out.println("Saved BEST K to:");
        System.out.println(bestKPath);

        // ============================================
        // TRAIN FINAL MODEL WITH BEST K
        // ============================================
        System.out.println("\n========== TRAINING FINAL MODEL ==========\n");

        KMeans bestKMeans = new KMeans()
                .setK(bestK)
                .setSeed(123)
                .setFeaturesCol("scaledFeatures")
                .setPredictionCol("cluster");

        Pipeline finalPipeline = new Pipeline()
                .setStages(new PipelineStage[]{
                        assembler,
                        scaler,
                        bestKMeans
                });

        PipelineModel finalModel = finalPipeline.fit(features);

        // Save final model
        String modelPath = "hdfs://localhost:9000/user/ads/retail_project/model/segmentation_model/";
        finalModel.write().overwrite().save(modelPath);

        System.out.println("\n========== FINAL MODEL SAVED ==========");
        System.out.println(modelPath);
        System.out.println("=========================================\n");

        spark.stop();
    }
}
