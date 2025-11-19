package com.retail.ml;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.PipelineStage;

import java.util.Arrays;
import static org.apache.spark.sql.functions.col; // Import col function

public class ChurnPrediction {

    public static void main(String[] args) throws Exception {
        // Robust arg parsing: supports "--base=value" or "--base value"
        String basePath = null;
        Integer repurchaseWindowDays = null;
        for (int i = 0; i < args.length; i++) {
            if (args[i].startsWith("--base=")) {
                basePath = args[i].split("=", 2)[1];
            } else if (args[i].equals("--base") && i + 1 < args.length) {
                basePath = args[i + 1];
            } else if (args[i].startsWith("--window=")) {
                repurchaseWindowDays = Integer.parseInt(args[i].split("=", 2)[1]);
            } else if (args[i].equals("--window") && i + 1 < args.length) {
                repurchaseWindowDays = Integer.parseInt(args[i + 1]);
            }
        }
        if (basePath == null) basePath = "hdfs:///user/kiran/dunnhunby";
        if (repurchaseWindowDays == null) repurchaseWindowDays = 90;

        System.out.println("BASE PATH = " + basePath);
        System.out.println("REPURCHASE WINDOW (days) = " + repurchaseWindowDays);

        SparkSession spark = SparkSession.builder()
                .appName("ChurnPrediction")
                .getOrCreate();

        spark.sparkContext().setLogLevel("WARN");

        // Paths (we accept both hdfs://... and file:/... or absolute)
        String txPath = String.format("%s/transaction_data.csv", basePath);
        String couponRedPath = String.format("%s/coupon_redempt.csv", basePath);
        String hhPath = String.format("%s/hh_demographic.csv", basePath);

        // -------- READ (keep original column names) --------
        Dataset<Row> tx = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(txPath)
                // Added RETAIL_DISC and COUPON_DISC
                .select("household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE", "STORE_ID", "WEEK_NO", "RETAIL_DISC", "COUPON_DISC")
                .withColumnRenamed("DAY", "day")
                .alias("tx");

        Dataset<Row> couponRed = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(couponRedPath)
                .select("household_key", "DAY", "COUPON_UPC")
                .withColumnRenamed("DAY", "day")
                .alias("cr");

        Dataset<Row> hh = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(hhPath)
                .alias("hh");

        // ------------- FEATURE: first purchase flag ---------------
        WindowSpec specHP = Window
                .partitionBy(functions.col("tx.household_key"), functions.col("tx.PRODUCT_ID"))
                .orderBy(functions.col("tx.day").asc());

        Dataset<Row> txWithRow = tx.withColumn("row_num", functions.row_number().over(specHP))
                .withColumn("is_first_purchase", functions.when(functions.col("row_num").equalTo(1), 1).otherwise(0))
                .alias("txw");

        // coupon days (per household) alias as cd
        Dataset<Row> couponDays = couponRed.select(
                functions.col("cr.household_key").alias("cd_household_key"),
                functions.col("cr.day").alias("coupon_day")
        ).distinct().alias("cd");

        // join to mark whether the purchase day had a coupon redemption by the household
        Dataset<Row> txFlagged = txWithRow.join(couponDays,
                functions.expr("txw.household_key = cd.cd_household_key AND txw.day = cd.coupon_day"),
                "left")
                .withColumn("first_purchase_with_coupon",
                        functions.when(functions.col("txw.is_first_purchase").equalTo(1)
                                .and(functions.col("coupon_day").isNotNull()), 1).otherwise(0))
                .select(
                        functions.col("txw.household_key").alias("household_key"),
                        functions.col("txw.BASKET_ID").alias("BASKET_ID"),
                        functions.col("txw.day").alias("day"),
                        functions.col("txw.PRODUCT_ID").alias("PRODUCT_ID"),
                        functions.col("txw.QUANTITY").alias("QUANTITY"), // Keep QUANTITY
                        functions.col("txw.SALES_VALUE").alias("SALES_VALUE"), // Keep SALES_VALUE
                        functions.col("txw.STORE_ID").alias("STORE_ID"),
                        functions.col("txw.WEEK_NO").alias("WEEK_NO"),
                        functions.col("txw.RETAIL_DISC").alias("RETAIL_DISC"), // Keep RETAIL_DISC
                        functions.col("txw.COUPON_DISC").alias("COUPON_DISC"), // Keep COUPON_DISC
                        functions.col("txw.row_num").alias("row_num"),
                        functions.col("txw.is_first_purchase").alias("is_first_purchase"),
                        functions.col("first_purchase_with_coupon")
                ).alias("txf");

        // ------------- Build first purchase DataFrame ---------------
        Dataset<Row> firstPurch = txFlagged.filter(functions.col("txf.is_first_purchase").equalTo(1))
                .select(functions.col("txf.household_key"),
                        functions.col("txf.PRODUCT_ID"),
                        functions.col("txf.day").alias("first_day"),
                        functions.col("txf.first_purchase_with_coupon"),
                        // Add new features
                        functions.col("txf.QUANTITY").alias("first_purchase_quantity"),
                        functions.col("txf.SALES_VALUE").alias("first_purchase_value"),
                        functions.col("txf.RETAIL_DISC").alias("first_purchase_retail_disc"),
                        functions.col("txf.COUPON_DISC").alias("first_purchase_coupon_disc"),
                        // ******** MODIFICATION IS HERE (line 121) ********
                        // Replaced dayofweek() with a math expression
                        functions.expr("((txf.day - 1) % 7) + 1").alias("first_day_of_week")
                )
                .alias("fp");


        // ------------- Find repurchases within window (use aliases) ---------------
        Dataset<Row> laterPurch = tx.select(
                functions.col("tx.household_key").alias("lp_household_key"),
                functions.col("tx.PRODUCT_ID").alias("lp_PRODUCT_ID"),
                functions.col("tx.day").alias("later_day")
        ).alias("lp");

        Dataset<Row> fpAlias = firstPurch.alias("fp");
        Dataset<Row> lpAlias = laterPurch.alias("lp");

        Dataset<Row> joined = fpAlias.join(
                lpAlias,
                functions.expr("fp.household_key = lp.lp_household_key AND fp.PRODUCT_ID = lp.lp_PRODUCT_ID"),
                "inner"
        ).filter(
                functions.col("lp.later_day").gt(functions.col("fp.first_day"))
                        .and(functions.col("lp.later_day").leq(functions.col("fp.first_day").plus(functions.lit(repurchaseWindowDays))))
        ).alias("joined");

        // ------------- For each pair, get first re-purchase day and count -------------
        Dataset<Row> repurchaseFlag = joined.groupBy(
                // Pass new features through the groupBy
                functions.col("household_key").alias("household_key"),
                functions.col("PRODUCT_ID").alias("PRODUCT_ID"),
                functions.col("first_day").alias("first_day"),
                functions.col("first_purchase_with_coupon").alias("first_purchase_with_coupon"),
                functions.col("first_purchase_quantity").alias("first_purchase_quantity"),
                functions.col("first_purchase_value").alias("first_purchase_value"),
                functions.col("first_purchase_retail_disc").alias("first_purchase_retail_disc"),
                functions.col("first_purchase_coupon_disc").alias("first_purchase_coupon_disc"),
                functions.col("first_day_of_week").alias("first_day_of_week")
        ).agg(
                functions.min(functions.col("later_day")).alias("first_repurchase_day"),
                functions.count(functions.col("later_day")).alias("repurchase_count")
        ).withColumn("repurchased_within_window", functions.lit(1)).alias("r");


        // Left join to include non-repurchased pairs; qualify columns on select to avoid ambiguity
        Dataset<Row> df = fpAlias.join(
                repurchaseFlag.alias("r"),
                functions.expr("fp.household_key = r.household_key AND fp.PRODUCT_ID = r.PRODUCT_ID"),
                "left"
        ).select(
                functions.col("fp.household_key").alias("household_key"),
                functions.col("fp.PRODUCT_ID").alias("PRODUCT_ID"),
                functions.col("fp.first_day").alias("first_day"),
                functions.col("fp.first_purchase_with_coupon").alias("first_purchase_with_coupon"),
                // Pass new features through the select
                functions.col("fp.first_purchase_quantity"),
                functions.col("fp.first_purchase_value"),
                functions.col("fp.first_purchase_retail_disc"),
                functions.col("fp.first_purchase_coupon_disc"),
                functions.col("fp.first_day_of_week"),
                // Coalesce the columns from the repurchaseFlag
                functions.coalesce(functions.col("r.repurchase_count"), functions.lit(0)).alias("repurchase_count"),
                functions.coalesce(functions.col("r.repurchased_within_window"), functions.lit(0)).alias("repurchased_within_window"),
                functions.col("r.first_repurchase_day").alias("first_repurchase_day")
        ).alias("d");

        // ------------- Additional aggregations (Leaky, not used in model) ---------------
        Dataset<Row> txAgg = tx.groupBy(
                functions.col("tx.household_key").alias("agg_household_key"),
                functions.col("tx.PRODUCT_ID").alias("agg_PRODUCT_ID")
        ).agg(
                functions.count("*").alias("total_purchases"),
                functions.sum(functions.col("tx.SALES_VALUE")).alias("total_spend"),
                functions.avg(functions.col("tx.SALES_VALUE")).alias("avg_spend")
        ).alias("t");

        // Join aggregated tx stats -> be explicit in selection
        Dataset<Row> dfFeat = df.alias("d").join(txAgg.alias("t"),
                functions.expr("d.household_key = t.agg_household_key AND d.PRODUCT_ID = t.agg_PRODUCT_ID"),
                "left")
                .select(
                        col("d.household_key"),
                        col("d.PRODUCT_ID"),
                        col("d.first_day"),
                        col("d.first_purchase_with_coupon"),
                        col("d.repurchased_within_window"),
                        // Pass new features through
                        col("d.first_purchase_quantity"),
                        col("d.first_purchase_value"),
                        col("d.first_purchase_retail_disc"),
                        col("d.first_purchase_coupon_disc"),
                        col("d.first_day_of_week"),
                        // Leaky features (we calculate them, but won't use them)
                        col("d.repurchase_count"),
                        functions.when(col("d.repurchased_within_window").equalTo(1),
                                col("d.first_repurchase_day").minus(col("d.first_day")))
                                .otherwise(functions.lit(9999)).alias("days_to_repurchase"),
                        functions.coalesce(col("t.total_purchases"), functions.lit(0)).alias("total_purchases"),
                        functions.coalesce(col("t.total_spend"), functions.lit(0.0)).alias("total_spend"),
                        functions.coalesce(col("t.avg_spend"), functions.lit(0.0)).alias("avg_spend")
                ).alias("dfFeat");

        Dataset<Row> dfWithHH = dfFeat.alias("df").join(hh.alias("h"),
                functions.expr("df.household_key = h.household_key"),
                "left")
                .select(
                        col("df.household_key"),
                        col("df.PRODUCT_ID"),
                        col("df.first_day"),
                        col("df.first_purchase_with_coupon"),
                        col("df.repurchased_within_window"),
                        // Pass new features through
                        col("df.first_purchase_quantity"),
                        col("df.first_purchase_value"),
                        col("df.first_purchase_retail_disc"),
                        col("df.first_purchase_coupon_disc"),
                        col("df.first_day_of_week"),
                        // Demographics
                        functions.coalesce(col("h.AGE_DESC"), functions.lit("UNKNOWN")).alias("AGE_DESC"),
                        functions.coalesce(col("h.INCOME_DESC"), functions.lit("UNKNOWN")).alias("INCOME_DESC"),
                        functions.coalesce(col("h.HOMEOWNER_DESC"), functions.lit("UNKNOWN")).alias("HOMEOWNER_DESC")
                ).alias("dfWithHH");

        Dataset<Row> mlDF = dfWithHH.select(
                col("repurchased_within_window").cast("double").alias("label"),
                // Features
                col("first_purchase_with_coupon").cast("double"),
                // Add new features to the ML dataset
                col("first_purchase_quantity").cast("double"),
                col("first_purchase_value").cast("double"),
                col("first_purchase_retail_disc").cast("double"),
                col("first_purchase_coupon_disc").cast("double"),
                col("first_day_of_week").cast("double"),
                // Categorical Features
                col("AGE_DESC"),
                col("INCOME_DESC"),
                col("HOMEOWNER_DESC")
        );

        // ------------- Categorical encoding ---------------
        String[] catCols = new String[]{"AGE_DESC", "INCOME_DESC", "HOMEOWNER_DESC"};
        String[] indexCols = Arrays.stream(catCols).map(c -> c + "_idx").toArray(String[]::new);
        String[] oheCols = Arrays.stream(catCols).map(c -> c + "_ohe").toArray(String[]::new);

        // Indexers
        StringIndexer[] indexers = new StringIndexer[catCols.length];
        for (int i = 0; i < catCols.length; i++) {
            indexers[i] = new StringIndexer().setInputCol(catCols[i]).setOutputCol(indexCols[i]).setHandleInvalid("keep");
        }

        OneHotEncoder encoder = new OneHotEncoder()
                .setInputCols(indexCols)
                .setOutputCols(oheCols)
                .setHandleInvalid("keep");


        String[] numericFeatureCols = new String[]{
            "first_purchase_with_coupon",
            "first_purchase_quantity",
            "first_purchase_value",
            "first_purchase_retail_disc",
            "first_purchase_coupon_disc",
            "first_day_of_week"
        };
        
        String[] featureCols = concatenate(numericFeatureCols, oheCols);

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("raw_features")
                .setHandleInvalid("keep");

        StandardScaler scaler = new StandardScaler()
                .setInputCol("raw_features")
                .setOutputCol("features")
                .setWithMean(false)
                .setWithStd(true);

        // RandomForest
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setNumTrees(100)
                .setMaxDepth(8)
                .setSeed(42);

        // Build pipeline stages
        PipelineStage[] stages = buildStages(indexers, encoder, assembler, scaler, rf);
        Pipeline pipeline = new Pipeline().setStages(stages);

        // Train/test split
        Dataset<Row>[] split = mlDF.randomSplit(new double[]{0.8, 0.2}, 42L);
        Dataset<Row> train = split[0];
        Dataset<Row> test = split[1];

        System.out.println("Training rows: " + train.count() + "   Test rows: " + test.count());

        PipelineModel model = pipeline.fit(train);

        // Evaluate
        Dataset<Row> preds = model.transform(test);
        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("label")
                .setRawPredictionCol("rawPrediction")
                .setMetricName("areaUnderROC");

        double auc = evaluator.evaluate(preds);
        System.out.println("Test AUC = " + auc);

        // Cohort aggregated metrics: repurchase rate by first_purchase_with_coupon
        Dataset<Row> cohort = mlDF.groupBy("first_purchase_with_coupon")
                .agg(functions.count("*").alias("n_pairs"),
                        functions.sum("label").alias("n_repurchases"),
                        functions.expr("sum(label)/count(*)").alias("repurchase_rate"));
        System.out.println("Cohort repurchase rates:");
        cohort.show(false);

        // Save the model to basePath/models/churn_rf
        String modelPath = basePath + "/models/churn_rf";
        System.out.println("Saving model to: " + modelPath);
        model.write().overwrite().save(modelPath);

        System.out.println("Done.");
        spark.stop();
    }

    private static String[] concatenate(String[] a, String[] b) {
        String[] res = new String[a.length + b.length];
        System.arraycopy(a, 0, res, 0, a.length);
        System.arraycopy(b, 0, res, a.length, b.length);
        return res;
    }

    private static PipelineStage[] buildStages(StringIndexer[] indexers, OneHotEncoder encoder, VectorAssembler assembler, StandardScaler scaler, RandomForestClassifier rf) {
        PipelineStage[] stages = new PipelineStage[indexers.length + 3];
        int idx = 0;
        for (StringIndexer s : indexers) {
            stages[idx++] = s;
        }
        stages[idx++] = encoder;
        stages[idx++] = assembler;
        stages[idx++] = scaler;
        // Append RF as last stage
        PipelineStage[] full = new PipelineStage[stages.length + 1];
        System.arraycopy(stages, 0, full, 0, stages.length);
        full[full.length - 1] = rf;
        return full;
    }
}