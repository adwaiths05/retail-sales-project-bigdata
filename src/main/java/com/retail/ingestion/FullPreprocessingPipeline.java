package com.retail.ingestion;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.types.DataTypes;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FullPreprocessingPipeline {

    public static void main(String[] args) {

        // === INPUT PATH (Single CSV File) ===
        final String INPUT_PATH = "file:///home/ads/Downloads/combined_dunnhumby_all_parts.csv";

        // === OUTPUT PATHS ===
        final String BASE_PATH = "file:///home/ads/Preprocessed_spark_data/";
        final String ITEM_PATH = BASE_PATH + "item_transactions/";
        final String CUSTOMER_PATH = BASE_PATH + "customer_segments/";
        final String TEMPORAL_PATH = BASE_PATH + "campaign_events/";

        // Outlier caps
        final double MAX_UNIT_QUANTITY = 1000.0;
        final double MAX_SALES_VALUE = 1000.0;
        final double MIN_DISCOUNT_RATE = -100.0;
        final double MAX_DISCOUNT_RATE = 100.0;

        // Initialize Spark
        SparkSession spark = SparkSession.builder()
                .appName("RetailDataPreprocessingPipeline")
                .getOrCreate();

        spark.sparkContext().setLogLevel("WARN");

        // === LOAD THE SINGLE COMBINED CSV ===
        System.out.println("Reading dataset from: " + INPUT_PATH);

        Dataset<Row> rawDf = spark.read()
                .option("header", "true")
                .option("inferSchema", "true")
                .csv(INPUT_PATH);

        System.out.println("Initial rows: " + rawDf.count());
        System.out.println("Columns: " + Arrays.toString(rawDf.columns()));

        // === STANDARDIZE COLUMN NAMES ===
        Dataset<Row> df = rawDf;
        for (String c : rawDf.columns()) {
            df = df.withColumnRenamed(c, c.toLowerCase().replace(" ", "_"));
        }

        // === BASIC IMPUTATION ===
        df = df.na().fill(0.0,
                new String[]{"sales_value", "quantity", "retail_disc", "coupon_disc", "coupon_match_disc"}
        );

        df = df.na().fill("UNKNOWN",
                new String[]{
                        "manufacturer", "department", "brand", "commodity_desc",
                        "age_desc", "income_desc", "homeowner_desc", "hh_comp_desc",
                        "household_size_desc", "kid_category_desc", "display", "mailer",
                        "coupon_upc", "campaign", "description", "start_day", "end_day",
                        "description_cd", "product_id_y"
                }
        );

        // === REMOVE NEGATIVE SALES / QUANTITY ===
        df = df.filter(functions.col("sales_value").geq(0))
               .filter(functions.col("quantity").geq(0));

        // === NET SALES ===
        df = df.withColumn("net_sales",
                functions.col("sales_value")
                        .minus(functions.col("retail_disc"))
                        .minus(functions.col("coupon_disc"))
                        .minus(functions.col("coupon_match_disc"))
        );

        // === UNIT PRICE ===
        df = df.withColumn("unit_price",
                functions.when(functions.col("quantity").gt(0),
                        functions.col("sales_value").divide(functions.col("quantity")))
                        .otherwise(0.0)
        );

        // === DISCOUNT RATE ===
        df = df.withColumn("discount_rate",
                functions.when(functions.col("sales_value").notEqual(0),
                        functions.col("retail_disc")
                                .plus(functions.col("coupon_disc"))
                                .plus(functions.col("coupon_match_disc"))
                                .divide(functions.col("sales_value"))
                                .multiply(100)
                ).otherwise(0.0)
        );

        // === DROP FULLY NULL COLUMNS ===
        long totalRows = df.count();
        List<String> dropCols = new ArrayList<>();

        for (String c : df.columns()) {
            long nulls = df.filter(functions.col(c).isNull()).count();
            if (nulls == totalRows) dropCols.add(c);
        }

        if (!dropCols.isEmpty()) {
            df = df.drop(dropCols.toArray(new String[0]));
        }

        // === OUTLIER FILTERING ===
        df = df.filter(functions.col("quantity").leq(MAX_UNIT_QUANTITY))
               .filter(functions.col("sales_value").leq(MAX_SALES_VALUE))
               .filter(functions.col("discount_rate").between(MIN_DISCOUNT_RATE, MAX_DISCOUNT_RATE));

        // === FLAG COLUMNS ===
        df = df.withColumn("is_discounted",
                functions.when(
                        functions.col("retail_disc").lt(0)
                                .or(functions.col("coupon_disc").lt(0))
                                .or(functions.col("coupon_match_disc").lt(0)), 1
                ).otherwise(0)
        );

        df = df.withColumn("campaign_flag",
                functions.when(
                        functions.col("campaign").isNotNull()
                                .and(functions.col("campaign").notEqual("UNKNOWN")), 1
                ).otherwise(0)
        );

        // === SELECT PRODUCT COLUMN ===
        String productCol = null;
        for (String p : Arrays.asList("product_id", "product_id_x", "product_id_y")) {
            if (Arrays.asList(df.columns()).contains(p)) {
                productCol = p;
                break;
            }
        }
        if (productCol == null)
            throw new RuntimeException("Product ID column not found!");

        // === TRANSACTION TABLE (FPM/Association Rules) ===
        Dataset<Row> transactionDf = df
                .select("basket_id", productCol)
                .filter(functions.col("basket_id").isNotNull())
                .filter(functions.col(productCol).isNotNull())
                .dropDuplicates();

        transactionDf.write().mode("overwrite").parquet(ITEM_PATH + "transactions/");

        // === TEMPORAL TABLE (FORECASTING) ===
        if (!Arrays.asList(df.columns()).contains("week_no")) {
            if (!Arrays.asList(df.columns()).contains("day"))
                df = df.withColumn("day", functions.lit(1).cast(DataTypes.IntegerType));

            df = df.withColumn("week_no", functions.ceil(functions.col("day").divide(7)));
        }

        Dataset<Row> temporalDf = df.groupBy("store_id", productCol, "week_no")
                .agg(
                        functions.sum("net_sales").alias("total_net_sales"),
                        functions.sum("quantity").alias("total_quantity"),
                        functions.avg("unit_price").alias("avg_unit_price"),
                        functions.avg("discount_rate").alias("avg_discount_rate"),
                        functions.max("is_discounted").alias("is_discounted"),
                        functions.max("campaign_flag").alias("campaign_flag")
                )
                .withColumnRenamed(productCol, "product_id");

        temporalDf.write().mode("overwrite").partitionBy("week_no").parquet(TEMPORAL_PATH);

        // === CUSTOMER TABLE (RFM + Demographics) ===
        int maxDay = df.agg(functions.max("day")).first().getInt(0);

        Dataset<Row> rfm = df.groupBy("household_key")
                .agg(
                        functions.max("day").alias("last_purchase"),
                        functions.countDistinct("basket_id").alias("frequency"),
                        functions.sum("net_sales").alias("monetary")
                )
                .withColumn("recency", functions.lit(maxDay).minus(functions.col("last_purchase")))
                .drop("last_purchase");

        List<String> demographicCols = new ArrayList<>(Arrays.asList(
                "age_desc", "marital_status_code", "income_desc", "homeowner_desc",
                "hh_comp_desc", "household_size_desc", "kid_category_desc"
        ));
        demographicCols.retainAll(Arrays.asList(df.columns()));

        List<Column> demoAggs = new ArrayList<>();
        for (String c : demographicCols) {
            demoAggs.add(functions.first(c, true).alias(c));
        }

        Dataset<Row> demo = df.groupBy("household_key").agg(
                demoAggs.get(0),
                demoAggs.subList(1, demoAggs.size()).toArray(new Column[0])
        );

        Dataset<Row> customerDf = rfm.join(demo, "household_key")
                                     .na().fill("UNKNOWN", demographicCols.toArray(new String[0]));

        if (Arrays.asList(customerDf.columns()).contains("income_desc"))
            customerDf.write().mode("overwrite").partitionBy("income_desc").parquet(CUSTOMER_PATH);
        else
            customerDf.write().mode("overwrite").parquet(CUSTOMER_PATH);

        // === ITEM TABLE ===
        Dataset<Row> itemDf = df.select(
                "household_key", "basket_id", "day", productCol, "quantity", "sales_value",
                "store_id", "retail_disc", "coupon_disc", "coupon_match_disc", "net_sales",
                "manufacturer", "brand", "department", "commodity_desc", "sub_commodity_desc",
                "display", "mailer", "unit_price", "discount_rate", "is_discounted",
                "campaign_flag", "week_no"
        ).withColumnRenamed(productCol, "product_id");

        if (Arrays.asList(itemDf.columns()).contains("department"))
            itemDf.write().mode("overwrite").partitionBy("department").parquet(ITEM_PATH);
        else
            itemDf.write().mode("overwrite").parquet(ITEM_PATH);

        // === SUMMARY ===
        System.out.println("transaction_table rows: " + transactionDf.count());
        System.out.println("temporal_table rows: " + temporalDf.count());
        System.out.println("customer_table rows: " + customerDf.count());
        System.out.println("item_table rows: " + itemDf.count());

        spark.stop();
        System.out.println("Processing complete. Output saved at: " + BASE_PATH);
    }
}
