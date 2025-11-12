import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import java.util.Arrays;
import java.util.List;

public class FullPreprocessingPipeline {

    public static void main(String[] args) {
        // --- 1. Configuration (ADJUSTED FOR LOCAL DVC WORKFLOW) ---
        
        // Input: HDFS (assuming raw CSVs are staged here for processing)
        final String INPUT_HDFS_PATH = "hdfs:///user/hadoop/data/raw/*.csv";

        // Output: LOCAL FILESYSTEM PATH (This is the path DVC will track)
        final String LOCAL_BASE_PATH = "data/processed/";
        final String LOCAL_ITEM_PATH = LOCAL_BASE_PATH + "item_transactions/";
        final String LOCAL_CUSTOMER_PATH = LOCAL_BASE_PATH + "customer_segments/";
        final String LOCAL_TEMPORAL_PATH = LOCAL_BASE_PATH + "campaign_events/";

        // --- 2. Initialize Spark Session ---
        // Using getOrCreate() allows running locally or on a cluster
        SparkSession spark = SparkSession
                .builder()
                .appName("RetailDataPreprocessingPipeline")
                .getOrCreate();

        // Standardized list of columns (kept for reference, but not used in the read logic)
        List<String> rawColumns = Arrays.asList(
            "household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE",
            "STORE_ID", "RETAIL_DISC", "TRANS_TIME", "WEEK_NO", "COUPON_DISC", "COUPON_MATCH_DISC",
            "MANUFACTURER", "DEPARTMENT", "BRAND", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "CURR_SIZE_OF_PRODUCT",
            "AGE_DESC", "MARITAL_STATUS_CODE", "INCOME_DESC", "HOMEOWNER_DESC", "HH_COMP_DESC", 
            "HOUSEHOLD_SIZE_DESC", "KID_CATEGORY_DESC", "display", "mailer",
            "COUPON_UPC", "CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"
        );

        // --- 3. Read, Standardize, and Clean Raw Data from HDFS ---
        System.out.println("Reading and standardizing raw CSV files from HDFS...");

        Dataset<Row> rawDf = spark.read()
                .option("header", "true")
                .option("inferSchema", "true") 
                .csv(INPUT_HDFS_PATH);

        Dataset<Row> cleanedDf = rawDf;
        
        // Standardize Column Names
        for (String col : rawDf.columns()) {
            String newCol = col.toLowerCase().replace(" ", "_");
            cleanedDf = cleanedDf.withColumnRenamed(col, newCol);
        }
        
        // Data Cleaning and Feature Engineering
        cleanedDf = cleanedDf
            // Impute missing numericals with 0.0
            .na().fill(0.0, new String[]{"sales_value", "quantity", "retail_disc", "coupon_disc", "coupon_match_disc"})
            // Ensure Sales and Quantity are non-negative
            .filter(functions.col("sales_value").$greater$eq(0.0))
            .filter(functions.col("quantity").$greater$eq(0))
            // Impute missing categoricals with 'UNKNOWN'
            .na().fill("UNKNOWN", new String[]{"manufacturer", "department", "brand", "commodity_desc", "age_desc", "income_desc"})
            // Calculate Net Sales
            .withColumn("net_sales", 
                functions.col("sales_value").minus(functions.col("retail_disc"))
                         .minus(functions.col("coupon_disc")).minus(functions.col("coupon_match_disc"))
            );
        
        // --- 4. Divide Data into Three Denormalized Tables and Write to LOCAL DISK ---

        // -----------------------------------------------------------
        // 4a. 🛒 Transaction / Item-Level Table
        // -----------------------------------------------------------
        System.out.println("Writing Item-Level Data to Local Disk for DVC tracking...");
        
        Dataset<Row> itemDf = cleanedDf.select(
            "household_key", "basket_id", "day", "product_id", "quantity", "sales_value", 
            "store_id", "retail_disc", "coupon_disc", "coupon_match_disc", "net_sales",
            "manufacturer", "brand", "department", "commodity_desc", "sub_commodity_desc",
            "display", "mailer"
        );
        
        // Write to local disk, partitioned by DEPARTMENT
        itemDf.write()
            .mode("overwrite")
            .partitionBy("department")
            .parquet(LOCAL_ITEM_PATH);
            
        System.out.println("✅ Item-Level Data saved locally to: " + LOCAL_ITEM_PATH);


        // -----------------------------------------------------------
        // 4b. 🧑‍🤝‍🧑 Customer / Household-Level Table
        // -----------------------------------------------------------
        System.out.println("Writing Customer-Level Data to Local Disk for DVC tracking...");

        Dataset<Row> customerDf = cleanedDf.select(
            "household_key", "age_desc", "marital_status_code", "income_desc", 
            "homeowner_desc", "hh_comp_desc", "household_size_desc", "kid_category_desc"
        ).dropDuplicates("household_key");

        // Write to local disk, partitioned by INCOME_DESC
        customerDf.write()
            .mode("overwrite")
            .partitionBy("income_desc")
            .parquet(LOCAL_CUSTOMER_PATH);
            
        System.out.println("✅ Customer-Level Data saved locally to: " + LOCAL_CUSTOMER_PATH);


        // -----------------------------------------------------------
        // 4c. ⏱️ Temporal / Campaign-Level Table
        // -----------------------------------------------------------
        System.out.println("Writing Temporal/Campaign Data to Local Disk for DVC tracking...");

        Dataset<Row> temporalDf = cleanedDf.select(
            "week_no", "day", "trans_time", "store_id", 
            "coupon_upc", "campaign", "description", "start_day", "end_day"
        ).dropDuplicates("week_no", "campaign");

        // Write to local disk, partitioned by WEEK_NO
        temporalDf.write()
            .mode("overwrite")
            .partitionBy("week_no")
            .parquet(LOCAL_TEMPORAL_PATH);
            
        System.out.println("✅ Temporal/Campaign Data saved locally to: " + LOCAL_TEMPORAL_PATH);

        spark.stop();
    }
}