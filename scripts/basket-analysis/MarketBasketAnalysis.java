// MarketBasketAnalysis.java
// Usage:
//   spark-submit --class MarketBasketAnalysis ... <inputCsv> <hdfsOutputDir> [groupCol] [itemCol] [minSupport] [numPartitions]
//
// Defaults:
//   groupCol = "basket_id"
//   itemCol  = "product_id"
//   minSupport = 0.01
//   numPartitions = 4

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import scala.Tuple2;

import org.apache.spark.mllib.fpm.FPGrowth;
import org.apache.spark.mllib.fpm.FPGrowthModel;

public class MarketBasketAnalysis implements Serializable {
    private static final long serialVersionUID = 1L;

    // Convert Scala Seq/Iterator or Java array-like objects to List<String> safely.
    @SuppressWarnings("unchecked")
    private static List<String> toStringListFromObject(Object obj) {
        List<String> out = new ArrayList<>();
        if (obj == null)
            return out;

        // Try Scala Seq
        try {
            if (obj instanceof scala.collection.Seq) {
                scala.collection.Seq<?> seq = (scala.collection.Seq<?>) obj;
                scala.collection.Iterator<?> it = seq.iterator();
                while (it.hasNext()) {
                    Object v = it.next();
                    if (v != null)
                        out.add(v.toString());
                }
                return out;
            }
        } catch (Throwable t) {
            // fallthrough
        }

        // Java array
        if (obj.getClass().isArray()) {
            Object[] arr = (Object[]) obj;
            for (Object v : arr)
                if (v != null)
                    out.add(v.toString());
            return out;
        }

        // Fallback: split string representation
        String s = obj.toString();
        s = s.replaceAll("^\\[|\\]$", "");
        if (!s.isEmpty()) {
            String[] parts = s.split("\\s*,\\s*");
            out.addAll(Arrays.asList(parts));
        }
        return out;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println(
                    "Usage: MarketBasketAnalysis <inputCsv> <hdfsOutputDir> [groupCol] [itemCol] [minSupport] [numPartitions]");
            System.exit(1);
        }

        final String input = args[0];
        final String outDir = args[1];
        final String groupColArg = (args.length > 2) ? args[2] : "basket_id";
        final String itemColArg = (args.length > 3) ? args[3] : "product_id";
        final double minSupport = (args.length > 4) ? Double.parseDouble(args[4]) : 0.01;
        final int numPartitions = (args.length > 5) ? Integer.parseInt(args[5]) : 4;

        SparkSession spark = SparkSession.builder()
                .appName("MarketBasketAnalysis")
                .getOrCreate();

        JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());

        System.out.println("Reading CSV: " + input);
        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("inferSchema", "false")
                .csv(input);

        // Find actual column names case-insensitively
        String[] cols = df.columns();
        String groupColActual = null;
        String itemColActual = null;
        for (String c : cols) {
            if (c.equalsIgnoreCase(groupColArg))
                groupColActual = c;
            if (c.equalsIgnoreCase(itemColArg))
                itemColActual = c;
        }
        if (groupColActual == null || itemColActual == null) {
            System.err.println("ERROR: required columns not present in CSV.");
            System.err.println("CSV columns: " + Arrays.toString(cols));
            System.err.println("Requested groupCol='" + groupColArg + "' itemCol='" + itemColArg + "'");
            spark.stop();
            System.exit(2);
        }

        System.out.println("Using group column: " + groupColActual + " and item column: " + itemColActual);

        // Select only the two needed columns
        Dataset<Row> pairsDF = df.select(groupColActual, itemColActual);

        // Map rows to (groupId, itemId) pairs
        JavaPairRDD<String, String> pairs = pairsDF.javaRDD().mapToPair(
                new PairFunction<Row, String, String>() {
                    private static final long serialVersionUID = 1L;

                    @Override
                    public Tuple2<String, String> call(Row r) throws Exception {
                        String g = "";
                        String it = "";
                        try {
                            Object og = r.get(0);
                            g = (og == null) ? "" : og.toString().trim();
                        } catch (Exception e) {
                            g = "";
                        }
                        try {
                            Object oi = r.get(1);
                            it = (oi == null) ? "" : oi.toString().trim();
                        } catch (Exception e) {
                            it = "";
                        }
                        return new Tuple2<>(g, it);
                    }
                }).filter(new org.apache.spark.api.java.function.Function<Tuple2<String, String>, Boolean>() {
                    private static final long serialVersionUID = 1L;

                    @Override
                    public Boolean call(Tuple2<String, String> t) {
                        return t != null && t._1 != null && !t._1.isEmpty() && t._2 != null && !t._2.isEmpty();
                    }
                });

        // Group by basket id to get Iterable<String> items, then convert to
        // List<String>
        JavaRDD<List<String>> baskets = pairs.groupByKey(numPartitions)
                .map(new Function<Tuple2<String, Iterable<String>>, List<String>>() {
                    private static final long serialVersionUID = 1L;

                    @Override
                    public List<String> call(Tuple2<String, Iterable<String>> t) throws Exception {
                        List<String> list = new ArrayList<>();
                        for (String s : t._2) {
                            if (s != null && !s.isEmpty())
                                list.add(s);
                        }
                        return list;
                    }
                });

        // Cache baskets for reuse
        baskets = baskets.cache();
        long count = baskets.count();
        System.out.println("Number of baskets: " + count);

        // Run FPGrowth (MLlib)
        FPGrowth fpg = new FPGrowth()
                .setMinSupport(minSupport)
                .setNumPartitions(numPartitions);

        FPGrowthModel<String> model = fpg.run(baskets);

        // Print top frequent itemsets (support, items) - take top 50
        System.out.println("Top frequent itemsets (support, items):");
        List<String> top50 = model.freqItemsets().toJavaRDD()
                .map(new Function<org.apache.spark.mllib.fpm.FPGrowth.FreqItemset<String>, String>() {
                    private static final long serialVersionUID = 1L;

                    @Override
                    public String call(org.apache.spark.mllib.fpm.FPGrowth.FreqItemset<String> fi) throws Exception {
                        Object itemsObj = fi.items();
                        List<String> list = toStringListFromObject(itemsObj);
                        return fi.freq() + "\t" + list.toString();
                    }
                })
                .take(50);

        // Use plain for-loop to print (avoids Consumer mismatch)
        for (String s : top50) {
            System.out.println(s);
        }

        // Save itemsets to HDFS (one line per itemset: freq \t item1,item2,...)
        final String itemsetOut = outDir + "/freq_itemsets";
        model.freqItemsets().toJavaRDD()
                .map(new Function<org.apache.spark.mllib.fpm.FPGrowth.FreqItemset<String>, String>() {
                    private static final long serialVersionUID = 1L;

                    @Override
                    public String call(org.apache.spark.mllib.fpm.FPGrowth.FreqItemset<String> fi) throws Exception {
                        Object itemsObj = fi.items();
                        List<String> list = toStringListFromObject(itemsObj);
                        return fi.freq() + "\t" + String.join(",", list);
                    }
                })
                .saveAsTextFile(itemsetOut);

        System.out.println("Saved frequent itemsets to: " + itemsetOut);

        spark.stop();
    }
}

/*
 * (Use the Below Shell Script to run the Market Basket Analysis)
 * ---------------------------------------------------------------
 * 
 * # Detect fs.defaultFS dynamically (same pattern as before)
 * FS="$(xmllint --xpath "string(//configuration/property[name='fs.defaultFS']/
 * value)" /usr/local/hadoop/etc/hadoop/core-site.xml 2>/dev/null)"
 * FS=${FS:-hdfs://localhost:9000}
 * echo "Using fs.defaultFS = $FS"
 * 
 * ABSJAR="$(pwd)/target/retail-ml-pipeline-1.0-SNAPSHOT.jar"
 * 
 * INPUT=
 * "hdfs://localhost:9000/user/hadoopusr/data/raw/Preprocessed_spark_data_unpacked/transaction_data.csv"
 * OUTPUT="hdfs://localhost:9000/user/hadoopusr/analysis_out/market_basket"
 * 
 * HADOOP_USER_NAME=hadoopusr \
 * spark-submit \
 * --class MarketBasketAnalysis \
 * --master local[*] \
 * --conf spark.hadoop.fs.defaultFS="$FS" \
 * --conf spark.ui.showConsoleProgress=false \
 * "$ABSJAR" \
 * "$INPUT" \
 * "$OUTPUT" \
 * basket_id \
 * product_id \
 * 0.01 \
 * 4
 * -------------------------------------------------------------------
 */