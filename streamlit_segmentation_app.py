import streamlit as st
import subprocess

# Correct JAR path
JAR_PATH = "target/retail-bigdata-1.0.jar"

st.title("Retail Segmentation Prediction ")

mode = st.radio("Select Prediction Mode", ["Predict by household_key", "Manual Input"])

# -----------------------------------------------------------
# MODE 1 — household_key prediction using Spark-submit
# -----------------------------------------------------------
if mode == "Predict by household_key":

    key = st.number_input("Enter household_key", step=1)

    if st.button("Predict Cluster", key="predict_key_mode1"):
        st.write("Running prediction...")
        
        cmd = [
            "spark-submit",
            "--class", "com.retail.ml.SegmentationPrediction",
            JAR_PATH,
            "household_key", str(int(key))
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="."
        )

        # JUST SHOW STDOUT (no logs)
        st.code(result.stdout)


# -----------------------------------------------------------
# MODE 2 — Manual input prediction using Spark-submit
# -----------------------------------------------------------
else:
    total_spend = st.number_input("Total Spend", step=1.0)
    total_qty = st.number_input("Total Quantity", step=1.0)
    num_visits = st.number_input("Number of Visits", step=1.0)
    unique_products = st.number_input("Unique Products", step=1.0)
    unique_brands = st.number_input("Unique Brands", step=1.0)
    avg_basket_value = st.number_input("Avg Basket Value", step=1.0)

    if st.button("Predict Cluster", key="predict_key_mode2"):
        st.write("Running prediction...")
        cmd = [
            "spark-submit",
            "--class", "com.retail.ml.SegmentationPrediction",
            JAR_PATH,
            "manual",
            str(total_spend), str(total_qty), str(num_visits),
            str(unique_products), str(unique_brands), str(avg_basket_value)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="."
        )

        # JUST SHOW STDOUT (no logs)
        st.code(result.stdout)
