# Product Repurchase (Churn) Prediction Model

## 1. Project Overview
This project builds and trains a machine learning model using **Apache Spark (Java)** to predict customer churn.

**Business Question:**  
*If a customer buys a product for the first time, can we predict if they will repurchase that same product within the next 90 days?*

Churn is defined as:  
- **Churn (0):** Customer does *not* repurchase the same product within 90 days  
- **Repurchase (1):** Customer *does* repurchase within 90 days  

---

## 2. Data Used
The model uses three raw CSV files from the `archive/` directory:

### **`transaction_data.csv`**
- Full history of customer transactions  

### **`hh_demographic.csv`**
- Household demographics (Age, Income, Homeownership, etc.)

### **`coupon_redempt.csv`**
- Coupon redemption information by household and date  

---

## 3. How to Run the Model

Everything (ingestion → feature engineering → training → evaluation) runs inside **`ChurnPrediction.java`**.

### **Prerequisites**
- Java 11+
- Apache Maven 3.x
- Apache Spark 3.5.x

---

### **Step 1 — Compile the Project**
From the project root:

```bash
mvn clean package
```

This produces a *fat JAR* in `target/`.

---

### **Step 2 — Run the Spark Job**
```bash
spark-submit   --class com.retail.ml.ChurnPrediction   --master local[*]   --driver-memory 4g   target/retail-sales-bigdata-1.0.0-fat.jar   --   --base=file:/home/user/Documents/archive   --window=90
```

**Flags explained:**
- `--class`: Main class  
- `--base`: Directory containing the CSV files  
- `--window`: Repurchase window (90 days)  

---

## 4. About the Code (`ChurnPrediction.java`)

### **Label Creation**
The model:
1. Identifies each household’s **first purchase** of every product.
2. Checks if the same product is purchased again within the next **90 days**.

Labels:
- `1` → Repurchased  
- `0` → Churned  

---

### **Feature Engineering (Leak-Free)**  
All features use only information known **at the first purchase**, avoiding data leakage.

### **Features**
#### **Transaction-based**
- first_purchase_with_coupon  
- first_purchase_quantity  
- first_purchase_value  
- first_purchase_retail_disc  
- first_purchase_coupon_disc  
- first_day_of_week  

#### **Demographic Features**
- AGE_DESC  
- INCOME_DESC  
- HOMEOWNER_DESC  

---

### **ML Pipeline**
The pipeline includes:
- StringIndexer  
- OneHotEncoder  
- VectorAssembler  
- StandardScaler  
- RandomForestClassifier  

---

## 5. Model Results & Insights

### **Final Model Performance**
After removing leakage:

**Test AUC = 0.560**

- 0.50 → Random guessing  
- 0.56 → Slight predictive power (weak but real)

---

### **Key Business Insight**
A surprising result regarding coupons:

| first_purchase_with_coupon | n_pairs | n_repurchases | repurchase_rate |
|---------------------------|---------|---------------|------------------|
| 0.0 (No Coupon)           | 1,384,588 | 214,268 | **15.5%** |
| 1.0 (Used Coupon)         | 16,881    | 2,387   | **14.1%** |

Customers using coupons on their first purchase were **less likely** to repurchase.

---

## 6. Future Improvements
Better features could increase AUC:

---

## Conclusion
The current churn model establishes a solid baseline and reveals valuable business insights, particularly around coupon behavior. However, stronger and more customer-centric features will be necessary to significantly improve prediction accuracy.
