# Customer Segmentation using RFM, CLV & Clustering

This project turns one year of ecommerce transactions into **actionable customer segments** and **forward-looking value (CLV) estimates**.

It is structured like a real analytics project:

- Clean raw transaction logs
- Build customer-level features (RFM + product mix + geography)
- Cluster customers using multiple algorithms
- Estimate 6-month Customer Lifetime Value (CLV)
- Explore segments through an interactive Streamlit dashboard

---

## 1. Business Problem

The retailer is treating every customer the same:

- Same campaigns to high-value and low-value customers  
- No clear way to identify **VIPs**, **growth customers**, or **at-risk segments**  
- Marketing spend is not targeted or optimized  

**Goal:**  
Use historical purchase behaviour to:

- Segment customers into meaningful behavioral groups  
- Estimate their **future value** (CLV)  
- Provide a practical view for marketing and CRM teams to decide **who to target, with what, and when**  

---

## 2. Data

- **Source:** Online Retail transactional dataset (typical ecommerce data)
- **File:** `data/raw/ecommerce_data.csv`  
- **Unit of analysis (raw):** each row is a line item on an invoice  

**Key columns used:**

- `InvoiceNo` – invoice ID  
- `StockCode` – product code  
- `Description` – product description  
- `Quantity` – units purchased (can be negative for returns in raw)  
- `InvoiceDate` – purchase date and time  
- `UnitPrice` – price per unit  
- `CustomerID` – unique customer identifier  
- `Country` – customer country  

The analysis aggregates this to **one row per customer**.

---

## 3. Features & Methodology

### 3.1 Data cleaning

Implemented in `src/data_prep.py`:

- Drop rows with missing `CustomerID`  
- Remove negative or zero `Quantity` and `UnitPrice`  
- Parse `InvoiceDate` as datetime  
- Normalize `Country` values (trim spaces, treat values like `"Unspecified"` as missing)  
- Compute `TotalPrice = Quantity * UnitPrice`  

### 3.2 Customer-level features (RFM + extras)

Implemented in `src/rfm_features.py`:

For each `CustomerID`:

**RFM:**

- `Recency` – days since last purchase (relative to dataset end)  
- `Frequency` – number of unique invoices  
- `Monetary` – total spend (`sum(TotalPrice)`)  

**Product mix (simple NLP on description keywords):**

- Map `Description` into coarse categories (e.g., Bags, Kitchen, HomeDecor, Toys, Other)  
- Compute **share of revenue per category**:

  - `CatShare_Bags`  
  - `CatShare_Kitchen`  
  - `CatShare_HomeDecor`  
  - `CatShare_Toys`  
  - `CatShare_Other`  

**Geography:**

- `PrimaryCountry` – country where the customer spent the most  
- `IsUK` – `1` if `PrimaryCountry == "United Kingdom"`, else `0`  
- Customers with no valid country are tagged as `PrimaryCountry = "Unknown"`  

All of this is combined into a single **feature matrix** in `build_feature_matrix()`.

---

## 4. Modelling

### 4.1 CLV (Customer Lifetime Value)

Implemented in `src/clv.py` using the `lifetimes` library:

1. Convert transactions into **lifetimes summary data** per customer:  
   - `frequency` – number of repeat purchases  
   - `recency` – time between first and last purchase  
   - `T` – age of the customer in the dataset  
   - `monetary_value` – average order value  

2. Fit two probabilistic models:

   - **BG/NBD (Beta-Geo Fitter)** – models the probability of future purchases  
   - **Gamma-Gamma Fitter** – models average spend per purchase  

3. Combine them to estimate:

   - `CLV_6m` – predicted customer lifetime value over the next **6 months**

This produces a **forward-looking** view of value, not just historical spend.

### 4.2 Clustering

Implemented in `src/clustering_models.py`:

1. **Feature selection for clustering**

   - Core behavioural features: `Recency`, `Frequency`, `Monetary`, `IsUK`  
   - Product mix features: all `CatShare_*` columns  

2. **Preprocessing**

   - Drop customers with missing core RFM  
   - Fill missing `IsUK` and category shares with `0`  
   - Standardize features using `StandardScaler`  

3. **Algorithms**

   - **K-Means**  
   - **Hierarchical Clustering (Agglomerative)**  
   - **Gaussian Mixture Models (GMM)**  

4. **Model evaluation (for K-Means)**

   - Evaluate `k` in a range (e.g., 2–8) using:
     - **Silhouette score**
     - **Inertia (within-cluster sum of squares)**  

5. **Outputs**

   - `Cluster_KMeans`  
   - `Cluster_Hierarchical`  
   - `Cluster_GMM`  

All cluster labels and CLV values are merged into a single table:

> `data/processed/rfm_segments.csv`

Each row is now a **fully profiled customer**.

---

## 5. Dashboard

Implemented in `dashboard/app.py` using **Streamlit** and **Altair**.

### 5.1 Main capabilities

- **Sidebar controls**
  - Choose cluster method (`Cluster_KMeans`, `Cluster_Hierarchical`, `Cluster_GMM`)  
  - Filter by `PrimaryCountry`  
  - Set minimum `CLV_6m`  
  - Set minimum `Frequency`  
  - Optionally show the raw filtered table  

- **KPI cards**
  - Customers (filtered)  
  - Total revenue (filtered)  
  - Average CLV (6 months, filtered)  

- **Segment summary table**
  - Customers per segment  
  - Total revenue  
  - Average recency, frequency  
  - Average CLV  

- **Auto-generated insights**
  - Segment contributing the highest share of revenue  
  - Segment with highest average CLV  
  - Size of filtered customer base and number of segments  

### 5.2 Visualizations

- **Recency vs Frequency by segment** (scatter)  
- **Monetary vs CLV (6m) by segment** (scatter)  
- **CLV distribution by segment** (boxplot)  
- **Revenue by country** (bar chart, top countries only)  

The dashboard is intended as a **mock internal tool** that marketing, CRM, or growth teams could use to explore and prioritize segments.

---

## 6. Project Structure

```text
CUSTOMER-SEGMENTATION-USING-RFM-ANALYSIS/
├── src/
│   ├── data_prep.py          
│   ├── rfm_features.py       
│   ├── clustering_models.py  
│   └── clv.py                
│
├── dashboard/
│   └── app.py                
│
├── notebooks/
│   └── CustomerSegmentationAnalysis.ipynb  
│
├── data/
│   ├── raw/
│   │   └── ecommerce_data.csv             
│   └── processed/
│       └── rfm_segments.csv               
│
├── reports/
│   └── figures/                           
│
├── assets/
│   └── banner.png                         
│
├── run_pipeline.py                        
├── requirements.txt
├── .gitignore
└── README.md
