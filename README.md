# 🚀 Customer Value Intelligence System

### *RFM Segmentation • CLV Forecasting • Behavioral Clustering • Interactive Dashboard*

This project transforms raw ecommerce transactions into **actionable customer intelligence**.
It identifies high-value segments, forecasts customer lifetime value (CLV), and provides a dashboard that marketing, CRM, and growth teams can use to make **data-driven retention and targeting decisions**.

The structure mirrors how customer analytics is done at companies like Amazon, Flipkart, Meesho, Walmart, Shopify, and Nykaa.

---

## ⭐ Overview

This end-to-end system performs:

* Data cleaning & preprocessing
* Customer-level feature engineering (RFM, product mix, geography)
* Probabilistic CLV forecasting (BG/NBD + Gamma-Gamma)
* Multi-model clustering (K-Means, Hierarchical, GMM)
* Interactive dashboard for insights (Streamlit + Altair)

The output is a production-style **Customer Value Engine** that powers segmentation, targeting, and retention strategies.

---

## 🔥 1. Business Problem

The retailer currently treats all customers the same:

* No differentiation between high-value and low-value buyers
* No way to identify *VIPs*, *at-risk customers*, or *growth segments*
* Marketing spend is not optimized
* Personalization efforts are limited

**Goal:**
Use customer purchase behavior to create meaningful segments and predict future value so teams can decide:

> **Who to target, what message to send, and how to increase retention and revenue.**

---

## 📦 2. Dataset

* **Source:** Online Retail Dataset (one year of ecommerce transactions)
* **File:** `data/raw/ecommerce_data.csv`

Each row is a single line item on an invoice.
All results are aggregated to one row per **CustomerID**.

**Key columns:**

* InvoiceNo, StockCode, Description
* Quantity, UnitPrice, InvoiceDate
* CustomerID, Country

---

## 🔧 3. Feature Engineering

### 🧹 Data Cleaning

* Remove missing `CustomerID`
* Remove negative quantity/price
* Convert timestamps
* Normalize country names (remove "Unspecified")
* Compute revenue per line (`TotalPrice`)

### 📊 Customer-Level Features

#### **RFM Metrics**

* **Recency** — days since last purchase
* **Frequency** — number of invoices
* **Monetary** — total spend

#### **Product Category Affinity**

Keyword-based category mapping:

* Home Decor
* Kitchen
* Bags
* Toys
* Other

Converted into revenue shares:
`CatShare_HomeDecor`, `CatShare_Kitchen`, etc.

#### **Geographic Features**

* `PrimaryCountry`
* `IsUK` (flag for key region)
* Customers with invalid countries marked as `"Unknown"`

This creates a comprehensive customer behavioral profile.

---

## 🤖 4. Modeling

### 4.1 Customer Lifetime Value (CLV)

Using the `lifetimes` package:

1. Convert transactions to **summary lifetimes format**
2. Fit:

   * **BG/NBD** for purchase probability
   * **Gamma-Gamma** for monetary value
3. Combine to estimate:

**`CLV_6m` — expected customer value over the next 6 months**

This gives a forward-looking view of customer potential.

---

### 4.2 Customer Segmentation

Three unsupervised ML approaches:

* **K-Means**
* **Agglomerative Hierarchical Clustering**
* **Gaussian Mixture Models (GMM)**

Features used:

* Recency
* Frequency
* Monetary
* IsUK
* Category revenue shares

Evaluation using silhouette score and business interpretability.

Final labels saved as:

* `Cluster_KMeans`
* `Cluster_Hierarchical`
* `Cluster_GMM`

Results saved to:

```
data/processed/rfm_segments.csv
```

---

## 📊 5. Dashboard (Streamlit)

The interactive dashboard lets users explore:

### ⚙️ Filters

* Cluster method
* Country
* Minimum frequency
* Minimum CLV
* Optional raw table view

### 📌 KPI Cards

* Customer count (filtered)
* Total revenue
* Average CLV

### 📈 Visualizations

* Recency vs Frequency (by segment)
* Monetary vs CLV
* CLV distribution per segment
* Revenue by country
* Auto-generated insights

This simulates a real internal “Customer Insights Portal”.

---

## 🧠 6. Key Insights from the Analysis

These insights come from the actual dataset:

### ⭐ 1. A small VIP segment drives **~50% of total revenue**

These customers:

* Buy frequently
* Make higher-value purchases
* Have low recency (recent activity)
* Show strong affinity to Home Decor & Kitchen categories
* Have the highest predicted CLV

### ⭐ 2. Most customers buy rarely

* Over **75%** purchase only once or twice
* Heavy long-tail distribution
* Need nurture flows and first-purchase incentives

### ⭐ 3. UK dominates geographic distribution

* UK customers produce a significant majority of revenue
* Cross-border customers contribute niche value

### ⭐ 4. CLV reveals hidden potential

Several low-spend customers:

* Have high probability of returning
* Are projected to generate strong CLV
* Represent “emerging loyalists”

### ⭐ 5. Segment Strategies

* **VIP:** early access, loyalty rewards, personalized outreach
* **Growth:** cross-sell, bundle offers, frequency nudges
* **At-risk:** reactivation messaging + discounts
* **Dormant:** email sequences + remarketing

---

## 🏗 7. Project Structure

```text
CUSTOMER-VALUE-INTELLIGENCE/
├── src/
│   ├── data_prep.py
│   ├── rfm_features.py
│   ├── clustering_models.py
│   └── clv.py
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── raw/ecommerce_data.csv
│   └── processed/rfm_segments.csv
│
├── run_pipeline.py
├── requirements.txt
├── README.md
└── notebooks/
```

---

---

## 8. Segment Playbooks (How a Marketing Team Would Use This)

To make the output usable for non-technical teams, each cluster can be mapped to a simple, business-friendly segment.

Below is an example mapping based on typical RFM + CLV patterns:

### 1. VIP Loyalists

**Who they are**

- Very recent purchases (low Recency)
- High Frequency and high Monetary value
- Highest CLV_6m among all segments

**Signals**

- `Recency` in the lowest 20–30%  
- `Frequency` and `Monetary` in the top 20%  
- Above-average `CLV_6m`  

**What to do**

- Early access to new launches
- Priority support and concierge-style service
- Exclusive loyalty rewards and birthday offers

**What to measure**

- Retention rate (90-day / 180-day)
- CLV uplift vs baseline
- NPS / satisfaction

---

### 2. High-Potential Newcomers

**Who they are**

- Recently acquired customers
- 1–2 purchases so far, but strong early signals
- Growing CLV, even if historical spend is moderate

**Signals**

- Medium `Recency` (recent first purchase)
- Low to medium `Frequency`, but positive `CLV_6m`
- High predicted purchase probability from BG/NBD

**What to do**

- Welcome journeys and onboarding emails
- “Complete the look” or “People also bought” recommendations
- Small, targeted incentives for 2nd and 3rd purchase

**What to measure**

- 2nd purchase conversion rate
- Time between 1st and 2nd order
- CLV growth over 3–6 months

---

### 3. Discount-Driven Shoppers

**Who they are**

- Purchase mostly during promotions or clearances
- Lower Monetary value but responsive to discounts
- Higher sensitivity to pricing and offers

**Signals**

- Low to medium `Monetary`
- Spike in orders around promo periods (can be tagged via campaign metadata)
- CLV_6m is positive but heavily offer-driven

**What to do**

- Campaigns tied to sales, seasonal promos, flash deals
- Volume-driven offers (BOGO, bundles)
- Limited personalization, high automation

**What to measure**

- Margin after discount
- Offer redemption rates
- Incremental orders during promo periods

---

### 4. At-Risk Loyalists

**Who they are**

- Historically strong customers who have gone quiet
- High past Monetary and Frequency, but Recency has increased
- CLV_6m starting to decline

**Signals**

- Historically high `Monetary`, `Frequency`
- Recency drifting into higher range (haven’t purchased in a while)
- CLV_6m lower than VIPs, but still meaningful

**What to do**

- Win-back campaigns: “We miss you” + personalized recommendations
- Small loyalty incentives to re-activate (credit, free shipping)
- Reminders for replenishable items (if applicable)

**What to measure**

- Re-activation rate
- Time to next purchase after win-back
- CLV change post reactivation

---

### 5. Dormant / One-and-Done Customers

**Who they are**

- Bought once and never came back
- Make up the bulk of the customer base
- Very low CLV

**Signals**

- `Frequency = 1`
- High `Recency` (no recent purchases)
- Low `CLV_6m`

**What to do**

- Light-touch reactivation (email, push) rather than heavy spend
- Generic awareness campaigns
- Use as a control group for testing segment-based strategies

**What to measure**

- Cost per reactivated customer
- Overall list health and engagement


## ⚙️ 9. How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the pipeline

```bash
python run_pipeline.py
```

### Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

