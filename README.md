# Churn Prediction - Olist Brazilian E-Commerce

> Predicting customers who are unlikely to place another order within the next 90 days, enabling early retention actions.

---

## Problem Statement

Most e-commerce platforms struggle with customer retention. This project builds a machine learning model to identify at-risk customers early, so business teams can take targeted retention actions before losing them.

**Churn Definition:** A customer who has not placed an order within 90 days after their last purchase.

---

## Dataset

[Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 100k+ orders from 2016–2018, structured across 9 relational tables.

| Table | Description |
|---|---|
| olist_orders | Main order data |
| olist_order_items | Items per order |
| olist_order_payments | Payment method & value |
| olist_order_reviews | Customer reviews & ratings |
| olist_customers | Customer data |
| olist_products | Product data |
| olist_sellers | Seller data |
| olist_geolocation | Geolocation data (zip code coordinates) |
| product_category_name_translation | Product category name (PT → EN) |

---

## Methodology

1. **Data Loading & Merging** — Load 9 CSV tables, merge into `master.csv`
2. **Churn Labeling** — Define churn as no order within 90 days → `is_churned`
3. **Exploratory Data Analysis** — RFM analysis, distributions, trends, correlations
4. **Feature Engineering** — Build meaningful features for modeling
5. **Modeling** — Train Random Forest & XGBoost, handle class imbalance
6. **Evaluation** — Optimize for ROC-AUC and F1-Score

---
**Key Findings from EDA:**
- Churn rate: **89.9%** — severe class imbalance
- **96.9%** of customers made only 1 purchase (one-time buyers)
- **Recency** is the only feature significantly correlated with churn (r = 0.47)
- Review score and monetary value show almost no difference between churned vs non-churned customers

---

## Tech Stack

- **Python** — Pandas, NumPy
- **Visualization** — Matplotlib, Seaborn
- **Modeling** — Scikit-learn, XGBoost
- **Environment** — Jupyter Notebook, VS Code

---

## Project Structure

```
churn-prediction-olist/
├── data/
│   ├── raw/          # Original dataset (not pushed to GitHub)
│   └── processed/    # Output after cleaning & feature engineering
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_final_report.ipynb
└── README.md
```

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/FarahAuliaKirana/churn-prediction-olist.git
cd churn-prediction-olist

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

# 3. Download the dataset
# Go to https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Download all CSV files into data/raw/

# 4. Run notebooks in order
# 01_data_loading.ipynb → 02_eda.ipynb → and so on
```

---

## Progress

- [x] Data loading & merging
- [x] Churn definition (90-day window)
- [x] RFM analysis
- [x] EDA — distributions, trends, correlations, category analysis
- [ ] Feature engineering
- [ ] Modeling
- [ ] Final report

**Status:** 🔄 In Progress — Week 1 Complete ✅ | Week 2 Starting

