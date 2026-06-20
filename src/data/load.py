"""
src/data/load.py
Merge 7 tabel Olist menjadi satu master dataframe.
Ekstrak dari notebook 01_data_loading.ipynb
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_FILES = {
    "customers":   "olist_customers_dataset.csv",
    "orders":      "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments":    "olist_order_payments_dataset.csv",
    "reviews":     "olist_order_reviews_dataset.csv",
    "products":    "olist_products_dataset.csv",
    "sellers":     "olist_sellers_dataset.csv",
    "category":    "product_category_name_translation.csv",
}

DATE_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
    "shipping_limit_date",
]


def load_raw_tables(raw_path: str | Path) -> dict[str, pd.DataFrame]:
    """Load semua CSV mentah ke dict of DataFrames."""
    raw_path = Path(raw_path)
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in RAW_FILES.items():
        fp = raw_path / filename
        if not fp.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {fp}")
        tables[name] = pd.read_csv(fp)
        logger.info("Loaded %s: %s", name, tables[name].shape)
    return tables


def merge_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge bertahap menggunakan LEFT JOIN.
    Urutan: orders → customers → order_items → payments → reviews → products+category
    """
    customers   = tables["customers"]
    orders      = tables["orders"]
    order_items = tables["order_items"]
    payments    = tables["payments"]
    reviews     = tables["reviews"]
    products    = tables["products"]
    category    = tables["category"]

    # Step 1: orders + customers
    df = orders.merge(customers, on="customer_id", how="left")
    logger.info("Step 1 orders+customers: %s", df.shape)

    # Step 2: + order_items
    df = df.merge(order_items, on="order_id", how="left")
    logger.info("Step 2 +order_items: %s", df.shape)

    # Step 3: + payments (aggregated per order)
    payments_agg = (
        payments.groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
            payment_type=("payment_type", "first"),
        )
        .reset_index()
    )
    df = df.merge(payments_agg, on="order_id", how="left")
    logger.info("Step 3 +payments: %s", df.shape)

    # Step 4: + reviews (aggregated per order)
    reviews_agg = (
        reviews.groupby("order_id")
        .agg(review_score=("review_score", "mean"))
        .reset_index()
    )
    df = df.merge(reviews_agg, on="order_id", how="left")
    logger.info("Step 4 +reviews: %s", df.shape)

    # Step 5: + products + category translation
    products_cat = products.merge(category, on="product_category_name", how="left")
    df = df.merge(
        products_cat[["product_id", "product_category_name_english"]],
        on="product_id",
        how="left",
    )
    logger.info("Step 5 +products+category: %s", df.shape)

    # Parse tanggal
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df


def validate_master(df: pd.DataFrame, original_order_count: int) -> None:
    """
    Validasi post-merge: duplikat, order coverage, missing values kritis.
    Raise ValueError jika ada masalah fatal.
    """
    # Duplikat
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        raise ValueError(f"Ditemukan {n_dup} baris duplikat setelah merge.")

    # Order coverage
    orders_in_master = df["order_id"].nunique()
    if orders_in_master != original_order_count:
        raise ValueError(
            f"Order coverage mismatch: expected {original_order_count}, "
            f"got {orders_in_master}"
        )

    # Missing values kritis (>5%)
    key_cols = ["customer_unique_id", "order_id", "order_purchase_timestamp"]
    for col in key_cols:
        pct = df[col].isnull().mean()
        if pct > 0.05:
            raise ValueError(f"Kolom kritis '{col}' punya {pct:.1%} missing values.")

    logger.info("Validasi master OK — shape: %s", df.shape)


def build_master(raw_path: str | Path) -> pd.DataFrame:
    """
    Entry point utama: load → merge → validate → return DataFrame.
    """
    tables = load_raw_tables(raw_path)
    original_order_count = tables["orders"]["order_id"].nunique()
    df = merge_tables(tables)
    validate_master(df, original_order_count)
    return df