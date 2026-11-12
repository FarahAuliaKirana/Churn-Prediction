"""
src/features/engineer.py
Definisi churn + feature matrix dari master DataFrame.
Ekstrak dari notebook 02_eda.ipynb & 03_feature_engineering.ipynb
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fitur yang WAJIB di-drop untuk mencegah data leakage
# - recency & log_recency: target (is_churned) didefinisikan dari recency > 90
# - first_order_month & first_order_dayofweek: artefak right-censoring (NB04)
LEAKY_FEATURES = [
    "recency",
    "log_recency",
    "first_order_month",
    "first_order_dayofweek",
]

CHURN_WINDOW_DAYS = 90
TOP_N_CATEGORIES = 10


# Kolom yang nilainya per ORDER. Di master DataFrame (grain = item) nilai ini
# berulang di setiap baris item, jadi harus di-dedup per order sebelum diagregasi.
ORDER_LEVEL_COLS = [
    "customer_unique_id",
    "order_id",
    "order_purchase_timestamp",
    "payment_value",
    "payment_installments",
    "payment_type",
    "review_score",
]


def _mode_or_unknown(x: pd.Series) -> str:
    m = x.mode()
    return m.iloc[0] if not m.empty else "unknown"


def order_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Satu baris per order.

    master.csv punya grain item (order dengan 3 item = 3 baris), sementara
    payment_value, payment_installments, payment_type, dan review_score sudah
    diagregasi per order di load.py. Tanpa dedup, sum/count atas kolom-kolom ini
    menghitung order multi-item berkali-kali.
    """
    cols = [c for c in ORDER_LEVEL_COLS if c in df.columns]
    return df[cols].drop_duplicates(subset="order_id")


# ---------------------------------------------------------------------------
# 1. Churn labeling
# ---------------------------------------------------------------------------


def label_churn(df: pd.DataFrame, churn_window_days: int = CHURN_WINDOW_DAYS) -> pd.DataFrame:
    """
    Buat label churn per customer.
    Churn = tidak order dalam `churn_window_days` hari sebelum reference_date.

    Returns DataFrame dengan kolom:
        customer_unique_id, last_order_date, days_since_last_order,
        recency, frequency, monetary, is_churned
    """
    reference_date = df["order_purchase_timestamp"].max()
    logger.info("Reference date: %s", reference_date)

    rfm = (
        order_level(df).groupby("customer_unique_id")
        .agg(
            last_order_date=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            monetary=("payment_value", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (reference_date - rfm["last_order_date"]).dt.days
    rfm["is_churned"] = (rfm["recency"] > churn_window_days).astype(int)

    logger.info(
        "Churn distribution: %s",
        rfm["is_churned"].value_counts(normalize=True).round(3).to_dict(),
    )
    return rfm


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------


def build_features(
    df: pd.DataFrame,
    rfm: pd.DataFrame,
    top_n_categories: int = TOP_N_CATEGORIES,
) -> pd.DataFrame:
    """
    Bangun feature matrix siap modeling dari master df + rfm labels.

    Steps:
        1. Log-transform RFM (skewed distributions)
        2. Agregasi ke level customer (kolom order-level di-dedup per order dulu)
        3. Derived features (business-meaningful)
        4. Merge RFM + aggregasi
        5. Encoding (OHE)
        6. Drop leaky features

    Missing values TIDAK diimputasi di sini: imputasi median dilakukan oleh
    SimpleImputer di dalam model pipeline, yang di-fit hanya pada data train.
    Imputasi di sini (atas seluruh dataset) membocorkan statistik test set.
    """
    # -- Step 1: Log-transform RFM --
    rfm = rfm.copy()
    rfm["log_recency"]   = np.log1p(rfm["recency"])
    rfm["log_monetary"]  = np.log1p(rfm["monetary"])
    rfm["log_frequency"] = np.log1p(rfm["frequency"])

    # -- Step 2: Customer-level aggregation --
    # 2a. Kolom order-level: payment & review (dedup per order)
    order_agg = (
        order_level(df).groupby("customer_unique_id")
        .agg(
            avg_payment_value=("payment_value", "mean"),
            total_payment_value=("payment_value", "sum"),
            avg_installments=("payment_installments", "mean"),
            max_installments=("payment_installments", "max"),
            avg_review_score=("review_score", "mean"),
            min_review_score=("review_score", "min"),
            review_count=("review_score", "count"),
            total_orders=("order_id", "nunique"),
            dominant_payment_type=("payment_type", _mode_or_unknown),
            first_order_date=("order_purchase_timestamp", "min"),
            last_order_date=("order_purchase_timestamp", "max"),
        )
        .reset_index()
    )

    # 2b. Kolom item-level: harga, ongkir, jumlah item, kategori
    item_agg = (
        df.groupby("customer_unique_id")
        .agg(
            avg_price=("price", "mean"),
            total_price=("price", "sum"),
            avg_freight=("freight_value", "mean"),
            # order_item_id adalah nomor urut item dalam order (1, 2, 3, ...),
            # jadi jumlah item = count, bukan sum.
            total_items=("order_item_id", "count"),
            customer_state=("customer_state", "first"),
            dominant_category=("product_category_name_english", _mode_or_unknown),
        )
        .reset_index()
    )

    customer_agg = order_agg.merge(item_agg, on="customer_unique_id", how="left")

    # -- Step 3: Derived features --
    customer_agg["customer_lifetime_days"] = (
        customer_agg["last_order_date"] - customer_agg["first_order_date"]
    ).dt.days.fillna(0)

    customer_agg["avg_order_value"] = (
        customer_agg["total_payment_value"] / customer_agg["total_orders"]
    )

    customer_agg["freight_ratio"] = (
        customer_agg["avg_freight"]
        / customer_agg["avg_payment_value"].replace(0, np.nan)
    ).fillna(0)

    customer_agg["review_gap"] = (
        customer_agg["avg_review_score"] - customer_agg["min_review_score"]
    )

    # Leaky temporal features — dibuat tapi akan di-drop nanti
    first_order = pd.to_datetime(customer_agg["first_order_date"])
    customer_agg["first_order_month"]     = first_order.dt.month
    customer_agg["first_order_dayofweek"] = first_order.dt.dayofweek

    customer_agg = customer_agg.drop(columns=["first_order_date", "last_order_date"])

    # -- Step 4: Merge RFM + aggregasi --
    rfm_cols = [
        "customer_unique_id", "recency", "frequency", "monetary",
        "log_recency", "log_monetary", "log_frequency", "is_churned",
    ]
    features = rfm[rfm_cols].merge(customer_agg, on="customer_unique_id", how="left")

    # -- Step 5: Cap kategori produk top-N, sisanya 'other' --
    top_cats = features["dominant_category"].value_counts().head(top_n_categories).index
    features["dominant_category"] = features["dominant_category"].where(
        features["dominant_category"].isin(top_cats), "other"
    )

    # OHE
    features = pd.get_dummies(
        features,
        columns=["customer_state", "dominant_payment_type", "dominant_category"],
        drop_first=False,
        dtype=int,
    )

    # -- Step 6: Drop leaky features --
    cols_to_drop = [c for c in LEAKY_FEATURES if c in features.columns]
    features = features.drop(columns=cols_to_drop)
    logger.info("Dropped leaky features: %s", cols_to_drop)
    logger.info("Final feature matrix shape: %s", features.shape)

    return features


def get_feature_cols(features: pd.DataFrame) -> list[str]:
    """Return list kolom fitur (exclude id dan target)."""
    exclude = {"customer_unique_id", "is_churned"}
    return [c for c in features.columns if c not in exclude]


def split_X_y(
    features: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split feature matrix jadi X dan y."""
    feature_cols = get_feature_cols(features)
    X = features[feature_cols]
    y = features["is_churned"]
    return X, y