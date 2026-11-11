"""
src/features/snapshot.py
Desain label berbasis snapshot untuk prediksi repeat purchase.

Masalah desain lama (src/features/engineer.py):
    is_churned = recency > 90 hari pada tanggal akhir dataset, sementara fitur
    diambil dari seluruh histori. Label itu identik dengan "order terakhir
    sebelum 19 Juli 2018", sehingga fitur yang berubah seiring waktu (ongkir,
    metode bayar, kategori) ikut membocorkan label. Uji placebo di NB04
    membuktikannya.

Desain snapshot:
    1. Pilih tanggal cutoff T.
    2. Fitur dihitung HANYA dari order dengan purchase timestamp < T.
       Review hanya dipakai jika review_creation_date < T.
    3. Populasi = customer yang punya minimal satu order sebelum T.
    4. Label is_repeat = 1 jika customer membuat order baru di [T, T + horizon).

Karena semua customer dalam satu snapshot dinilai pada tanggal T yang sama,
recency (T - order terakhir) menjadi fitur yang sah — bukan lagi sumber label.
"""
from __future__ import annotations

import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src.features.engineer import _mode_or_unknown, order_level

logger = logging.getLogger(__name__)

HORIZON_DAYS = 90
RECENT_WINDOW_DAYS = 90

NUMERIC_FEATURES = [
    "recency_days",
    "tenure_days",
    "frequency",
    "orders_last_90d",
    "monetary",
    "avg_order_value",
    "avg_installments",
    "max_installments",
    "avg_price",
    "total_price",
    "avg_freight",
    "freight_ratio",
    "total_items",
    "avg_review_score",
    "min_review_score",
    "review_count",
]

CATEGORICAL_FEATURES = [
    "customer_state",
    "dominant_payment_type",
    "dominant_category",
]

TARGET = "is_repeat"


def _review_features(
    orders_hist: pd.DataFrame,
    reviews: Optional[pd.DataFrame],
    cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """
    Fitur review per customer, hanya dari review yang sudah ditulis sebelum cutoff.
    Median jeda purchase → review di Olist adalah 10 hari, jadi tanpa filter ini
    order yang dibeli tepat sebelum T akan membocorkan review dari masa depan.
    """
    if reviews is None:
        return pd.DataFrame({
            "customer_unique_id": pd.Series(dtype=str),
            "avg_review_score": pd.Series(dtype=float),
            "min_review_score": pd.Series(dtype=float),
            "review_count": pd.Series(dtype=float),
        })

    r = reviews[reviews["review_creation_date"] < cutoff]
    r = r.groupby("order_id", as_index=False)["review_score"].mean()
    r = r.merge(orders_hist[["order_id", "customer_unique_id"]], on="order_id", how="inner")
    return (
        r.groupby("customer_unique_id")
        .agg(
            avg_review_score=("review_score", "mean"),
            min_review_score=("review_score", "min"),
            review_count=("review_score", "count"),
        )
        .reset_index()
    )


def build_snapshot(
    master: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    horizon_days: int = HORIZON_DAYS,
    reviews: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Bangun satu snapshot: satu baris per customer yang aktif sebelum `cutoff`.

    Args:
        master:   master DataFrame (grain item) dari src/data/load.py.
        cutoff:   tanggal snapshot T.
        horizon:  panjang jendela label dalam hari.
        reviews:  tabel review mentah (order_id, review_score, review_creation_date).
                  Jika None, fitur review bernilai NaN (diimputasi di pipeline).

    Returns:
        DataFrame dengan kolom customer_unique_id, snapshot_date,
        NUMERIC_FEATURES, CATEGORICAL_FEATURES, is_repeat.
    """
    cutoff = pd.Timestamp(cutoff)
    label_end = cutoff + pd.Timedelta(days=horizon_days)
    ts = master["order_purchase_timestamp"]

    hist = master[ts < cutoff]
    if hist.empty:
        raise ValueError(f"Tidak ada order sebelum cutoff {cutoff.date()}")

    orders_hist = order_level(hist)
    recent_start = cutoff - pd.Timedelta(days=RECENT_WINDOW_DAYS)

    # -- Fitur level order --
    order_agg = (
        orders_hist.assign(_recent=orders_hist["order_purchase_timestamp"] >= recent_start)
        .groupby("customer_unique_id")
        .agg(
            first_order=("order_purchase_timestamp", "min"),
            last_order=("order_purchase_timestamp", "max"),
            frequency=("order_id", "nunique"),
            orders_last_90d=("_recent", "sum"),
            monetary=("payment_value", "sum"),
            avg_order_value=("payment_value", "mean"),
            avg_installments=("payment_installments", "mean"),
            max_installments=("payment_installments", "max"),
            dominant_payment_type=("payment_type", _mode_or_unknown),
        )
        .reset_index()
    )

    # -- Fitur level item --
    item_agg = (
        hist.groupby("customer_unique_id")
        .agg(
            avg_price=("price", "mean"),
            total_price=("price", "sum"),
            avg_freight=("freight_value", "mean"),
            total_items=("order_item_id", "count"),
            customer_state=("customer_state", "first"),
            dominant_category=("product_category_name_english", _mode_or_unknown),
        )
        .reset_index()
    )

    snap = order_agg.merge(item_agg, on="customer_unique_id", how="left")
    snap = snap.merge(_review_features(orders_hist, reviews, cutoff),
                      on="customer_unique_id", how="left")

    # -- Fitur turunan --
    snap["recency_days"] = (cutoff - snap["last_order"]).dt.days
    snap["tenure_days"] = (cutoff - snap["first_order"]).dt.days
    snap["freight_ratio"] = snap["avg_freight"] / snap["avg_order_value"].replace(0, np.nan)
    snap["orders_last_90d"] = snap["orders_last_90d"].astype(int)
    snap["review_count"] = snap["review_count"].fillna(0).astype(int)

    # -- Label: order baru di [cutoff, cutoff + horizon) --
    future = master[(ts >= cutoff) & (ts < label_end)]
    repeaters = set(future["customer_unique_id"].unique())
    snap[TARGET] = snap["customer_unique_id"].isin(repeaters).astype(int)

    snap["snapshot_date"] = cutoff
    for col in CATEGORICAL_FEATURES:
        snap[col] = snap[col].fillna("unknown").astype(str)

    out = snap[["customer_unique_id", "snapshot_date", *NUMERIC_FEATURES,
                *CATEGORICAL_FEATURES, TARGET]]
    logger.info(
        "Snapshot %s: %d customers, repeat rate %.4f",
        cutoff.date(), len(out), out[TARGET].mean(),
    )
    return out


def build_snapshots(
    master: pd.DataFrame,
    cutoffs: Iterable[str | pd.Timestamp],
    horizon_days: int = HORIZON_DAYS,
    reviews: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Gabungkan beberapa snapshot (satu customer bisa muncul di beberapa snapshot)."""
    frames = [build_snapshot(master, c, horizon_days, reviews) for c in cutoffs]
    return pd.concat(frames, ignore_index=True)


def load_reviews(raw_path: str) -> pd.DataFrame:
    """Load tabel review mentah dengan tanggal pembuatan review."""
    from pathlib import Path

    fp = Path(raw_path) / "olist_order_reviews_dataset.csv"
    return pd.read_csv(
        fp,
        usecols=["order_id", "review_score", "review_creation_date"],
        parse_dates=["review_creation_date"],
    )
