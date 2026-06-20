"""
src/api/schemas.py
Pydantic models untuk FastAPI request & response.

Changelog v1.2.0:
- DriftStats diperluas: tambah drift_flags dan flagged_features
  untuk support endpoint /monitoring/drift yang sekarang aktif.
- log_monetary dan log_frequency tidak diterima dari client —
  dihitung server-side di main.py dari monetary/frequency.
- HealthResponse sekarang expose threshold aktif (dibaca dari metrics.json).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ChurnRequest(BaseModel):
    """
    Churn prediction payload untuk satu customer.
    Semua nilai adalah agregasi dari history order customer.

    Note: log_monetary dan log_frequency sengaja tidak ada di sini —
    API menghitungnya server-side dari monetary dan frequency untuk
    menjamin konsistensi (mencegah client kirim nilai yang kontradiktif).
    """
    customer_id: Optional[str] = Field(None, description="Customer ID untuk tracking")

    # Payment features
    avg_payment_value:   float = Field(..., ge=0, description="Rata-rata nilai pembayaran per order (BRL)")
    total_payment_value: float = Field(..., ge=0, description="Total nilai pembayaran semua order (BRL)")
    avg_installments:    float = Field(..., ge=1, description="Rata-rata jumlah cicilan")
    max_installments:    int   = Field(..., ge=1, description="Cicilan terbanyak dalam satu order")

    # Price & freight features
    avg_price:     float = Field(..., ge=0, description="Rata-rata harga produk (BRL)")
    total_price:   float = Field(..., ge=0, description="Total harga produk (BRL)")
    avg_freight:   float = Field(..., ge=0, description="Rata-rata ongkos kirim (BRL)")
    freight_ratio: float = Field(..., ge=0, le=2, description="Ongkir sebagai proporsi dari avg payment")

    # Review features
    avg_review_score: float = Field(..., ge=1, le=5, description="Rata-rata skor review (1–5)")
    min_review_score: float = Field(..., ge=1, le=5, description="Skor review terendah")
    review_gap:       float = Field(..., ge=0, description="avg_review - min_review (proxy konsistensi)")
    review_count:     int   = Field(..., ge=0, description="Jumlah review yang diberikan")

    # Order behaviour
    total_items:     int   = Field(..., ge=1, description="Total item yang dibeli")
    total_orders:    int   = Field(..., ge=1, description="Total order yang dilakukan")
    avg_order_value: float = Field(..., ge=0, description="Rata-rata nilai per order (BRL)")

    # Temporal / derived
    frequency:              int   = Field(..., ge=1, description="Jumlah order unik")
    monetary:               float = Field(..., ge=0, description="Total pembayaran (BRL)")
    customer_lifetime_days: int   = Field(..., ge=0, description="Hari dari order pertama ke terakhir")

    # Categorical (API handle OHE secara internal)
    customer_state:        str = Field(..., description="Kode state (e.g. 'SP', 'RJ')")
    dominant_payment_type: str = Field(..., description="Metode pembayaran dominan")
    dominant_category:     str = Field(..., description="Kategori produk dominan")

    @model_validator(mode="after")
    def check_review_gap(self) -> "ChurnRequest":
        """review_gap harus = avg - min (toleransi floating point)."""
        expected = round(self.avg_review_score - self.min_review_score, 4)
        if abs(self.review_gap - expected) > 0.1:
            raise ValueError(
                f"review_gap ({self.review_gap}) tidak sesuai dengan "
                f"avg_review_score - min_review_score ({expected})"
            )
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "customer_id": "cust_001",
            "avg_payment_value": 150.0,
            "total_payment_value": 150.0,
            "avg_installments": 3.0,
            "max_installments": 3,
            "avg_price": 120.0,
            "total_price": 120.0,
            "avg_freight": 30.0,
            "freight_ratio": 0.20,
            "avg_review_score": 4.0,
            "min_review_score": 4.0,
            "review_gap": 0.0,
            "review_count": 1,
            "total_items": 1,
            "total_orders": 1,
            "avg_order_value": 150.0,
            "frequency": 1,
            "monetary": 150.0,
            "customer_lifetime_days": 0,
            "customer_state": "SP",
            "dominant_payment_type": "credit_card",
            "dominant_category": "health_beauty"
        }
    }}


class ChurnResponse(BaseModel):
    """Hasil prediksi churn untuk satu customer."""
    customer_id:       Optional[str] = None
    churn_probability: float         = Field(..., ge=0, le=1)
    is_churn:          bool
    threshold_used:    float
    model_version:     str


class BatchChurnRequest(BaseModel):
    """Batch prediction untuk banyak customer sekaligus."""
    customers: list[ChurnRequest] = Field(..., min_length=1, max_length=1000)


class BatchChurnResponse(BaseModel):
    """Hasil batch prediction."""
    predictions: list[ChurnResponse]
    total:       int
    churn_count: int
    churn_rate:  float


class HealthResponse(BaseModel):
    """Response dari /health endpoint."""
    status:        Literal["ok", "degraded"]
    model_loaded:  bool
    model_version: Optional[str]
    threshold:     Optional[float] = None  # threshold aktif dari metrics.json


class DriftStats(BaseModel):
    """
    Summary drift stats dari /monitoring/drift.

    drift_flags: per-fitur dict dengan train_mean, recent_mean, pct_change, flagged.
    flagged_features: list fitur yang drift > 20% dari baseline training.
    """
    window_hours:      int
    request_count:     int
    feature_means:     dict[str, float]
    churn_rate_recent: Optional[float]
    drift_flags:       Optional[dict[str, Any]] = None
    flagged_features:  list[str] = Field(default_factory=list)