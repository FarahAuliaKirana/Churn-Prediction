"""
src/api/schemas.py
Pydantic models untuk FastAPI request & response.

Changelog v2.0.0:
- Tambah schema v2 (repeat purchase, snapshot design): RepeatRequest,
  RepeatResponse, BatchRepeatRequest, BatchRepeatResponse.
- HealthResponse expose status model v2.

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
    customer_id: Optional[str] = Field(None, description="Customer ID for tracking")

    # Payment features
    avg_payment_value:   float = Field(..., ge=0, description="Average payment per order (BRL)")
    total_payment_value: float = Field(..., ge=0, description="Total payment across all orders (BRL)")
    avg_installments:    float = Field(..., ge=1, description="Average number of installments")
    max_installments:    int   = Field(..., ge=1, description="Most installments in a single order")

    # Price & freight features
    avg_price:     float = Field(..., ge=0, description="Average product price (BRL)")
    total_price:   float = Field(..., ge=0, description="Total product price (BRL)")
    avg_freight:   float = Field(..., ge=0, description="Average shipping cost (BRL)")
    freight_ratio: float = Field(..., ge=0, le=2, description="Shipping cost as a share of the average payment")

    # Review features
    avg_review_score: float = Field(..., ge=1, le=5, description="Average review score (1–5)")
    min_review_score: float = Field(..., ge=1, le=5, description="Lowest review score")
    review_gap:       float = Field(..., ge=0, description="avg_review - min_review (consistency proxy)")
    review_count:     int   = Field(..., ge=0, description="Number of reviews given")

    # Order behaviour
    total_items:     int   = Field(..., ge=1, description="Total items purchased")
    total_orders:    int   = Field(..., ge=1, description="Total orders placed")
    avg_order_value: float = Field(..., ge=0, description="Average order value (BRL)")

    # Temporal / derived
    frequency:              int   = Field(..., ge=1, description="Number of distinct orders")
    monetary:               float = Field(..., ge=0, description="Total payment (BRL)")
    customer_lifetime_days: int   = Field(..., ge=0, description="Days from first to last order")

    # Categorical (API handle OHE secara internal)
    customer_state:        str = Field(..., description="State code (e.g. 'SP', 'RJ')")
    dominant_payment_type: str = Field(..., description="Most used payment method")
    dominant_category:     str = Field(..., description="Most purchased product category")

    @model_validator(mode="after")
    def check_review_gap(self) -> "ChurnRequest":
        """review_gap harus = avg - min (toleransi floating point)."""
        expected = round(self.avg_review_score - self.min_review_score, 4)
        if abs(self.review_gap - expected) > 0.1:
            raise ValueError(
                f"review_gap ({self.review_gap}) does not match "
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
    repeat_model_loaded:  bool = False     # model v2 (repeat purchase)
    repeat_model_version: Optional[str] = None


# ---------------------------------------------------------------------------
# v2 — Repeat purchase (snapshot design, RFM Logistic Regression)
# ---------------------------------------------------------------------------

RepeatSegment = Literal["top1pct", "top5pct", "top10pct", "rest"]


class RepeatRequest(BaseModel):
    """
    Input model v2: empat fitur RFM yang dihitung pada tanggal scoring
    (semua dari order SEBELUM tanggal scoring).
    """
    customer_id:  Optional[str] = Field(None, description="Customer ID for tracking")
    recency_days: int   = Field(..., ge=0, description="Days since the last order")
    frequency:    int   = Field(..., ge=1, description="Number of distinct orders")
    monetary:     float = Field(..., ge=0, description="Total paid across all orders (BRL)")
    tenure_days:  int   = Field(..., ge=0, description="Days since the first order")

    @model_validator(mode="after")
    def check_tenure(self) -> "RepeatRequest":
        """Order pertama tidak mungkin lebih baru dari order terakhir."""
        if self.tenure_days < self.recency_days:
            raise ValueError(
                f"tenure_days ({self.tenure_days}) must be >= recency_days ({self.recency_days})"
            )
        if self.frequency == 1 and self.tenure_days != self.recency_days:
            raise ValueError("A customer with 1 order must have tenure_days == recency_days")
        return self

    model_config = {"json_schema_extra": {
        "example": {
            "customer_id": "cust_001",
            "recency_days": 20,
            "frequency": 2,
            "monetary": 250.0,
            "tenure_days": 180,
        }
    }}


class RepeatResponse(BaseModel):
    """Hasil scoring repeat purchase untuk satu customer."""
    customer_id:       Optional[str] = None
    repeat_probability: float = Field(..., ge=0, le=1,
                                      description="Chance of ordering again within 90 days")
    segment:           RepeatSegment = Field(
        ..., description="Targeting segment from score cut-offs in the latest test snapshot")
    segment_lift:      Optional[float] = Field(
        None, description="Historical lift of this segment over the base rate (None for 'rest')")
    segment_precision: Optional[float] = Field(
        None, description="Historical repeat rate of this segment (None for 'rest')")
    model_version:     str


class BatchRepeatRequest(BaseModel):
    """Batch scoring repeat purchase."""
    customers: list[RepeatRequest] = Field(..., min_length=1, max_length=1000)


class BatchRepeatResponse(BaseModel):
    """Hasil batch scoring repeat purchase."""
    predictions:    list[RepeatResponse]
    total:          int
    segment_counts: dict[str, int]


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