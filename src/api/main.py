"""
src/api/main.py
FastAPI application untuk serving model churn (v1) dan repeat purchase (v2).

Endpoints:
    GET  /health                   — cek status API, kedua model, dan threshold aktif
    POST /v2/repeat/score          — v2: peluang order lagi dalam 90 hari + segmen targeting
    POST /v2/repeat/score/batch    — v2: batch scoring (vectorized, max 1000)
    POST /predict                  — v1 (legacy): prediksi churn satu customer
    POST /predict/batch            — v1 (legacy): prediksi churn banyak customer
    GET  /monitoring/drift         — statistik drift fitur v1 dalam rolling window
    GET  /monitoring/recent        — N prediksi v1 terakhir (untuk debugging)

Changelog v2.0.0:
- Tambah endpoint v2 yang memakai RFM Logistic Regression dari desain snapshot
  (train_snapshot_pipeline.py). Model ini divalidasi out-of-time di 3 periode;
  model v1 terbukti sebagian besar mengenali periode pembelian (uji placebo NB04).
- Endpoint v1 dipertahankan untuk kompatibilitas, ditandai deprecated di docs.

Changelog v1.3.0:
- Fix CORS: allow_origins dibaca dari env var CORS_ORIGINS (aman untuk production).
- Fix rate limiting: /predict/batch dibatasi BATCH_RATE_LIMIT req/menit per IP.
- Threshold, model, dan feature_cols tetap dibaca dari file saat startup (tidak berubah).
- DriftMonitor tetap terintegrasi ke /predict dan /predict/batch.
- Batch prediction tetap vectorized.
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.schemas import (
    BatchChurnRequest,
    BatchChurnResponse,
    BatchRepeatRequest,
    BatchRepeatResponse,
    ChurnRequest,
    ChurnResponse,
    DriftStats,
    HealthResponse,
    RepeatRequest,
    RepeatResponse,
)
from src.features.snapshot import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.models.evaluate import DEFAULT_THRESHOLD
from src.models.train import load_model
from src.monitoring.drift import DriftMonitor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config via environment variables
# ---------------------------------------------------------------------------
MODEL_PATH        = os.getenv("MODEL_PATH",        "outputs/model_xgb.pkl")
FEATURE_COLS_PATH = os.getenv("FEATURE_COLS_PATH", "outputs/feature_cols.json")
METRICS_PATH      = os.getenv("METRICS_PATH",      "outputs/metrics.json")
MONITOR_DB_PATH   = os.getenv("MONITOR_DB_PATH",   "monitoring/predictions.db")
MODEL_VERSION     = os.getenv("MODEL_VERSION",     "1.0.0")

# Model v2 — repeat purchase (snapshot design). Default: RFM Logistic Regression,
# model paling stabil di backtest 3 periode (lihat NB06).
REPEAT_MODEL_PATH    = os.getenv("REPEAT_MODEL_PATH",    "outputs/model_snapshot_rfm_logreg.pkl")
REPEAT_METRICS_PATH  = os.getenv("REPEAT_METRICS_PATH",  "outputs/metrics_snapshot.json")
REPEAT_MODEL_NAME    = os.getenv("REPEAT_MODEL_NAME",    "rfm_logreg")
REPEAT_MODEL_VERSION = os.getenv("REPEAT_MODEL_VERSION", "2.0.0-rfm-logreg")

# CORS: default aman (Streamlit local saja).
# Untuk deploy ke Streamlit Cloud / HuggingFace Spaces:
#   export CORS_ORIGINS=https://your-app.streamlit.app,https://your-hf-space.hf.space
# Untuk buka semua (demo/dev lokal):
#   export CORS_ORIGINS=*
_cors_env = os.getenv("CORS_ORIGINS", "")
CORS_ORIGINS: list[str] = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env
    else ["http://localhost:8501", "http://127.0.0.1:8501"]
)

# Rate limit /predict/batch: N request per menit per IP.
# Default 20 sudah cukup longgar untuk batch testing tapi cegah DoS.
# Override: export BATCH_RATE_LIMIT=50
BATCH_RATE_LIMIT = int(os.getenv("BATCH_RATE_LIMIT", "20"))
_RATE_WINDOW = 60.0  # detik

# ---------------------------------------------------------------------------
# Simple in-memory rate limiter (cukup untuk portfolio; gunakan Redis di prod)
# ---------------------------------------------------------------------------
_rate_store: dict[str, list[float]] = defaultdict(list)


def _is_rate_limited(client_ip: str, limit: int) -> bool:
    """Return True jika request MELEBIHI limit (harus di-reject)."""
    now = time.monotonic()
    cutoff = now - _RATE_WINDOW
    # Hapus timestamp yang sudah di luar window
    _rate_store[client_ip] = [t for t in _rate_store[client_ip] if t > cutoff]
    if len(_rate_store[client_ip]) >= limit:
        return True
    _rate_store[client_ip].append(now)
    return False


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_model        = None
_feature_cols = None
_threshold    = DEFAULT_THRESHOLD
_monitor: Optional[DriftMonitor] = None
_repeat_model = None
_repeat_segments: list[dict] = []   # batas skor segmen dari metrics_snapshot.json


def _load_repeat_model() -> None:
    """Load model v2 + batas skor segmen. Non-fatal: endpoint v2 menjawab 503 jika gagal."""
    global _repeat_model, _repeat_segments
    model_path = Path(REPEAT_MODEL_PATH)
    if model_path.exists():
        _repeat_model = load_model(model_path)
        logger.info("Repeat model (v2) loaded from: %s", model_path)
    else:
        logger.warning("Repeat model (v2) tidak ditemukan: %s — jalankan train_snapshot_pipeline.py",
                       model_path)

    metrics_path = Path(REPEAT_METRICS_PATH)
    if metrics_path.exists():
        with open(metrics_path) as f:
            segments = json.load(f).get("segments", {}).get(REPEAT_MODEL_NAME, [])
        # Urutkan dari batas skor tertinggi (segmen tersempit) ke terendah
        _repeat_segments = sorted(segments, key=lambda s: s["min_score"], reverse=True)
        logger.info("Repeat segments loaded: %s", [s["segment"] for s in _repeat_segments])
    else:
        _repeat_segments = []
        logger.warning("metrics_snapshot.json tidak ditemukan: %s — semua skor masuk segmen 'rest'",
                       metrics_path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model, feature cols, threshold, dan init monitor saat startup."""
    global _model, _feature_cols, _threshold, _monitor, _repeat_model, _repeat_segments

    model_path = Path(MODEL_PATH)
    if model_path.exists():
        _model = load_model(model_path)
        logger.info("Model loaded from: %s", model_path)
    else:
        logger.warning("Model file tidak ditemukan: %s", model_path)

    fc_path = Path(FEATURE_COLS_PATH)
    if fc_path.exists():
        with open(fc_path) as f:
            _feature_cols = json.load(f)
        logger.info("Feature cols loaded: %d kolom", len(_feature_cols))
    else:
        logger.warning("feature_cols.json tidak ditemukan: %s", fc_path)

    metrics_path = Path(METRICS_PATH)
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        _threshold = metrics.get("threshold", DEFAULT_THRESHOLD)
        logger.info("Threshold loaded from metrics.json: %.4f", _threshold)
    else:
        _threshold = DEFAULT_THRESHOLD
        logger.warning(
            "metrics.json tidak ditemukan — pakai fallback threshold: %.4f. "
            "Jalankan train_pipeline.py untuk generate file ini.",
            _threshold,
        )

    _monitor = DriftMonitor(db_path=MONITOR_DB_PATH)
    logger.info("DriftMonitor initialised at: %s", MONITOR_DB_PATH)

    _load_repeat_model()

    yield

    _model = None
    _feature_cols = None
    _monitor = None
    _repeat_model = None
    _repeat_segments = []
    logger.info("Shutting down — resources released.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Olist Repeat Purchase & Churn API",
    description=(
        "**v2 — Repeat purchase (recommended):** `/v2/repeat/score` uses an RFM logistic "
        "regression from the snapshot design, validated out-of-time across 3 periods "
        "(top-1% lift 6.1–6.7×). Returns the chance of ordering again within 90 days and a "
        "targeting segment.\n\n"
        "**v1 — Churn (legacy, deprecated):** `/predict` uses XGBoost with the original churn "
        "label. A placebo test showed it mostly recognises purchase timing, so do not use it "
        "for targeting decisions.\n\n"
        f"**Rate limit:** batch endpoints allow `{BATCH_RATE_LIMIT}` requests per minute per IP "
        "(override with the `BATCH_RATE_LIMIT` env var).\n\n"
        "**CORS:** controlled by the `CORS_ORIGINS` env var (default: local Streamlit)."
    ),
    version=REPEAT_MODEL_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _request_to_dataframe(req: ChurnRequest) -> pd.DataFrame:
    data = req.model_dump()
    data.pop("customer_id", None)

    monetary  = data.get("monetary", 0)
    frequency = data.get("frequency", 1)
    data["log_monetary"]  = float(np.log1p(monetary))
    data["log_frequency"] = float(np.log1p(frequency))

    state    = data.pop("customer_state")
    pay_type = data.pop("dominant_payment_type")
    category = data.pop("dominant_category")

    row = pd.DataFrame([data])

    if _feature_cols:
        for col in _feature_cols:
            if col.startswith("customer_state_"):
                row[col] = int(state == col.replace("customer_state_", ""))
            elif col.startswith("dominant_payment_type_"):
                row[col] = int(pay_type == col.replace("dominant_payment_type_", ""))
            elif col.startswith("dominant_category_"):
                row[col] = int(category == col.replace("dominant_category_", ""))
        row = row.reindex(columns=_feature_cols, fill_value=0)

    return row


def _predict_one(req: ChurnRequest) -> ChurnResponse:
    df    = _request_to_dataframe(req)
    proba = float(_model.predict_proba(df)[0, 1])

    result = ChurnResponse(
        customer_id=req.customer_id,
        churn_probability=round(proba, 4),
        is_churn=proba >= _threshold,
        threshold_used=_threshold,
        model_version=MODEL_VERSION,
    )

    if _monitor is not None:
        _monitor.log(
            features=req.model_dump(),
            churn_probability=proba,
            is_churn=result.is_churn,
            customer_id=req.customer_id,
        )

    return result


def _repeat_requests_to_dataframe(reqs: list[RepeatRequest]) -> pd.DataFrame:
    """
    Susun input untuk pipeline v2. Pipeline di-fit dengan semua kolom snapshot,
    jadi kolom yang tidak dipakai RFM LogReg diisi NaN / 'unknown'.
    """
    df = pd.DataFrame([r.model_dump(exclude={"customer_id"}) for r in reqs])
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        df[col] = "unknown"
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def _assign_segment(score: float) -> dict:
    """Segmen tersempit yang batas skornya terlampaui; 'rest' jika tidak ada."""
    for seg in _repeat_segments:
        if score >= seg["min_score"]:
            return seg
    return {"segment": "rest", "lift": None, "precision": None}


def _score_repeat(reqs: list[RepeatRequest]) -> list[RepeatResponse]:
    probas = _repeat_model.predict_proba(_repeat_requests_to_dataframe(reqs))[:, 1]
    out = []
    for req, p in zip(reqs, probas):
        seg = _assign_segment(float(p))
        out.append(RepeatResponse(
            customer_id=req.customer_id,
            repeat_probability=round(float(p), 6),
            segment=seg["segment"],
            segment_lift=seg["lift"],
            segment_precision=seg["precision"],
            model_version=REPEAT_MODEL_VERSION,
        ))
    return out


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Alamat utama diarahkan ke dokumentasi interaktif, bukan 404."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Status of the API, both models, and the active v1 threshold."""
    return HealthResponse(
        status="ok" if _model is not None else "degraded",
        model_loaded=_model is not None,
        model_version=MODEL_VERSION if _model is not None else None,
        threshold=_threshold if _model is not None else None,
        repeat_model_loaded=_repeat_model is not None,
        repeat_model_version=REPEAT_MODEL_VERSION if _repeat_model is not None else None,
    )


@app.post("/v2/repeat/score", response_model=RepeatResponse, tags=["v2 — Repeat purchase"])
async def score_repeat(req: RepeatRequest):
    """
    Chance that a customer orders again within 90 days, plus a targeting segment.

    All inputs are measured on the scoring date, using only orders before it.
    Segments (`top1pct`, `top5pct`, `top10pct`, `rest`) use score cut-offs from the
    latest test snapshot; `segment_lift` is that segment's historical lift over the
    base rate.
    """
    if _repeat_model is None:
        raise HTTPException(status_code=503, detail="v2 model is not loaded. Check /health.")
    try:
        return _score_repeat([req])[0]
    except Exception as e:
        logger.error("Repeat scoring error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Scoring failed. Check server logs.")


@app.post("/v2/repeat/score/batch", response_model=BatchRepeatResponse,
          tags=["v2 — Repeat purchase"])
async def score_repeat_batch(batch: BatchRepeatRequest, request: Request):
    """Batch repeat-purchase scoring (up to 1,000 customers, vectorized, rate limited per IP)."""
    if _repeat_model is None:
        raise HTTPException(status_code=503, detail="v2 model is not loaded. Check /health.")

    client_ip = request.client.host if request.client else "unknown"
    if _is_rate_limited(client_ip, BATCH_RATE_LIMIT):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too Many Requests: at most {BATCH_RATE_LIMIT} requests per minute "
                f"per IP for this endpoint. Try again in 60 seconds."
            ),
        )
    try:
        preds = _score_repeat(batch.customers)
    except Exception as e:
        logger.error("Repeat batch scoring error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Batch scoring failed. Check server logs.")

    counts = {s: 0 for s in ("top1pct", "top5pct", "top10pct", "rest")}
    for p in preds:
        counts[p.segment] += 1
    return BatchRepeatResponse(predictions=preds, total=len(preds), segment_counts=counts)


@app.post("/predict", response_model=ChurnResponse, tags=["v1 — Churn (legacy)"], deprecated=True)
async def predict(req: ChurnRequest):
    """Churn prediction for one customer (v1, deprecated — use /v2/repeat/score)."""
    if _model is None:
        raise HTTPException(status_code=503, detail="v1 model is not loaded. Check /health.")
    if _feature_cols is None:
        raise HTTPException(status_code=503, detail="feature_cols.json not found.")
    try:
        return _predict_one(req)
    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed. Check server logs.")


@app.post("/predict/batch", response_model=BatchChurnResponse, tags=["v1 — Churn (legacy)"],
          deprecated=True)
async def predict_batch(batch: BatchChurnRequest, request: Request):
    """
    Churn prediction for up to 1,000 customers (v1, deprecated).

    Vectorized: satu predict_proba call untuk seluruh batch.

    Rate limited: BATCH_RATE_LIMIT request/menit per IP (default 20).
    Set env var BATCH_RATE_LIMIT untuk mengubah nilai ini.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="v1 model is not loaded. Check /health.")
    if _feature_cols is None:
        raise HTTPException(status_code=503, detail="feature_cols.json not found.")

    client_ip = request.client.host if request.client else "unknown"
    if _is_rate_limited(client_ip, BATCH_RATE_LIMIT):
        raise HTTPException(
            status_code=429,
            detail=(
                f"Too Many Requests: at most {BATCH_RATE_LIMIT} requests per minute "
                f"per IP for this endpoint. Try again in 60 seconds."
            ),
        )

    try:
        frames = [_request_to_dataframe(req) for req in batch.customers]
        df_all = pd.concat(frames, ignore_index=True)
        probas = _model.predict_proba(df_all)[:, 1]

        predictions = []
        for req, proba in zip(batch.customers, probas):
            proba  = float(proba)
            result = ChurnResponse(
                customer_id=req.customer_id,
                churn_probability=round(proba, 4),
                is_churn=proba >= _threshold,
                threshold_used=_threshold,
                model_version=MODEL_VERSION,
            )
            predictions.append(result)

            if _monitor is not None:
                _monitor.log(
                    features=req.model_dump(),
                    churn_probability=proba,
                    is_churn=result.is_churn,
                    customer_id=req.customer_id,
                )

        churn_count = sum(p.is_churn for p in predictions)
        return BatchChurnResponse(
            predictions=predictions,
            total=len(predictions),
            churn_count=churn_count,
            churn_rate=round(churn_count / len(predictions), 4),
        )

    except Exception as e:
        logger.error("Batch prediction error: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Batch prediction failed. Check server logs.")


@app.get("/monitoring/drift", response_model=DriftStats, tags=["Monitoring"])
async def get_drift(
    window_hours: int = Query(default=24, ge=1, le=720, description="Rolling window dalam jam"),
):
    """Statistik drift fitur dalam rolling window terakhir."""
    if _monitor is None:
        raise HTTPException(status_code=503, detail="Drift monitor is not initialised.")

    stats = _monitor.drift_stats(
        window_hours=window_hours,
        baseline_path=METRICS_PATH if Path(METRICS_PATH).exists() else None,
    )

    if "error" in stats:
        raise HTTPException(status_code=500, detail=stats["error"])

    return DriftStats(
        window_hours=stats.get("window_hours", window_hours),
        request_count=stats.get("request_count", 0),
        feature_means=stats.get("feature_means", {}),
        churn_rate_recent=stats.get("churn_rate_recent"),
        drift_flags=stats.get("drift_flags"),
        flagged_features=stats.get("flagged_features", []),
    )


@app.get("/monitoring/recent", tags=["Monitoring"])
async def get_recent_predictions(
    limit: int = Query(default=50, ge=1, le=500, description="Jumlah prediksi terakhir"),
):
    """N prediksi terakhir dari database monitoring."""
    if _monitor is None:
        raise HTTPException(status_code=503, detail="Drift monitor is not initialised.")
    return {"predictions": _monitor.recent_predictions(limit=limit)}