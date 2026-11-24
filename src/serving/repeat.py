"""
Scoring model v2 (repeat purchase) yang dipakai bersama oleh API dan dashboard.

API (`/v2/repeat/score`) dan dashboard Streamlit memanggil fungsi yang sama, jadi skor dan
segmen yang keluar dijamin identik. Dashboard memakai modul ini langsung saat API tidak
tersedia, misalnya di Streamlit Community Cloud yang hanya menjalankan Streamlit.

Modul ini sengaja hanya bergantung pada pandas, numpy, dan joblib (tanpa FastAPI, SHAP, XGBoost).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.features.snapshot import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_NAME = "rfm_logreg"
MODEL_VERSION = "2.0.0-rfm-logreg"
SEGMENT_NAMES = ("top1pct", "top5pct", "top10pct", "rest")
REST = {"segment": "rest", "lift": None, "precision": None}


def build_frame(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """
    Susun input untuk pipeline v2. Pipeline di-fit dengan semua kolom snapshot,
    jadi kolom yang tidak dipakai RFM LogReg diisi NaN / 'unknown'.
    """
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "customer_id"} for r in rows])
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        df[col] = "unknown"
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def load_segments(metrics_path: str | Path, model_name: str = MODEL_NAME) -> list[dict]:
    """Batas skor segmen dari metrics_snapshot.json, diurutkan dari segmen tersempit."""
    with open(metrics_path) as f:
        segments = json.load(f).get("segments", {}).get(model_name, [])
    return sorted(segments, key=lambda s: s["min_score"], reverse=True)


def assign_segment(score: float, segments: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Segmen tersempit yang batas skornya terlampaui; 'rest' jika tidak ada."""
    for seg in segments:
        if score >= seg["min_score"]:
            return seg
    return REST


def score_rows(
    model,
    segments: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    model_version: str = MODEL_VERSION,
) -> list[dict]:
    """Skor sekumpulan pelanggan. Bentuk keluaran sama dengan respons /v2/repeat/score."""
    probas = model.predict_proba(build_frame(rows))[:, 1]
    out = []
    for row, p in zip(rows, probas):
        seg = assign_segment(float(p), segments)
        out.append({
            "customer_id": row.get("customer_id"),
            "repeat_probability": round(float(p), 6),
            "segment": seg["segment"],
            "segment_lift": seg["lift"],
            "segment_precision": seg["precision"],
            "model_version": model_version,
        })
    return out


class RepeatScorer:
    """Model v2 beserta segmennya, siap dipakai tanpa API."""

    def __init__(self, model, segments: Sequence[Mapping[str, Any]],
                 model_version: str = MODEL_VERSION):
        self.model = model
        self.segments = list(segments)
        self.model_version = model_version

    @classmethod
    def from_files(cls, model_path: str | Path, metrics_path: str | Path,
                   model_name: str = MODEL_NAME, model_version: str = MODEL_VERSION):
        """Muat model .pkl dan segmen. Segmen kosong jika metrics tidak ada (semua 'rest')."""
        metrics_path = Path(metrics_path)
        segments = load_segments(metrics_path, model_name) if metrics_path.exists() else []
        return cls(joblib.load(model_path), segments, model_version)

    def score(self, rows: Sequence[Mapping[str, Any]]) -> list[dict]:
        return score_rows(self.model, self.segments, rows, self.model_version)
