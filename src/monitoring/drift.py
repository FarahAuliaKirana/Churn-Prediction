"""
src/monitoring/drift.py
Lightweight input drift monitoring untuk churn prediction API.

Approach:
- Setiap /predict call menyimpan satu baris feature values + prediction result
  ke SQLite database (monitoring/predictions.db).
- /monitoring/drift menghitung rata-rata fitur dalam rolling window dan
  membandingkannya terhadap training-time baseline di metrics.json.
- Tidak ada external dependency — SQLite adalah bagian dari Python stdlib.

Mengapa SQLite dan bukan file-per-request?
  Concurrent writes ke plain JSON/CSV menyebabkan corruption. SQLite WAL mode
  menangani multiple writers dengan aman dan dapat dibaca oleh pandas, DBeaver,
  atau analytics tool manapun tanpa setup tambahan.

Changelog v1.2.0:
- Tidak ada perubahan di file ini — semua perbaikan ada di main.py
  (integrasi monitor.log() ke /predict dan /predict/batch,
   serta penambahan endpoint /monitoring/drift dan /monitoring/recent).

Changelog v1.2.1:
- Fix DeprecationWarning: datetime.utcnow() diganti datetime.now(timezone.utc)
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_lock = threading.Lock()


class DriftMonitor:
    """
    Thread-safe drift monitor backed by SQLite.

    Usage:
        monitor = DriftMonitor("monitoring/predictions.db")
        monitor.log(features_dict, churn_probability, is_churn)
        stats = monitor.drift_stats(window_hours=24)
    """

    NUMERIC_FEATURES = [
        "avg_payment_value", "total_payment_value", "avg_installments",
        "avg_price", "avg_freight", "freight_ratio",
        "avg_review_score", "min_review_score", "review_gap", "review_count",
        "total_items", "total_orders", "avg_order_value",
        "frequency", "monetary", "customer_lifetime_days",
    ]

    def __init__(self, db_path: str | Path = "monitoring/predictions.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")  # safe untuk concurrent writes
        return conn

    def _init_db(self) -> None:
        """Buat tabel jika belum ada."""
        cols = ", ".join(f"{f} REAL" for f in self.NUMERIC_FEATURES)
        with _lock, self._connect() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    customer_id TEXT,
                    {cols},
                    churn_probability REAL,
                    is_churn INTEGER
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts ON predictions(ts)"
            )
        logger.info("DriftMonitor initialised at %s", self.db_path)

    def log(
        self,
        features: dict,
        churn_probability: float,
        is_churn: bool,
        customer_id: Optional[str] = None,
    ) -> None:
        """Log satu prediksi ke database. Non-blocking saat error."""
        try:
            row = {f: features.get(f) for f in self.NUMERIC_FEATURES}
            row["ts"] = datetime.now(timezone.utc).isoformat()  # fix: was utcnow()
            row["customer_id"] = customer_id
            row["churn_probability"] = churn_probability
            row["is_churn"] = int(is_churn)

            cols = ", ".join(row.keys())
            placeholders = ", ".join("?" for _ in row)
            with _lock, self._connect() as conn:
                conn.execute(
                    f"INSERT INTO predictions ({cols}) VALUES ({placeholders})",
                    list(row.values()),
                )
        except Exception as exc:
            # Monitoring tidak boleh mengganggu prediction path
            logger.warning("DriftMonitor.log failed (non-fatal): %s", exc)

    def drift_stats(
        self,
        window_hours: int = 24,
        baseline_path: Optional[str | Path] = None,
    ) -> dict:
        """
        Hitung rata-rata fitur dalam `window_hours` jam terakhir.
        Opsional: bandingkan terhadap training baseline di metrics.json.

        Returns dict:
            window_hours, request_count, feature_means,
            churn_rate_recent, drift_flags (jika baseline diberikan),
            flagged_features (fitur dengan drift > 20%)
        """
        since = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()  # fix: was utcnow()

        try:
            with self._connect() as conn:
                df = pd.read_sql_query(
                    "SELECT * FROM predictions WHERE ts >= ?",
                    conn,
                    params=(since,),
                )
        except Exception as exc:
            logger.error("DriftMonitor.drift_stats read error: %s", exc)
            return {"error": str(exc)}

        if df.empty:
            return {
                "window_hours": window_hours,
                "request_count": 0,
                "feature_means": {},
                "churn_rate_recent": None,
            }

        feature_means = {
            f: round(float(df[f].mean()), 4)
            for f in self.NUMERIC_FEATURES
            if f in df.columns and df[f].notna().any()
        }

        churn_rate = float(df["is_churn"].mean()) if "is_churn" in df.columns else None

        result: dict = {
            "window_hours": window_hours,
            "request_count": len(df),
            "feature_means": feature_means,
            "churn_rate_recent": round(churn_rate, 4) if churn_rate is not None else None,
        }

        # Bandingkan terhadap training baseline jika disediakan
        if baseline_path:
            try:
                with open(baseline_path) as f:
                    baseline = json.load(f)
                train_means = baseline.get("feature_means_train", {})
                drift_flags = {}
                for feat, recent_mean in feature_means.items():
                    if feat in train_means:
                        train_mean = train_means[feat]
                        if train_mean and train_mean != 0:
                            pct_change = abs(recent_mean - train_mean) / abs(train_mean)
                            drift_flags[feat] = {
                                "train_mean": round(train_mean, 4),
                                "recent_mean": recent_mean,
                                "pct_change": round(pct_change * 100, 2),
                                "flagged": pct_change > 0.20,  # threshold 20%
                            }
                result["drift_flags"] = drift_flags
                flagged = [k for k, v in drift_flags.items() if v["flagged"]]
                result["flagged_features"] = flagged
            except Exception as exc:
                logger.warning("Tidak bisa load baseline untuk drift comparison: %s", exc)

        return result

    def recent_predictions(self, limit: int = 100) -> list[dict]:
        """Return N prediksi terakhir (untuk debugging & audit)."""
        try:
            with self._connect() as conn:
                df = pd.read_sql_query(
                    "SELECT ts, customer_id, churn_probability, is_churn "
                    "FROM predictions ORDER BY id DESC LIMIT ?",
                    conn,
                    params=(limit,),
                )
            return df.to_dict(orient="records")
        except Exception as exc:
            logger.error("DriftMonitor.recent_predictions error: %s", exc)
            return []