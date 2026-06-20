"""
src/models/train.py
XGBoost training pipeline (champion model).

Changelog v1.2.0:
- find_optimal_threshold dihapus dari sini — dipindah ke evaluate.py
  sebagai single source of truth. train.py sekarang import dari sana.
- DEFAULT_THRESHOLD juga diambil dari evaluate.py agar konsisten.
- Threshold disimpan ke metrics.json dan dibaca API dari sana (bukan hardcode).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Single source of truth untuk threshold logic
from src.models.evaluate import find_optimal_threshold

logger = logging.getLogger(__name__)

# XGBoost hyperparameters — tuned in notebook 04_modeling.ipynb
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 9,   # ~90:10 class imbalance
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}


def build_pipeline() -> Pipeline:
    """Build sklearn Pipeline: Imputer → XGBoost.

    Note: StandardScaler intentionally excluded — XGBoost adalah tree ensemble
    dan invariant terhadap monotonic feature scaling. SimpleImputer tetap ada
    karena median imputation adalah pilihan preprocessing yang bermakna untuk
    menangani missing values saat inference.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**XGB_PARAMS)),
        ]
    )


def train(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    n_cv_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Pipeline, dict]:
    """
    Train XGBoost pipeline dengan cross-validation.

    Returns:
        pipeline: fitted Pipeline (Imputer → XGBoost)
        metrics:  dict berisi AUC, CV AUC, F1, optimal threshold,
                  dan full classification report.
                  Threshold disimpan ke metrics.json — API membacanya dari sana.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    logger.info("Train: %s | Churn rate: %.1f%%", X_train.shape, y_train.mean() * 100)
    logger.info("Test : %s | Churn rate: %.1f%%", X_test.shape,  y_test.mean()  * 100)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Threshold optimal dihitung dari held-out test set, lalu disimpan ke metrics.json
    threshold = find_optimal_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    f1  = f1_score(y_test, y_pred, average="weighted")

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    metrics = {
        "roc_auc":       round(auc, 4),
        "f1_weighted":   round(f1, 4),
        "cv_auc_mean":   round(float(cv_scores.mean()), 4),
        "cv_auc_std":    round(float(cv_scores.std()), 4),
        # Threshold disimpan ke metrics.json — API membacanya saat startup
        "threshold":     round(threshold, 4),
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "churn_rate_train": round(float(y_train.mean()), 4),
        "classification_report": classification_report(y_test, y_pred),
    }

    logger.info("ROC-AUC  : %.4f", auc)
    logger.info("CV AUC   : %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
    logger.info("F1       : %.4f", f1)
    logger.info("Threshold: %.4f (dibaca API dari metrics.json)", threshold)
    logger.info("\n%s", metrics["classification_report"])

    return pipe, metrics


def evaluate_against_baseline(
    new_auc: float,
    baseline_model_path: str | Path,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_improvement: float = 0.001,
) -> bool:
    """
    Bandingkan model baru vs model produksi saat ini.
    Return True jika model baru harus di-deploy.

    Args:
        min_improvement: minimum delta AUC untuk trigger deployment
                         (menghindari micro-improvement yang noisy).
    """
    baseline_path = Path(baseline_model_path)
    if not baseline_path.exists():
        logger.info("Tidak ada baseline model — deploy model baru.")
        return True

    baseline = joblib.load(baseline_path)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]
    baseline_auc   = roc_auc_score(y_test, baseline_proba)

    logger.info("Baseline AUC : %.4f", baseline_auc)
    logger.info("New model AUC: %.4f", new_auc)

    delta = new_auc - baseline_auc
    if delta >= min_improvement:
        logger.info("✅ New model lebih baik (Δ=+%.4f) — akan deploy.", delta)
        return True
    else:
        logger.warning(
            "⚠️  New model tidak cukup meningkat (Δ=%.4f, min=%.4f) — tetap pakai baseline.",
            delta, min_improvement,
        )
        return False


def save_model(pipeline: Pipeline, output_path: str | Path) -> None:
    """Simpan pipeline ke disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    logger.info("Model disimpan ke: %s", output_path)


def load_model(model_path: str | Path) -> Pipeline:
    """Load pipeline dari disk."""
    return joblib.load(model_path)