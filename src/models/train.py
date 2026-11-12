"""
src/models/train.py
XGBoost training pipeline (champion model).

Changelog v1.3.0:
- scale_pos_weight tidak lagi hardcode 9. Kelas positif (churn) adalah
  MAYORITAS (~90%), jadi nilai 9 justru memberi bobot 9x ke kelas mayoritas
  dan mendorong model menandai hampir semua customer sebagai churn.
  Default sekarang tanpa reweighting (probabilitas tetap terkalibrasi);
  class_weighting="balanced" memakai neg/pos (~0.11) bila diperlukan.
- Split train / validation / test (60/20/20). Threshold dipilih di validation,
  semua metrik akhir dilaporkan di test yang tidak tersentuh.
- CV hanya di train+val (test tidak ikut).
- metrics.json sekarang berisi threshold report, ranking report (lift), dan
  feature_means_train untuk baseline drift monitoring.

Changelog v1.2.0:
- find_optimal_threshold dihapus dari sini — dipindah ke evaluate.py
  sebagai single source of truth. train.py sekarang import dari sana.
- Threshold disimpan ke metrics.json dan dibaca API dari sana (bukan hardcode).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Tuple

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

from src.models.evaluate import find_optimal_threshold, ranking_report, threshold_report

logger = logging.getLogger(__name__)

# XGBoost hyperparameters — tuned in notebook 04_modeling.ipynb.
# scale_pos_weight sengaja tidak di sini: ditentukan oleh `class_weighting`.
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "random_state": 42,
    "n_jobs": -1,
}

# Fitur numerik yang dibandingkan oleh DriftMonitor (src/monitoring/drift.py)
DRIFT_BASELINE_FEATURES = [
    "avg_payment_value", "total_payment_value", "avg_installments", "max_installments",
    "avg_price", "total_price", "avg_freight", "freight_ratio",
    "avg_review_score", "min_review_score", "review_gap", "review_count",
    "total_items", "total_orders", "avg_order_value",
    "frequency", "monetary", "customer_lifetime_days",
]


def compute_scale_pos_weight(y: pd.Series) -> float:
    """XGBoost: scale_pos_weight = jumlah negatif / jumlah positif."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return neg / pos if pos else 1.0


def build_pipeline(scale_pos_weight: float = 1.0) -> Pipeline:
    """Build sklearn Pipeline: Imputer → XGBoost.

    Note: StandardScaler intentionally excluded — XGBoost adalah tree ensemble
    dan invariant terhadap monotonic feature scaling. SimpleImputer di-fit hanya
    pada data train, sehingga statistik test set tidak bocor.
    """
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(scale_pos_weight=scale_pos_weight, **XGB_PARAMS)),
        ]
    )


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    """
    Stratified split 60/20/20. Test split identik dengan
    train_test_split(X, y, test_size=0.2, stratify=y, random_state=42),
    sehingga train_pipeline.py bisa membandingkan baseline di test set yang sama.
    """
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev,
        test_size=val_size / (1 - test_size),
        stratify=y_dev,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    n_cv_folds: int = 5,
    random_state: int = 42,
    class_weighting: Literal["none", "balanced"] = "none",
    min_churn_recall: float = 0.0,
) -> Tuple[Pipeline, dict]:
    """
    Train XGBoost pipeline.

    Alur:
        1. Split train / val / test (stratified).
        2. Fit di train.
        3. Pilih threshold di val (Youden's J).
        4. Laporkan AUC, F1, confusion matrix, lift di test.
        5. CV AUC di train+val (test tidak dipakai).

    Returns:
        pipeline: fitted Pipeline (Imputer → XGBoost)
        metrics:  dict siap disimpan ke metrics.json (+ classification_report string).
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, test_size=test_size, val_size=val_size, random_state=random_state,
    )
    for name, yy in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        logger.info("%-5s: %6d rows | churn rate %.1f%%", name, len(yy), yy.mean() * 100)

    spw = compute_scale_pos_weight(y_train) if class_weighting == "balanced" else 1.0
    logger.info("class_weighting=%s → scale_pos_weight=%.4f", class_weighting, spw)

    pipe = build_pipeline(scale_pos_weight=spw)
    pipe.fit(X_train, y_train)

    # Threshold dari VALIDATION set
    val_proba = pipe.predict_proba(X_val)[:, 1]
    threshold = find_optimal_threshold(y_val, val_proba, min_churn_recall=min_churn_recall)

    # Semua metrik akhir dari TEST set
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average="weighted")
    thr_rep = threshold_report(y_test, y_proba, threshold)
    rank_rep = ranking_report(y_test, y_proba)

    # Cross-validation di train+val (dev set) saja. Dev set diambil dengan split
    # pertama yang sama persis, agar urutan baris (dan fold CV) identik dengan NB04.
    X_dev, _, y_dev, _ = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(
        build_pipeline(scale_pos_weight=spw), X_dev, y_dev, cv=cv, scoring="roc_auc", n_jobs=-1,
    )

    feature_means_train = {
        f: round(float(X_train[f].mean()), 4)
        for f in DRIFT_BASELINE_FEATURES
        if f in X_train.columns
    }

    metrics = {
        "roc_auc":       round(float(auc), 4),
        "f1_weighted":   round(float(f1), 4),
        "cv_auc_mean":   round(float(cv_scores.mean()), 4),
        "cv_auc_std":    round(float(cv_scores.std()), 4),
        # Threshold dipilih di validation set — API membacanya saat startup
        "threshold":     round(threshold, 4),
        "threshold_method": "youden_j_on_validation",
        "class_weighting": class_weighting,
        "scale_pos_weight": round(spw, 4),
        "n_train":       len(X_train),
        "n_val":         len(X_val),
        "n_test":        len(X_test),
        "churn_rate_train": round(float(y_train.mean()), 4),
        "test_threshold_report": thr_rep,
        "test_ranking_report": rank_rep,
        "feature_means_train": feature_means_train,
        "classification_report": classification_report(y_test, y_pred),
    }

    logger.info("ROC-AUC (test): %.4f", auc)
    logger.info("CV AUC (dev)  : %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
    logger.info("F1 weighted   : %.4f", f1)
    logger.info("Threshold     : %.4f (dipilih di validation)", threshold)
    logger.info("Threshold report (test): %s", thr_rep)
    logger.info("Ranking report (test)  : %s", rank_rep)
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
