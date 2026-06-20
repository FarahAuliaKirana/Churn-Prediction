"""
src/models/evaluate.py
Evaluasi model: SHAP interpretability + metrik ringkas.
Ekstrak dari notebook 04_modeling.ipynb (bagian SHAP & threshold tuning)

Changelog v1.2.0:
- find_optimal_threshold dipindahkan ke sini sebagai single source of truth.
  train.py tidak lagi mendefinisikan fungsi duplikat — ia import dari sini.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.470  # fallback jika threshold tidak ada di metrics.json


def find_optimal_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    min_churn_recall: float = 0.99,
) -> float:
    """
    Cari threshold optimal yang memaksimalkan F1 dengan constraint
    minimum churn recall >= `min_churn_recall`.

    Single source of truth — dipakai oleh train.py dan evaluate.py.
    Default fallback: 0.470 dari NB04.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    best_thresh = DEFAULT_THRESHOLD
    best_f1 = 0.0

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1     = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0

        if recall >= min_churn_recall and f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    logger.info("Optimal threshold: %.3f (F1=%.4f)", best_thresh, best_f1)
    return best_thresh


def compute_shap_values(
    pipeline: Pipeline,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Hitung SHAP values menggunakan TreeExplainer.
    Di-sample max_samples baris agar cepat (cukup untuk summary plot).

    Returns:
        shap_values: array of shape (n_samples, n_features)
        X_sample   : DataFrame yang dipakai (sudah di-transform oleh pipeline)
    """
    model = pipeline.named_steps["model"]

    # Transform X lewat preprocessing steps (tanpa model)
    preprocessing = Pipeline(pipeline.steps[:-1])
    X_transformed = preprocessing.transform(X)

    # Sample jika besar
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        X_sample      = pd.DataFrame(X_transformed, columns=X.columns).iloc[idx]
        X_sample_orig = X.iloc[idx]
    else:
        X_sample      = pd.DataFrame(X_transformed, columns=X.columns)
        X_sample_orig = X

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # XGBoost bisa return list of 2 arrays (binary) atau single array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values, X_sample_orig


def plot_shap_summary(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    output_dir: Optional[str | Path] = None,
    top_n: int = 20,
) -> None:
    """Buat SHAP summary plot dan simpan ke file jika output_dir diberikan."""
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, max_display=top_n, show=False)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        logger.info("SHAP summary plot disimpan ke %s", output_dir / "shap_summary.png")

    plt.close()


def top_shap_features(
    shap_values: np.ndarray,
    feature_names: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Return DataFrame top-N fitur berdasarkan mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df.sort_values("mean_abs_shap", ascending=False).head(top_n).reset_index(drop=True)