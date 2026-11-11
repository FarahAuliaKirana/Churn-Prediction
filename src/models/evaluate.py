"""
src/models/evaluate.py
Evaluasi model: SHAP interpretability + metrik ringkas.
Ekstrak dari notebook 04_modeling.ipynb (bagian SHAP & threshold tuning)

Changelog v1.3.0:
- find_optimal_threshold: Youden's J (bukan F1 kelas churn + recall >= 0.99,
  yang membuat model menandai ~99% customer sebagai churn). Dipilih di
  validation set oleh train.py.
- Tambah threshold_report() dan ranking_report() (lift per decile, baseline
  "semua churn").

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

DEFAULT_THRESHOLD = 0.5  # fallback jika threshold tidak ada di metrics.json


def find_optimal_threshold(
    y_true: pd.Series,
    y_proba: np.ndarray,
    min_churn_recall: float = 0.0,
) -> float:
    """
    Pilih threshold yang memaksimalkan Youden's J (TPR - FPR), opsional dengan
    constraint churn recall >= `min_churn_recall`.

    Kenapa bukan F1 kelas churn? Churn adalah kelas MAYORITAS (~90%). F1 kelas
    mayoritas hampir selalu maksimal saat threshold sangat rendah, sehingga
    model berakhir menandai hampir semua customer sebagai churn (lift ~1.0).
    Youden's J menimbang kedua kelas setara, jadi model tetap membedakan.

    WAJIB dipanggil dengan data validation, bukan test set.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)

    if y_true.min() == y_true.max():
        logger.warning("Hanya satu kelas di y_true — pakai DEFAULT_THRESHOLD.")
        return float(DEFAULT_THRESHOLD)

    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    valid = np.isfinite(thresholds) & (tpr >= min_churn_recall)
    if not valid.any():
        logger.warning("Tidak ada threshold yang memenuhi recall %.2f.", min_churn_recall)
        return float(DEFAULT_THRESHOLD)

    j = np.where(valid, tpr - fpr, -np.inf)
    best = int(np.argmax(j))
    best_thresh = float(np.clip(thresholds[best], 0.0, 1.0))
    logger.info(
        "Optimal threshold: %.4f (Youden J=%.4f, TPR=%.3f, FPR=%.3f)",
        best_thresh, j[best], tpr[best], fpr[best],
    )
    return best_thresh


def threshold_report(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
) -> dict:
    """
    Confusion matrix + metrik per kelas pada `threshold`, dibandingkan
    dengan baseline naif "tandai semua customer sebagai churn".
    """
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_proba) >= threshold).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    def _div(a: int, b: int) -> float:
        return round(a / b, 4) if b else 0.0

    churn_precision = _div(tp, tp + fp)
    base_rate = round(float(y.mean()), 4)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "churn_precision": churn_precision,
        "churn_recall": _div(tp, tp + fn),
        "non_churn_precision": _div(tn, tn + fn),
        "non_churn_recall": _div(tn, tn + fp),
        "flagged_rate": _div(tp + fp, len(y)),
        # Baseline "semua churn": precision = base rate, non-churn recall = 0
        "baseline_all_churn_precision": base_rate,
        "churn_precision_lift": round(churn_precision / base_rate, 4) if base_rate else 0.0,
    }


def ranking_report(
    y_true: pd.Series,
    y_proba: np.ndarray,
    top_frac: float = 0.10,
) -> dict:
    """
    Metrik ranking yang tidak bergantung threshold — dipakai untuk targeting.

    - Decile risiko tertinggi: churn rate vs base rate. Karena base rate churn
      ~90%, lift maksimum yang mungkin hanya ~1.11x; angka ini wajar kecil.
    - Decile risiko terendah: proporsi customer yang KEMBALI (non-churn) vs
      base rate non-churn (~10%). Inilah sisi yang informatif untuk aktivasi
      repeat purchase.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba, dtype=float)
    k = max(1, int(round(len(y) * top_frac)))
    order = np.argsort(-p, kind="stable")

    base_churn = float(y.mean())
    base_return = 1.0 - base_churn
    top_churn = float(y[order[:k]].mean())
    bottom_return = float(1 - y[order[-k:]].mean())

    return {
        "top_frac": top_frac,
        "base_churn_rate": round(base_churn, 4),
        "top_decile_churn_rate": round(top_churn, 4),
        "top_decile_churn_lift": round(top_churn / base_churn, 4) if base_churn else 0.0,
        "max_possible_churn_lift": round(1 / base_churn, 4) if base_churn else 0.0,
        "base_return_rate": round(base_return, 4),
        "bottom_decile_return_rate": round(bottom_return, 4),
        "bottom_decile_return_lift": round(bottom_return / base_return, 4) if base_return else 0.0,
    }


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