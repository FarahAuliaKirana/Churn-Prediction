"""
src/models/snapshot_model.py
Training & evaluasi model repeat purchase berbasis snapshot.

Validasi bersifat out-of-time:
    train      : snapshot-snapshot lama
    validation : snapshot berikutnya (pemilihan hyperparameter)
    test       : snapshot terakhir yang jendela labelnya masih lengkap

Model dibandingkan dengan baseline:
    - random            : skor acak (precision = base rate)
    - recency_rule      : urutkan customer dari yang paling baru belanja
                          → "model" yang hanya tahu timing
    - rfm_logreg        : Logistic Regression di log(recency, frequency, monetary, tenure)
Jika XGBoost tidak mengalahkan recency_rule secara signifikan, berarti fitur
tambahan tidak memberi sinyal di luar timing pembelian.
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.features.snapshot import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET

logger = logging.getLogger(__name__)

TOP_FRACS = (0.01, 0.05, 0.10)
RFM_FEATURES = ["recency_days", "frequency", "monetary", "tenure_days"]

BASE_XGB_PARAMS = {
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
}

# Grid kecil, dipilih berdasarkan PR-AUC di validation snapshot
PARAM_GRID = [
    {"max_depth": 3, "n_estimators": 200},
    {"max_depth": 3, "n_estimators": 400},
    {"max_depth": 5, "n_estimators": 200},
    {"max_depth": 5, "n_estimators": 400},
]


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def build_xgb_pipeline(max_depth: int = 3, n_estimators: int = 200) -> Pipeline:
    """Imputer + OneHot (kategori jarang → 'infrequent') + XGBoost. Semua di-fit di train."""
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  max_categories=12, sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )
    model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, **BASE_XGB_PARAMS)
    return Pipeline([("pre", pre), ("model", model)])


def build_rfm_logreg() -> Pipeline:
    """Baseline linear: log1p(RFM + tenure) → scaler → Logistic Regression."""
    pre = ColumnTransformer(
        [("rfm", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p)),
            ("scale", StandardScaler()),
        ]), RFM_FEATURES)]
    )
    return Pipeline([("pre", pre), ("model", LogisticRegression(max_iter=1000))])


def recency_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Skor = semakin baru order terakhir, semakin tinggi. Tidak perlu training."""
    return -df["recency_days"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ranking_metrics(
    y_true: Sequence[int],
    score: Sequence[float],
    fracs: Sequence[float] = TOP_FRACS,
) -> dict:
    """ROC-AUC, PR-AUC, dan precision / recall / lift di top-k% customer."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=float)
    base = y.mean()
    order = np.argsort(-s, kind="stable")
    out = {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": round(float(base), 5),
        "roc_auc": round(float(roc_auc_score(y, s)), 4),
        "pr_auc": round(float(average_precision_score(y, s)), 4),
    }
    for f in fracs:
        k = max(1, int(round(len(y) * f)))
        hits = y[order[:k]].sum()
        prec = hits / k
        key = f"top{int(round(f * 100))}pct"
        out[f"precision_{key}"] = round(float(prec), 4)
        out[f"recall_{key}"] = round(float(hits / y.sum()), 4) if y.sum() else 0.0
        out[f"lift_{key}"] = round(float(prec / base), 2) if base else 0.0
    return out


def bootstrap_diff(
    y_true: Sequence[int],
    score_a: Sequence[float],
    score_b: Sequence[float],
    metric: Callable[[np.ndarray, np.ndarray], float] = average_precision_score,
    n_boot: int = 500,
    seed: int = 42,
) -> dict:
    """
    Paired bootstrap selisih metric(a) - metric(b) di data yang sama.
    CI 95% yang tidak memuat 0 berarti perbedaannya signifikan.
    """
    y = np.asarray(y_true).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
    diffs = np.array(diffs)
    return {
        "diff": round(float(metric(y, a) - metric(y, b)), 4),
        "ci_low": round(float(np.percentile(diffs, 2.5)), 4),
        "ci_high": round(float(np.percentile(diffs, 97.5)), 4),
        "p_value_one_sided": round(float((diffs <= 0).mean()), 4),
        "n_boot": int(len(diffs)),
    }


def segment_cutoffs(
    score: Sequence[float],
    metrics: dict,
    fracs: Sequence[float] = TOP_FRACS,
) -> list[dict]:
    """
    Batas skor untuk segmen top-k% di test snapshot terbaru, beserta precision & lift
    historisnya. Dipakai API untuk menerjemahkan probabilitas menjadi segmen targeting.
    Diurutkan dari segmen paling sempit (top 1%) ke paling lebar.

    Segmen bersifat EKSKLUSIF (top5pct = peringkat 1–5%, tanpa top 1%), sesuai cara API
    menempatkan customer. Karena itu precision & lift dihitung per band, bukan kumulatif:
    precision band = (hits top-k − hits top-k sebelumnya) / (n top-k − n top-k sebelumnya).
    """
    s = np.asarray(score, dtype=float)
    n = len(s)
    base = metrics["base_rate"]
    out = []
    prev_k, prev_hits = 0, 0.0
    for f in sorted(fracs):
        key = f"top{int(round(f * 100))}pct"
        k = max(1, int(round(n * f)))
        hits = metrics[f"precision_{key}"] * k
        band_precision = (hits - prev_hits) / (k - prev_k) if k > prev_k else 0.0
        out.append({
            "segment": key,
            "min_score": round(float(np.quantile(s, 1 - f)), 6),
            "precision": round(float(band_precision), 4),
            "lift": round(float(band_precision / base), 2) if base else 0.0,
            "cumulative_precision": metrics[f"precision_{key}"],
            "cumulative_lift": metrics[f"lift_{key}"],
        })
        prev_k, prev_hits = k, hits
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _xy(df: pd.DataFrame):
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]


def select_params(train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Pilih hyperparameter XGBoost berdasarkan PR-AUC di validation snapshot."""
    X_tr, y_tr = _xy(train_df)
    X_va, y_va = _xy(val_df)
    rows = []
    for params in PARAM_GRID:
        pipe = build_xgb_pipeline(**params).fit(X_tr, y_tr)
        s = pipe.predict_proba(X_va)[:, 1]
        rows.append({**params,
                     "val_pr_auc": average_precision_score(y_va, s),
                     "val_roc_auc": roc_auc_score(y_va, s)})
    grid = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False).reset_index(drop=True)
    best = {"max_depth": int(grid.loc[0, "max_depth"]),
            "n_estimators": int(grid.loc[0, "n_estimators"])}
    logger.info("Grid validation:\n%s", grid.round(4).to_string())
    logger.info("Parameter terpilih: %s", best)
    return best, grid


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_boot: int = 500,
) -> tuple[dict[str, Pipeline], dict, dict[str, np.ndarray]]:
    """
    1. Pilih hyperparameter di validation.
    2. Fit ulang XGBoost & baseline di train + validation.
    3. Evaluasi semua model di test snapshot (out-of-time).

    Returns:
        models : {"xgboost": Pipeline, "rfm_logreg": Pipeline} (fit di train + val)
        metrics: dict siap disimpan ke JSON
        scores : skor setiap model di test snapshot (untuk plot di notebook)
    """
    best_params, grid = select_params(train_df, val_df)

    dev_df = pd.concat([train_df, val_df], ignore_index=True)
    X_dev, y_dev = _xy(dev_df)
    X_te, y_te = _xy(test_df)

    xgb = build_xgb_pipeline(**best_params).fit(X_dev, y_dev)
    rfm = build_rfm_logreg().fit(X_dev, y_dev)

    rng = np.random.default_rng(42)
    scores = {
        "random": rng.random(len(test_df)),
        "recency_rule": recency_rule_score(test_df),
        "rfm_logreg": rfm.predict_proba(X_te)[:, 1],
        "xgboost": xgb.predict_proba(X_te)[:, 1],
    }
    test_metrics = {name: ranking_metrics(y_te, s) for name, s in scores.items()}

    significance = {
        "xgboost_vs_recency_rule_pr_auc": bootstrap_diff(
            y_te, scores["xgboost"], scores["recency_rule"], n_boot=n_boot),
        "xgboost_vs_rfm_logreg_pr_auc": bootstrap_diff(
            y_te, scores["xgboost"], scores["rfm_logreg"], n_boot=n_boot),
    }

    metrics = {
        "design": {
            "train_snapshots": sorted(str(d.date()) for d in train_df["snapshot_date"].unique()),
            "val_snapshot": str(pd.Timestamp(val_df["snapshot_date"].iloc[0]).date()),
            "test_snapshot": str(pd.Timestamp(test_df["snapshot_date"].iloc[0]).date()),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
        },
        "best_params": best_params,
        "validation_grid": grid.round(4).to_dict(orient="records"),
        "test": test_metrics,
        "significance": significance,
        "segments": {
            name: segment_cutoffs(scores[name], test_metrics[name])
            for name in ("rfm_logreg", "xgboost")
        },
        # Koefisien LogReg pada fitur log1p + standardisasi (arah & besar pengaruh)
        "rfm_logreg_coefficients": {
            f"log_{f}": round(float(c), 4)
            for f, c in zip(RFM_FEATURES, rfm.named_steps["model"].coef_[0])
        },
    }
    for name, m in test_metrics.items():
        logger.info("%-13s ROC-AUC %.4f | PR-AUC %.4f | lift@10%% %.2f",
                    name, m["roc_auc"], m["pr_auc"], m["lift_top10pct"])
    return {"xgboost": xgb, "rfm_logreg": rfm}, metrics, scores
