"""
src/models/backtest.py
Backtesting model repeat purchase: evaluasi out-of-time di beberapa periode.

Satu test snapshot (Jun 2018) belum cukup untuk menjawab "apakah performanya stabil,
atau kebetulan di satu periode?". Backtest mengulang seluruh prosedur training —
pemilihan hyperparameter, refit, evaluasi — di beberapa titik waktu, masing-masing
hanya memakai data yang tersedia sebelum periode test-nya (expanding window):

    fold  train snapshots              validation   test
    1     2017-06                      2017-09      2017-12
    2     2017-06, 2017-09             2017-12      2018-03
    3     2017-06, 2017-09, 2017-12    2018-03      2018-06   (= desain utama NB06)

Syarat anti-leakage per fold: jendela label setiap snapshot train berakhir sebelum
snapshot validation, dan jendela label validation berakhir sebelum snapshot test.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

import pandas as pd

from src.features.snapshot import HORIZON_DAYS, build_snapshot
from src.models.snapshot_model import train_and_evaluate

logger = logging.getLogger(__name__)

# Grid snapshot kuartalan. Snapshot 2017-03-01 tidak dipakai untuk training
# karena hanya punya 17 repeater.
QUARTERLY_CUTOFFS = ["2017-06-01", "2017-09-01", "2017-12-01", "2018-03-01", "2018-06-01"]
DEFAULT_TEST_CUTOFFS = ["2017-12-01", "2018-03-01", "2018-06-01"]

SUMMARY_METRICS = ["roc_auc", "pr_auc", "pr_auc_lift", "lift_top1pct", "lift_top5pct", "lift_top10pct"]


def make_folds(
    test_cutoffs: Sequence[str] = DEFAULT_TEST_CUTOFFS,
    grid: Sequence[str] = QUARTERLY_CUTOFFS,
    horizon_days: int = HORIZON_DAYS,
) -> list[dict]:
    """
    Buat fold expanding-window: validation = snapshot tepat sebelum test,
    train = semua snapshot di grid sebelum validation.

    Raises:
        ValueError jika test cutoff tidak ada di grid, fold tidak punya snapshot
        train, atau ada jendela label yang tumpang tindih dengan periode berikutnya.
    """
    grid_ts = sorted(pd.Timestamp(c) for c in grid)
    horizon = pd.Timedelta(days=horizon_days)
    folds = []
    for t in test_cutoffs:
        test = pd.Timestamp(t)
        if test not in grid_ts:
            raise ValueError(f"Test cutoff {t} tidak ada di grid")
        i = grid_ts.index(test)
        if i < 2:
            raise ValueError(f"Test cutoff {t} tidak punya cukup snapshot sebelumnya")
        val = grid_ts[i - 1]
        train = grid_ts[: i - 1]

        for tr in train:
            if tr + horizon > val:
                raise ValueError(f"Label snapshot train {tr.date()} tumpang tindih dengan validation {val.date()}")
        if val + horizon > test:
            raise ValueError(f"Label validation {val.date()} tumpang tindih dengan test {test.date()}")

        folds.append({
            "test": str(test.date()),
            "val": str(val.date()),
            "train": [str(d.date()) for d in train],
        })
    return folds


def build_snapshot_cache(
    master: pd.DataFrame,
    cutoffs: Iterable[str],
    reviews: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Bangun setiap snapshot sekali saja; dipakai ulang oleh banyak fold."""
    return {c: build_snapshot(master, c, reviews=reviews) for c in sorted(set(cutoffs))}


def run_backtest(
    snapshots: dict[str, pd.DataFrame],
    folds: Sequence[dict],
    n_boot: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Jalankan train_and_evaluate di setiap fold.

    Returns:
        fold_metrics : satu baris per (fold, model) — metrik ranking di test snapshot
        significance : satu baris per (fold, perbandingan) — paired bootstrap PR-AUC
    """
    metric_rows, sig_rows = [], []
    for k, fold in enumerate(folds, start=1):
        train_df = pd.concat([snapshots[c] for c in fold["train"]], ignore_index=True)
        val_df = snapshots[fold["val"]]
        test_df = snapshots[fold["test"]]
        logger.info("Fold %d: train %s | val %s | test %s", k, fold["train"], fold["val"], fold["test"])

        _, metrics, _ = train_and_evaluate(train_df, val_df, test_df, n_boot=n_boot)

        for model, m in metrics["test"].items():
            metric_rows.append({
                "fold": k,
                "test_snapshot": fold["test"],
                "n_train_snapshots": len(fold["train"]),
                "train_positives": int(train_df["is_repeat"].sum()),
                "model": model,
                **{key: m[key] for key in ("n", "positives", "base_rate", "roc_auc", "pr_auc",
                                           "precision_top1pct", "lift_top1pct",
                                           "lift_top5pct", "lift_top10pct", "recall_top10pct")},
                "pr_auc_lift": round(m["pr_auc"] / m["base_rate"], 2) if m["base_rate"] else 0.0,
                "best_params": str(metrics["best_params"]) if model == "xgboost" else "",
            })
        for comparison, s in metrics["significance"].items():
            sig_rows.append({
                "fold": k,
                "test_snapshot": fold["test"],
                "comparison": comparison,
                **s,
                "significant": s["ci_low"] > 0,
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(sig_rows)


def summarize_backtest(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata, minimum, dan maksimum setiap metrik per model di semua fold."""
    agg = fold_metrics.groupby("model")[SUMMARY_METRICS].agg(["mean", "min", "max"])
    order = [m for m in ("random", "recency_rule", "rfm_logreg", "xgboost") if m in agg.index]
    return agg.loc[order].round(4)
