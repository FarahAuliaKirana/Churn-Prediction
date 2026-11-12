"""
tests/test_backtest.py
Unit tests untuk pembentukan fold backtest (anti-leakage) dan agregasi hasil.
"""
import numpy as np
import pandas as pd
import pytest

from src.models.backtest import make_folds, run_backtest, summarize_backtest


class TestMakeFolds:
    def test_default_folds_expanding_window(self):
        folds = make_folds()
        assert [f["test"] for f in folds] == ["2017-12-01", "2018-03-01", "2018-06-01"]
        assert folds[0] == {"test": "2017-12-01", "val": "2017-09-01", "train": ["2017-06-01"]}
        assert folds[2]["train"] == ["2017-06-01", "2017-09-01", "2017-12-01"]
        assert folds[2]["val"] == "2018-03-01"

    def test_train_grows_each_fold(self):
        sizes = [len(f["train"]) for f in make_folds()]
        assert sizes == sorted(sizes) and sizes[0] < sizes[-1]

    def test_no_label_window_overlap(self):
        horizon = pd.Timedelta(days=90)
        for f in make_folds():
            val, test = pd.Timestamp(f["val"]), pd.Timestamp(f["test"])
            assert val + horizon <= test
            for tr in f["train"]:
                assert pd.Timestamp(tr) + horizon <= val

    def test_rejects_overlapping_grid(self):
        # Jarak antar snapshot 1 bulan < horizon 90 hari → label bertumpuk
        grid = ["2018-01-01", "2018-02-01", "2018-03-01"]
        with pytest.raises(ValueError):
            make_folds(["2018-03-01"], grid=grid)

    def test_rejects_test_without_history(self):
        with pytest.raises(ValueError):
            make_folds(["2017-09-01"])

    def test_rejects_unknown_cutoff(self):
        with pytest.raises(ValueError):
            make_folds(["2018-04-15"])


def _toy_snapshot(cutoff: str, n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Snapshot sintetis: peluang repeat turun seiring recency."""
    rng = np.random.default_rng(seed)
    recency = rng.integers(1, 365, n)
    p = 0.08 * np.exp(-recency / 60)
    return pd.DataFrame({
        "customer_unique_id": [f"c{i}" for i in range(n)],
        "snapshot_date": pd.Timestamp(cutoff),
        "recency_days": recency,
        "tenure_days": recency + rng.integers(0, 100, n),
        "frequency": rng.integers(1, 3, n),
        "orders_last_90d": (recency < 90).astype(int),
        "monetary": rng.uniform(20, 500, n),
        "avg_order_value": rng.uniform(20, 500, n),
        "avg_installments": rng.integers(1, 6, n).astype(float),
        "max_installments": rng.integers(1, 6, n),
        "avg_price": rng.uniform(10, 400, n),
        "total_price": rng.uniform(10, 400, n),
        "avg_freight": rng.uniform(5, 40, n),
        "freight_ratio": rng.uniform(0, 0.5, n),
        "total_items": rng.integers(1, 3, n),
        "avg_review_score": rng.integers(1, 6, n).astype(float),
        "min_review_score": rng.integers(1, 6, n).astype(float),
        "review_count": rng.integers(0, 2, n),
        "customer_state": rng.choice(["SP", "RJ"], n),
        "dominant_payment_type": rng.choice(["credit_card", "boleto"], n),
        "dominant_category": rng.choice(["toys", "bed_bath_table"], n),
        "is_repeat": (rng.random(n) < p).astype(int),
    })


@pytest.fixture(scope="module")
def results():
    cutoffs = ["2017-06-01", "2017-09-01", "2017-12-01", "2018-03-01"]
    snaps = {c: _toy_snapshot(c, seed=i) for i, c in enumerate(cutoffs)}
    folds = make_folds(["2017-12-01", "2018-03-01"], grid=cutoffs)
    return run_backtest(snaps, folds, n_boot=50)


class TestRunBacktest:
    def test_one_row_per_fold_and_model(self, results):
        fold_metrics, _ = results
        assert len(fold_metrics) == 2 * 4
        assert set(fold_metrics["model"]) == {"random", "recency_rule", "rfm_logreg", "xgboost"}

    def test_pr_auc_lift_is_pr_auc_over_base_rate(self, results):
        fm, _ = results
        expected = (fm["pr_auc"] / fm["base_rate"]).round(2)
        np.testing.assert_allclose(fm["pr_auc_lift"], expected, atol=0.01)

    def test_significance_rows(self, results):
        _, sig = results
        assert len(sig) == 2 * 2
        assert {"diff", "ci_low", "ci_high", "significant"} <= set(sig.columns)

    def test_summary_has_mean_min_max(self, results):
        fm, _ = results
        summary = summarize_backtest(fm)
        assert list(summary.index) == ["random", "recency_rule", "rfm_logreg", "xgboost"]
        assert ("roc_auc", "mean") in summary.columns
        assert (summary[("roc_auc", "min")] <= summary[("roc_auc", "max")]).all()
