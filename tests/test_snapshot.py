"""
tests/test_snapshot.py
Unit tests untuk desain snapshot: fitur hanya dari masa lalu, label hanya dari masa depan.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.snapshot import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_snapshot,
    build_snapshots,
)
from src.models.snapshot_model import bootstrap_diff, ranking_metrics, recency_rule_score


def _row(cust, order, ts, pay=100.0, item=1, price=80.0, freight=20.0):
    return {
        "customer_unique_id": cust, "order_id": order,
        "order_purchase_timestamp": pd.Timestamp(ts),
        "payment_value": pay, "payment_installments": 1, "payment_type": "credit_card",
        "review_score": 5.0, "order_item_id": item, "price": price, "freight_value": freight,
        "customer_state": "SP", "product_category_name_english": "toys",
    }


@pytest.fixture
def master() -> pd.DataFrame:
    return pd.DataFrame([
        # A: order sebelum cutoff, lalu order lagi di jendela label → repeat
        _row("A", "a1", "2018-01-10"),
        _row("A", "a2", "2018-03-05", pay=999.0),
        # B: hanya order sebelum cutoff → bukan repeat
        _row("B", "b1", "2018-02-20"),
        _row("B", "b1", "2018-02-20", item=2),          # order multi-item
        # C: order kedua SETELAH jendela label → bukan repeat
        _row("C", "c1", "2017-12-01"),
        _row("C", "c2", "2018-07-01"),
        # D: order pertama setelah cutoff → tidak masuk populasi snapshot
        _row("D", "d1", "2018-03-10"),
    ])


@pytest.fixture
def reviews() -> pd.DataFrame:
    return pd.DataFrame({
        "order_id": ["a1", "b1", "c1"],
        "review_score": [4, 1, 3],
        # review b1 ditulis SETELAH cutoff → tidak boleh dipakai
        "review_creation_date": pd.to_datetime(["2018-01-20", "2018-03-02", "2017-12-10"]),
    })


CUTOFF = "2018-03-01"


class TestSnapshotLabelAndPopulation:
    def test_population_only_customers_active_before_cutoff(self, master):
        snap = build_snapshot(master, CUTOFF)
        assert set(snap["customer_unique_id"]) == {"A", "B", "C"}

    def test_label_only_from_horizon_window(self, master):
        snap = build_snapshot(master, CUTOFF, horizon_days=90).set_index("customer_unique_id")
        assert snap.loc["A", TARGET] == 1   # order 5 Mar, di dalam jendela
        assert snap.loc["B", TARGET] == 0
        assert snap.loc["C", TARGET] == 0   # order 1 Jul, di luar jendela 90 hari

    def test_features_ignore_future_orders(self, master):
        snap = build_snapshot(master, CUTOFF).set_index("customer_unique_id")
        # Order a2 (BRL 999) terjadi setelah cutoff — tidak boleh masuk monetary
        assert snap.loc["A", "monetary"] == pytest.approx(100.0)
        assert snap.loc["A", "frequency"] == 1
        assert snap.loc["C", "frequency"] == 1

    def test_recency_and_tenure_relative_to_cutoff(self, master):
        snap = build_snapshot(master, CUTOFF).set_index("customer_unique_id")
        assert snap.loc["A", "recency_days"] == (pd.Timestamp(CUTOFF) - pd.Timestamp("2018-01-10")).days
        assert snap.loc["C", "tenure_days"] == (pd.Timestamp(CUTOFF) - pd.Timestamp("2017-12-01")).days

    def test_multi_item_order_counted_once(self, master):
        snap = build_snapshot(master, CUTOFF).set_index("customer_unique_id")
        assert snap.loc["B", "monetary"] == pytest.approx(100.0)
        assert snap.loc["B", "total_items"] == 2

    def test_output_columns(self, master):
        snap = build_snapshot(master, CUTOFF)
        for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET, "snapshot_date"]:
            assert col in snap.columns

    def test_raises_when_no_history(self, master):
        with pytest.raises(ValueError):
            build_snapshot(master, "2010-01-01")


class TestSnapshotReviews:
    def test_reviews_written_after_cutoff_are_excluded(self, master, reviews):
        snap = build_snapshot(master, CUTOFF, reviews=reviews).set_index("customer_unique_id")
        assert snap.loc["A", "avg_review_score"] == 4
        assert np.isnan(snap.loc["B", "avg_review_score"])   # review b1 dibuat 2 Mar
        assert snap.loc["B", "review_count"] == 0

    def test_without_reviews_features_are_nan(self, master):
        snap = build_snapshot(master, CUTOFF)
        assert snap["avg_review_score"].isna().all()


class TestBuildSnapshots:
    def test_concat_multiple_cutoffs(self, master):
        snaps = build_snapshots(master, ["2018-01-01", CUTOFF])
        assert snaps["snapshot_date"].nunique() == 2
        # Pada cutoff 1 Jan 2018 hanya C yang sudah pernah order
        first = snaps[snaps["snapshot_date"] == pd.Timestamp("2018-01-01")]
        assert set(first["customer_unique_id"]) == {"C"}


class TestRankingMetrics:
    def test_perfect_ranking(self):
        y = np.array([1] * 10 + [0] * 90)
        s = np.linspace(1, 0, 100)
        m = ranking_metrics(y, s, fracs=(0.10,))
        assert m["roc_auc"] == 1.0
        assert m["precision_top10pct"] == 1.0
        assert m["lift_top10pct"] == pytest.approx(10.0)
        assert m["recall_top10pct"] == 1.0

    def test_recency_rule_prefers_recent(self):
        df = pd.DataFrame({"recency_days": [5, 100, 30]})
        assert np.argmax(recency_rule_score(df)) == 0

    def test_bootstrap_detects_clear_difference(self):
        rng = np.random.default_rng(0)
        y = np.array([1] * 50 + [0] * 450)
        good = y + rng.normal(0, 0.3, 500)
        bad = rng.normal(0, 1, 500)
        res = bootstrap_diff(y, good, bad, n_boot=200)
        assert res["ci_low"] > 0
