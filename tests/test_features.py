"""
tests/test_features.py
Unit tests untuk feature engineering pipeline.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    LEAKY_FEATURES,
    build_features,
    label_churn,
    split_X_y,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_master() -> pd.DataFrame:
    """Master DataFrame minimal untuk testing."""
    n = 200
    rng = np.random.default_rng(42)

    dates = pd.date_range("2017-01-01", periods=n, freq="3D")

    return pd.DataFrame({
        "customer_unique_id": [f"cust_{i % 50}" for i in range(n)],
        "order_id":           [f"ord_{i}"       for i in range(n)],
        "order_purchase_timestamp": dates,
        "payment_value":   rng.uniform(50, 500, n),
        "payment_installments": rng.integers(1, 12, n),
        "payment_type":    rng.choice(["credit_card", "boleto", "debit_card"], n),
        "price":           rng.uniform(30, 400, n),
        "freight_value":   rng.uniform(5, 60, n),
        "review_score":    rng.integers(1, 6, n).astype(float),
        "order_item_id":   rng.integers(1, 4, n),
        "customer_state":  rng.choice(["SP", "RJ", "MG"], n),
        "product_category_name_english": rng.choice(
            ["health_beauty", "bed_bath_table", "toys", "sports_leisure"], n
        ),
    })


@pytest.fixture
def sample_rfm(sample_master) -> pd.DataFrame:
    return label_churn(sample_master)


@pytest.fixture
def sample_features(sample_master, sample_rfm) -> pd.DataFrame:
    return build_features(sample_master, sample_rfm)


# ---------------------------------------------------------------------------
# Tests: label_churn
# ---------------------------------------------------------------------------

class TestLabelChurn:
    def test_output_columns(self, sample_rfm):
        expected = {"customer_unique_id", "recency", "frequency", "monetary", "is_churned"}
        assert expected.issubset(set(sample_rfm.columns))

    def test_is_churned_binary(self, sample_rfm):
        assert set(sample_rfm["is_churned"].unique()).issubset({0, 1})

    def test_unique_customers(self, sample_master, sample_rfm):
        n_unique = sample_master["customer_unique_id"].nunique()
        assert len(sample_rfm) == n_unique

    def test_recency_nonnegative(self, sample_rfm):
        assert (sample_rfm["recency"] >= 0).all()

    def test_frequency_positive(self, sample_rfm):
        assert (sample_rfm["frequency"] >= 1).all()


# ---------------------------------------------------------------------------
# Tests: build_features
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def test_no_leaky_features(self, sample_features):
        """Fitur leakage TIDAK boleh ada di feature matrix."""
        for col in LEAKY_FEATURES:
            assert col not in sample_features.columns, f"Leaky feature '{col}' masih ada!"

    def test_no_missing_values_in_target_or_rfm(self, sample_features):
        """Target & RFM tidak boleh NaN. Fitur lain boleh NaN — diimputasi di pipeline."""
        for col in ["is_churned", "frequency", "monetary", "total_orders"]:
            assert sample_features[col].notna().all(), f"{col} mengandung NaN"

    def test_ohe_columns_exist(self, sample_features):
        """Pastikan OHE menghasilkan kolom state, payment, category."""
        state_cols    = [c for c in sample_features.columns if c.startswith("customer_state_")]
        payment_cols  = [c for c in sample_features.columns if c.startswith("dominant_payment_type_")]
        category_cols = [c for c in sample_features.columns if c.startswith("dominant_category_")]

        assert len(state_cols)    > 0, "Tidak ada OHE customer_state"
        assert len(payment_cols)  > 0, "Tidak ada OHE payment_type"
        assert len(category_cols) > 0, "Tidak ada OHE category"

    def test_freight_ratio_bounded(self, sample_features):
        """freight_ratio harus >= 0."""
        assert (sample_features["freight_ratio"] >= 0).all()

    def test_log_transforms_nonnegative(self, sample_features):
        for col in ["log_monetary", "log_frequency"]:
            if col in sample_features.columns:
                assert (sample_features[col] >= 0).all()

    def test_row_count_matches_customers(self, sample_master, sample_features):
        n_unique = sample_master["customer_unique_id"].nunique()
        assert len(sample_features) == n_unique

    def test_derived_features_exist(self, sample_features):
        """Derived features harus ada."""
        for col in ["freight_ratio", "review_gap", "customer_lifetime_days", "avg_order_value"]:
            assert col in sample_features.columns, f"Derived feature '{col}' tidak ada!"

    def test_customer_lifetime_days_nonnegative(self, sample_features):
        assert (sample_features["customer_lifetime_days"] >= 0).all()


# ---------------------------------------------------------------------------
# Tests: split_X_y
# ---------------------------------------------------------------------------

class TestSplitXy:
    def test_no_target_in_X(self, sample_features):
        X, y = split_X_y(sample_features)
        assert "is_churned" not in X.columns

    def test_no_id_in_X(self, sample_features):
        X, y = split_X_y(sample_features)
        assert "customer_unique_id" not in X.columns

    def test_y_is_binary(self, sample_features):
        _, y = split_X_y(sample_features)
        assert set(y.unique()).issubset({0, 1})

    def test_shapes_consistent(self, sample_features):
        X, y = split_X_y(sample_features)
        assert len(X) == len(y) == len(sample_features)

    def test_feature_cols_numeric(self, sample_features):
        """Semua kolom X harus numerik (setelah OHE)."""
        X, _ = split_X_y(sample_features)
        non_numeric = X.select_dtypes(exclude="number").columns.tolist()
        assert len(non_numeric) == 0, f"Non-numeric columns in X: {non_numeric}"

# ---------------------------------------------------------------------------
# Tests: grain item vs order (regresi bug double-count)
# ---------------------------------------------------------------------------

@pytest.fixture
def multi_item_master() -> pd.DataFrame:
    """
    Dua customer, grain item seperti master.csv:
    - cust_a: 1 order berisi 3 item (payment & review berulang di tiap baris)
    - cust_b: 2 order masing-masing 1 item
    """
    return pd.DataFrame({
        "customer_unique_id": ["cust_a", "cust_a", "cust_a", "cust_b", "cust_b"],
        "order_id":           ["o1", "o1", "o1", "o2", "o3"],
        "order_purchase_timestamp": pd.to_datetime(
            ["2018-01-01", "2018-01-01", "2018-01-01", "2018-02-01", "2018-08-01"]
        ),
        "payment_value":        [300.0, 300.0, 300.0, 100.0, 50.0],
        "payment_installments": [3, 3, 3, 1, 1],
        "payment_type":   ["credit_card"] * 3 + ["boleto", "boleto"],
        "price":          [80.0, 90.0, 100.0, 90.0, 40.0],
        "freight_value":  [10.0, 10.0, 10.0, 10.0, 10.0],
        "review_score":   [5.0, 5.0, 5.0, 2.0, 4.0],
        "order_item_id":  [1, 2, 3, 1, 1],
        "customer_state": ["SP"] * 3 + ["RJ", "RJ"],
        "product_category_name_english": ["toys"] * 5,
    })


class TestOrderGrain:
    def test_monetary_not_double_counted(self, multi_item_master):
        rfm = label_churn(multi_item_master).set_index("customer_unique_id")
        assert rfm.loc["cust_a", "monetary"] == pytest.approx(300.0)
        assert rfm.loc["cust_b", "monetary"] == pytest.approx(150.0)

    def test_total_items_is_count_not_sum_of_item_ids(self, multi_item_master):
        rfm = label_churn(multi_item_master)
        feats = build_features(multi_item_master, rfm).set_index("customer_unique_id")
        assert feats.loc["cust_a", "total_items"] == 3   # bukan 1+2+3 = 6
        assert feats.loc["cust_b", "total_items"] == 2

    def test_payment_and_review_aggregated_per_order(self, multi_item_master):
        rfm = label_churn(multi_item_master)
        feats = build_features(multi_item_master, rfm).set_index("customer_unique_id")
        assert feats.loc["cust_a", "total_payment_value"] == pytest.approx(300.0)
        assert feats.loc["cust_a", "avg_payment_value"] == pytest.approx(300.0)
        assert feats.loc["cust_a", "review_count"] == 1   # 1 order = 1 review
        assert feats.loc["cust_b", "review_count"] == 2
        assert feats.loc["cust_b", "avg_order_value"] == pytest.approx(75.0)
