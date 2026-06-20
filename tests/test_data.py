"""
tests/test_data.py
Unit tests untuk src/data/load.py — merge pipeline & validasi master DataFrame.
"""
import pandas as pd
import pytest
import numpy as np
from unittest.mock import patch

from src.data.load import (
    RAW_FILES,
    DATE_COLS,
    load_raw_tables,
    merge_tables,
    validate_master,
    build_master,
)


# ---------------------------------------------------------------------------
# Helper: buat minimal DataFrames yang mewakili 8 tabel Olist
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_tables():
    """Tabel minimal yang cukup untuk merge_tables tanpa crash."""
    customers = pd.DataFrame({
        "customer_id":        ["c1", "c2"],
        "customer_unique_id": ["u1", "u2"],
        "customer_state":     ["SP", "RJ"],
    })
    orders = pd.DataFrame({
        "order_id":                        ["o1", "o2"],
        "customer_id":                     ["c1", "c2"],
        "order_purchase_timestamp":        ["2021-01-01", "2021-02-01"],
        "order_approved_at":               ["2021-01-01", "2021-02-01"],
        "order_delivered_carrier_date":    [None, None],
        "order_delivered_customer_date":   [None, None],
        "order_estimated_delivery_date":   [None, None],
    })
    order_items = pd.DataFrame({
        "order_id":            ["o1", "o2"],
        "order_item_id":       [1, 1],
        "product_id":          ["p1", "p2"],
        "price":               [100.0, 200.0],
        "freight_value":       [10.0, 20.0],
        "shipping_limit_date": ["2021-01-05", "2021-02-05"],
    })
    payments = pd.DataFrame({
        "order_id":             ["o1", "o2"],
        "payment_value":        [110.0, 220.0],
        "payment_installments": [1, 2],
        "payment_type":         ["credit_card", "boleto"],
    })
    reviews = pd.DataFrame({
        "order_id":     ["o1", "o2"],
        "review_score": [5.0, 3.0],
    })
    products = pd.DataFrame({
        "product_id":            ["p1", "p2"],
        "product_category_name": ["beleza_saude", "brinquedos"],
    })
    category = pd.DataFrame({
        "product_category_name":         ["beleza_saude", "brinquedos"],
        "product_category_name_english": ["health_beauty", "toys"],
    })
    sellers = pd.DataFrame({
        "seller_id":    ["s1"],
        "seller_state": ["SP"],
    })
    return {
        "customers":   customers,
        "orders":      orders,
        "order_items": order_items,
        "payments":    payments,
        "reviews":     reviews,
        "products":    products,
        "sellers":     sellers,
        "category":    category,
    }


# ---------------------------------------------------------------------------
# Tests: load_raw_tables
# ---------------------------------------------------------------------------

class TestLoadRawTables:
    def test_raises_file_not_found_when_csv_missing(self, tmp_path):
        """Harus raise FileNotFoundError jika CSV tidak ada di direktori."""
        with pytest.raises(FileNotFoundError, match="tidak ditemukan"):
            load_raw_tables(tmp_path)

    def test_loads_all_tables_when_files_exist(self, tmp_path):
        """Harus return dict berisi semua tabel jika semua CSV ada."""
        for name, filename in RAW_FILES.items():
            fp = tmp_path / filename
            if name == "customers":
                pd.DataFrame({"customer_id": ["c1"], "customer_unique_id": ["u1"],
                              "customer_state": ["SP"]}).to_csv(fp, index=False)
            elif name == "orders":
                pd.DataFrame({"order_id": ["o1"], "customer_id": ["c1"],
                              "order_purchase_timestamp": ["2021-01-01"]}).to_csv(fp, index=False)
            else:
                pd.DataFrame({"dummy": [1]}).to_csv(fp, index=False)

        tables = load_raw_tables(tmp_path)
        assert set(tables.keys()) == set(RAW_FILES.keys())
        assert all(isinstance(v, pd.DataFrame) for v in tables.values())

    def test_raw_files_has_expected_8_keys(self):
        """RAW_FILES harus mengandung 8 key dataset Olist."""
        expected = {"customers", "orders", "order_items", "payments",
                    "reviews", "products", "sellers", "category"}
        assert set(RAW_FILES.keys()) == expected

    def test_date_cols_list_not_empty(self):
        """DATE_COLS harus berisi setidaknya kolom order_purchase_timestamp."""
        assert "order_purchase_timestamp" in DATE_COLS


# ---------------------------------------------------------------------------
# Tests: merge_tables
# ---------------------------------------------------------------------------

class TestMergeTables:
    def test_merge_returns_dataframe(self, minimal_tables):
        df = merge_tables(minimal_tables)
        assert isinstance(df, pd.DataFrame)

    def test_merge_contains_key_columns(self, minimal_tables):
        """Kolom kritis dari berbagai tabel harus ada setelah merge."""
        df = merge_tables(minimal_tables)
        for col in ["order_id", "customer_unique_id", "payment_value", "review_score"]:
            assert col in df.columns, f"Kolom '{col}' tidak ada setelah merge"

    def test_date_columns_parsed_as_datetime(self, minimal_tables):
        """order_purchase_timestamp harus di-parse sebagai datetime."""
        df = merge_tables(minimal_tables)
        assert pd.api.types.is_datetime64_any_dtype(df["order_purchase_timestamp"])

    def test_category_translation_applied(self, minimal_tables):
        """product_category_name_english harus ada dan berisi terjemahan yang benar."""
        df = merge_tables(minimal_tables)
        assert "product_category_name_english" in df.columns
        assert "health_beauty" in df["product_category_name_english"].values

    def test_payments_aggregated_sum_per_order(self, minimal_tables):
        """Payment ganda per order harus diagregasi — payment_value adalah sum."""
        tables = dict(minimal_tables)
        extra_payment = pd.DataFrame({
            "order_id":             ["o1"],
            "payment_value":        [50.0],
            "payment_installments": [1],
            "payment_type":         ["boleto"],
        })
        tables["payments"] = pd.concat(
            [tables["payments"], extra_payment], ignore_index=True
        )
        df = merge_tables(tables)
        o1_payment = df[df["order_id"] == "o1"]["payment_value"].iloc[0]
        assert o1_payment == 160.0  # 110 + 50

    def test_reviews_aggregated_mean_per_order(self, minimal_tables):
        """Review ganda per order harus diagregasi — review_score adalah mean."""
        tables = dict(minimal_tables)
        extra_review = pd.DataFrame({
            "order_id":     ["o1"],
            "review_score": [3.0],
        })
        tables["reviews"] = pd.concat(
            [tables["reviews"], extra_review], ignore_index=True
        )
        df = merge_tables(tables)
        o1_review = df[df["order_id"] == "o1"]["review_score"].iloc[0]
        assert o1_review == 4.0  # mean(5.0, 3.0)

    def test_row_count_at_least_num_orders(self, minimal_tables):
        """Jumlah baris setelah merge tidak boleh kurang dari jumlah orders."""
        df = merge_tables(minimal_tables)
        n_orders = minimal_tables["orders"]["order_id"].nunique()
        assert len(df) >= n_orders

    def test_left_join_preserves_all_orders(self, minimal_tables):
        """LEFT JOIN harus mempertahankan semua order_id dari tabel orders."""
        df = merge_tables(minimal_tables)
        original_order_ids = set(minimal_tables["orders"]["order_id"])
        merged_order_ids = set(df["order_id"].dropna())
        assert original_order_ids.issubset(merged_order_ids)


# ---------------------------------------------------------------------------
# Tests: validate_master
# ---------------------------------------------------------------------------

class TestValidateMaster:
    def test_passes_on_valid_dataframe(self, minimal_tables):
        """validate_master tidak boleh raise pada DataFrame yang valid."""
        df = merge_tables(minimal_tables)
        validate_master(df, original_order_count=df["order_id"].nunique())

    def test_raises_on_duplicate_rows(self, minimal_tables):
        """Harus raise ValueError jika ada baris duplikat."""
        df = merge_tables(minimal_tables)
        df_dup = pd.concat([df, df], ignore_index=True)
        with pytest.raises(ValueError, match="duplikat"):
            validate_master(df_dup, original_order_count=2)

    def test_raises_on_order_count_mismatch(self, minimal_tables):
        """Harus raise ValueError jika jumlah order_id tidak sesuai ekspektasi."""
        df = merge_tables(minimal_tables)
        with pytest.raises(ValueError, match="coverage mismatch"):
            validate_master(df, original_order_count=999)

    def test_raises_on_excessive_missing_in_key_cols(self, minimal_tables):
        """Harus raise ValueError jika kolom kritis punya >5% missing values."""
        df = merge_tables(minimal_tables)
        df = df.copy()
        df["customer_unique_id"] = None
        with pytest.raises(ValueError, match="missing values"):
            validate_master(df, original_order_count=df["order_id"].nunique())

    def test_raises_order_id_missing(self, minimal_tables):
        """Harus raise jika order_id punya >5% null."""
        df = merge_tables(minimal_tables)
        df = df.copy()
        df["order_id"] = None
        with pytest.raises(ValueError):
            validate_master(df, original_order_count=df["order_id"].nunique())


# ---------------------------------------------------------------------------
# Tests: build_master (integration — di-mock agar tidak butuh file nyata)
# ---------------------------------------------------------------------------

class TestBuildMaster:
    def test_build_master_calls_pipeline_in_order(self, tmp_path):
        """build_master harus: load_raw_tables → merge_tables → validate_master."""
        fake_orders = pd.DataFrame({"order_id": ["o1", "o2"]})
        fake_df = pd.DataFrame({
            "order_id":                 ["o1", "o2"],
            "customer_unique_id":       ["u1", "u2"],
            "order_purchase_timestamp": pd.to_datetime(["2021-01-01", "2021-02-01"]),
        })

        with patch("src.data.load.load_raw_tables") as mock_load, \
             patch("src.data.load.merge_tables") as mock_merge, \
             patch("src.data.load.validate_master") as mock_validate:

            mock_load.return_value = {"orders": fake_orders}
            mock_merge.return_value = fake_df
            mock_validate.return_value = None

            result = build_master(tmp_path)

            mock_load.assert_called_once_with(tmp_path)
            mock_merge.assert_called_once()
            mock_validate.assert_called_once()
            assert isinstance(result, pd.DataFrame)

    def test_build_master_returns_merged_dataframe(self, tmp_path):
        """build_master harus return DataFrame hasil merge_tables."""
        fake_orders = pd.DataFrame({"order_id": ["o1"]})
        fake_df = pd.DataFrame({
            "order_id":                 ["o1"],
            "customer_unique_id":       ["u1"],
            "order_purchase_timestamp": pd.to_datetime(["2021-01-01"]),
        })

        with patch("src.data.load.load_raw_tables", return_value={"orders": fake_orders}), \
             patch("src.data.load.merge_tables", return_value=fake_df), \
             patch("src.data.load.validate_master", return_value=None):

            result = build_master(tmp_path)
            assert len(result) == 1
            assert "order_id" in result.columns