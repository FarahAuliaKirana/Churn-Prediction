"""
conftest.py
Shared pytest fixtures — available to all test files.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def minimal_master_df() -> pd.DataFrame:
    """
    Minimal master DataFrame yang mewakili output build_master().
    Dipakai oleh test_features dan test_api saat butuh end-to-end fixture.
    """
    n = 100
    rng = np.random.default_rng(0)
    dates = pd.date_range("2017-06-01", periods=n, freq="5D")

    return pd.DataFrame({
        "customer_unique_id": [f"cust_{i % 20}" for i in range(n)],
        "order_id":           [f"ord_{i}"        for i in range(n)],
        "order_purchase_timestamp": dates,
        "payment_value":      rng.uniform(50, 500, n),
        "payment_installments": rng.integers(1, 6, n),
        "payment_type":       rng.choice(["credit_card", "boleto", "debit_card"], n),
        "price":              rng.uniform(30, 400, n),
        "freight_value":      rng.uniform(5, 50, n),
        "review_score":       rng.integers(1, 6, n).astype(float),
        "order_item_id":      rng.integers(1, 3, n),
        "customer_state":     rng.choice(["SP", "RJ", "MG", "BA"], n),
        "product_category_name_english": rng.choice(
            ["health_beauty", "toys", "bed_bath_table", "sports_leisure"], n
        ),
    })