"""
tests/test_api_v2.py
Tests untuk endpoint v2 (repeat purchase, RFM Logistic Regression).
Model di-mock agar tes tidak bergantung pada file .pkl hasil training.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main as api
from src.api.main import app
from src.api.schemas import RepeatRequest
from src.features.snapshot import CATEGORICAL_FEATURES, NUMERIC_FEATURES

client = TestClient(app)

PAYLOAD = {
    "customer_id": "cust_001",
    "recency_days": 20,
    "frequency": 2,
    "monetary": 250.0,
    "tenure_days": 180,
}

# Diurutkan dari batas skor tertinggi, seperti yang dilakukan _load_repeat_model()
SEGMENTS = [
    {"segment": "top1pct", "min_score": 0.02, "precision": 0.036, "lift": 6.36},
    {"segment": "top5pct", "min_score": 0.01, "precision": 0.016, "lift": 2.82},
    {"segment": "top10pct", "min_score": 0.008, "precision": 0.013, "lift": 2.25},
]


def _mock_model(probas):
    m = MagicMock()
    m.predict_proba.side_effect = lambda X: np.column_stack(
        [1 - np.asarray(probas[: len(X)]), np.asarray(probas[: len(X)])]
    )
    return m


@pytest.fixture
def v2(request):
    probas = getattr(request, "param", [0.05])
    model = _mock_model(probas)
    with patch.object(api, "_repeat_model", model), \
         patch.object(api, "_repeat_segments", SEGMENTS):
        yield model


class TestRepeatScore:
    def test_returns_200_and_schema(self, v2):
        r = client.post("/v2/repeat/score", json=PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"customer_id", "repeat_probability", "segment",
                             "segment_lift", "segment_precision", "model_version"}
        assert body["customer_id"] == "cust_001"

    @pytest.mark.parametrize("v2,expected", [
        ([0.05], "top1pct"),
        ([0.015], "top5pct"),
        ([0.009], "top10pct"),
        ([0.001], "rest"),
    ], indirect=["v2"])
    def test_segment_assignment(self, v2, expected):
        body = client.post("/v2/repeat/score", json=PAYLOAD).json()
        assert body["segment"] == expected

    def test_segment_on_exact_cutoff_belongs_to_segment(self):
        with patch.object(api, "_repeat_segments", SEGMENTS):
            assert api._assign_segment(0.01)["segment"] == "top5pct"

    @pytest.mark.parametrize("v2", [[0.001]], indirect=True)
    def test_rest_has_no_lift(self, v2):
        body = client.post("/v2/repeat/score", json=PAYLOAD).json()
        assert body["segment_lift"] is None and body["segment_precision"] is None

    def test_top_segment_reports_historical_lift(self, v2):
        body = client.post("/v2/repeat/score", json=PAYLOAD).json()
        assert body["segment_lift"] == 6.36
        assert body["segment_precision"] == 0.036

    def test_no_segments_means_rest(self):
        with patch.object(api, "_repeat_model", _mock_model([0.9])), \
             patch.object(api, "_repeat_segments", []):
            assert client.post("/v2/repeat/score", json=PAYLOAD).json()["segment"] == "rest"

    def test_503_when_model_not_loaded(self):
        with patch.object(api, "_repeat_model", None):
            assert client.post("/v2/repeat/score", json=PAYLOAD).status_code == 503

    def test_500_on_model_error(self):
        broken = MagicMock()
        broken.predict_proba.side_effect = RuntimeError("boom")
        with patch.object(api, "_repeat_model", broken):
            assert client.post("/v2/repeat/score", json=PAYLOAD).status_code == 500


class TestRepeatValidation:
    @pytest.mark.parametrize("field,value", [
        ("recency_days", -1), ("frequency", 0), ("monetary", -5.0), ("tenure_days", -1),
    ])
    def test_negative_or_zero_values_rejected(self, v2, field, value):
        assert client.post("/v2/repeat/score", json={**PAYLOAD, field: value}).status_code == 422

    def test_tenure_shorter_than_recency_rejected(self, v2):
        bad = {**PAYLOAD, "recency_days": 200, "tenure_days": 100}
        assert client.post("/v2/repeat/score", json=bad).status_code == 422

    def test_single_order_requires_tenure_equal_recency(self, v2):
        bad = {**PAYLOAD, "frequency": 1, "recency_days": 30, "tenure_days": 90}
        assert client.post("/v2/repeat/score", json=bad).status_code == 422
        ok = {**PAYLOAD, "frequency": 1, "recency_days": 30, "tenure_days": 30}
        assert client.post("/v2/repeat/score", json=ok).status_code == 200

    def test_missing_field_rejected(self, v2):
        body = {k: v for k, v in PAYLOAD.items() if k != "monetary"}
        assert client.post("/v2/repeat/score", json=body).status_code == 422


class TestRepeatBatch:
    @pytest.mark.parametrize("v2", [[0.05, 0.015, 0.009, 0.001, 0.001]], indirect=True)
    def test_batch_counts_segments(self, v2):
        api._rate_store.clear()
        r = client.post("/v2/repeat/score/batch", json={"customers": [PAYLOAD] * 5})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 5
        assert body["segment_counts"] == {"top1pct": 1, "top5pct": 1, "top10pct": 1, "rest": 2}

    def test_batch_is_vectorized(self, v2):
        api._rate_store.clear()
        v2.predict_proba.side_effect = lambda X: np.column_stack(
            [np.full(len(X), 0.95), np.full(len(X), 0.05)])
        client.post("/v2/repeat/score/batch", json={"customers": [PAYLOAD] * 50})
        assert v2.predict_proba.call_count == 1

    def test_batch_empty_rejected(self, v2):
        assert client.post("/v2/repeat/score/batch", json={"customers": []}).status_code == 422

    def test_batch_503_when_model_not_loaded(self):
        with patch.object(api, "_repeat_model", None):
            r = client.post("/v2/repeat/score/batch", json={"customers": [PAYLOAD]})
            assert r.status_code == 503

    def test_batch_rate_limited(self, v2):
        api._rate_store.clear()
        with patch.object(api, "BATCH_RATE_LIMIT", 2):
            codes = [client.post("/v2/repeat/score/batch", json={"customers": [PAYLOAD]}).status_code
                     for _ in range(3)]
        api._rate_store.clear()
        assert codes == [200, 200, 429]


class TestRepeatInputFrame:
    def test_frame_has_all_pipeline_columns(self):
        df = api._repeat_requests_to_dataframe([RepeatRequest(**PAYLOAD)])
        assert list(df.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES
        assert df.loc[0, "recency_days"] == 20
        assert df.loc[0, "customer_state"] == "unknown"
        assert np.isnan(df.loc[0, "avg_freight"])


class TestRoot:
    def test_root_redirects_to_docs(self):
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == "/docs"

    def test_root_not_in_openapi_schema(self):
        assert "/" not in client.get("/openapi.json").json()["paths"]


class TestHealthV2:
    def test_health_reports_repeat_model(self, v2):
        body = client.get("/health").json()
        assert body["repeat_model_loaded"] is True
        assert body["repeat_model_version"] == api.REPEAT_MODEL_VERSION

    def test_health_repeat_model_missing(self):
        with patch.object(api, "_repeat_model", None):
            body = client.get("/health").json()
            assert body["repeat_model_loaded"] is False
            assert body["repeat_model_version"] is None
