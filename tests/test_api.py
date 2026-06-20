"""
tests/test_api.py
Unit & integration tests untuk FastAPI endpoints.

Changelog v1.3.0 (dari v1.2.0):
- Tambah TestEdgeCases: input ekstrem (freight=0, monetary=0), unknown customer_state,
  batch 1000 customer — ini yang biasa ditanya interviewer.
- Tambah TestRateLimit: /predict/batch 429 setelah melebihi BATCH_RATE_LIMIT.
- Tambah TestCORS: header CORS sesuai env var, bukan allow_origins=["*"].
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

from src.api.main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "avg_payment_value":    150.0,
    "total_payment_value":  150.0,
    "avg_installments":     3.0,
    "max_installments":     3,
    "avg_price":            120.0,
    "total_price":          120.0,
    "avg_freight":          30.0,
    "freight_ratio":        0.20,
    "avg_review_score":     4.0,
    "min_review_score":     4.0,
    "review_gap":           0.0,
    "review_count":         1,
    "total_items":          1,
    "total_orders":         1,
    "avg_order_value":      150.0,
    "frequency":            1,
    "monetary":             150.0,
    "customer_lifetime_days": 0,
    "customer_state":         "SP",
    "dominant_payment_type":  "credit_card",
    "dominant_category":      "health_beauty",
}

MOCK_FEATURE_COLS = ["avg_freight", "avg_payment_value", "monetary", "frequency",
                     "log_monetary", "log_frequency"]
MOCK_THRESHOLD = 0.47


# ---------------------------------------------------------------------------
# Tests: /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_schema(self):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] in ("ok", "degraded")

    def test_health_exposes_threshold_when_model_loaded(self):
        mock = MagicMock()
        with patch("src.api.main._model", mock), \
             patch("src.api.main._threshold", MOCK_THRESHOLD):
            data = client.get("/health").json()
            assert data["threshold"] == MOCK_THRESHOLD


# ---------------------------------------------------------------------------
# Tests: /predict
# ---------------------------------------------------------------------------

class TestPredict:
    @pytest.fixture(autouse=True)
    def mock_deps(self):
        mock = MagicMock()
        mock.predict_proba.return_value = np.array([[0.15, 0.85]])
        mock_monitor = MagicMock()
        with patch("src.api.main._model", mock), \
             patch("src.api.main._feature_cols", MOCK_FEATURE_COLS), \
             patch("src.api.main._threshold", MOCK_THRESHOLD), \
             patch("src.api.main._monitor", mock_monitor):
            yield mock

    def test_predict_returns_200(self):
        assert client.post("/predict", json=SAMPLE_PAYLOAD).status_code == 200

    def test_predict_response_schema(self):
        data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert "churn_probability" in data
        assert "is_churn" in data
        assert "threshold_used" in data
        assert "model_version" in data

    def test_churn_probability_range(self):
        data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert 0.0 <= data["churn_probability"] <= 1.0

    def test_is_churn_matches_threshold(self):
        data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert data["is_churn"] == (data["churn_probability"] >= data["threshold_used"])

    def test_threshold_used_comes_from_metrics_not_hardcode(self):
        custom_threshold = 0.30
        with patch("src.api.main._threshold", custom_threshold):
            data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
            assert data["threshold_used"] == custom_threshold

    def test_invalid_payload_returns_422(self):
        bad = {**SAMPLE_PAYLOAD, "avg_payment_value": -100}
        assert client.post("/predict", json=bad).status_code == 422

    def test_missing_field_returns_422(self):
        incomplete = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "avg_freight"}
        assert client.post("/predict", json=incomplete).status_code == 422

    def test_drift_monitor_log_called_on_predict(self):
        mock_monitor = MagicMock()
        with patch("src.api.main._monitor", mock_monitor):
            client.post("/predict", json=SAMPLE_PAYLOAD)
            mock_monitor.log.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: /predict — edge cases (yang biasa ditanya interviewer)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.fixture(autouse=True)
    def mock_deps(self):
        mock = MagicMock()
        mock.predict_proba.return_value = np.array([[0.5, 0.5]])
        with patch("src.api.main._model", mock), \
             patch("src.api.main._feature_cols", MOCK_FEATURE_COLS), \
             patch("src.api.main._threshold", MOCK_THRESHOLD), \
             patch("src.api.main._monitor", MagicMock()):
            yield mock

    def test_freight_zero(self):
        """freight=0 adalah valid (customer pickup / free shipping)."""
        payload = {**SAMPLE_PAYLOAD, "avg_freight": 0.0, "freight_ratio": 0.0}
        assert client.post("/predict", json=payload).status_code == 200

    def test_monetary_zero(self):
        """monetary=0 bisa terjadi jika order di-cancel sebelum dibayar."""
        payload = {**SAMPLE_PAYLOAD, "monetary": 0.0, "avg_payment_value": 0.0,
                   "total_payment_value": 0.0}
        assert client.post("/predict", json=payload).status_code == 200

    def test_unknown_customer_state(self):
        """State yang tidak ada di training data harus tidak crash."""
        payload = {**SAMPLE_PAYLOAD, "customer_state": "XX"}
        assert client.post("/predict", json=payload).status_code == 200

    def test_unknown_payment_type(self):
        """Payment type di luar training set harus tidak crash."""
        payload = {**SAMPLE_PAYLOAD, "dominant_payment_type": "crypto"}
        assert client.post("/predict", json=payload).status_code == 200

    def test_unknown_category(self):
        """Kategori produk baru harus tidak crash."""
        payload = {**SAMPLE_PAYLOAD, "dominant_category": "new_category_2025"}
        assert client.post("/predict", json=payload).status_code == 200

    def test_high_value_customer(self):
        """Customer dengan nilai ekstrem tinggi harus tidak crash."""
        payload = {
            **SAMPLE_PAYLOAD,
            "monetary": 999_999.0,
            "avg_payment_value": 999_999.0,
            "total_payment_value": 999_999.0,
            "total_price": 999_999.0,
            "max_installments": 24,
            "total_orders": 500,
            "review_count": 500,
        }
        assert client.post("/predict", json=payload).status_code == 200

    def test_customer_lifetime_very_long(self):
        """Customer lifetime 10 tahun harus tidak crash."""
        payload = {**SAMPLE_PAYLOAD, "customer_lifetime_days": 3650}
        assert client.post("/predict", json=payload).status_code == 200


# ---------------------------------------------------------------------------
# Tests: /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    @pytest.fixture(autouse=True)
    def mock_deps(self):
        mock = MagicMock()
        mock.predict_proba.return_value = np.array([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]])
        mock_monitor = MagicMock()
        with patch("src.api.main._model", mock), \
             patch("src.api.main._feature_cols", MOCK_FEATURE_COLS), \
             patch("src.api.main._threshold", MOCK_THRESHOLD), \
             patch("src.api.main._monitor", mock_monitor):
            yield mock

    def test_batch_predict_returns_200(self):
        payload = {"customers": [SAMPLE_PAYLOAD, SAMPLE_PAYLOAD]}
        assert client.post("/predict/batch", json=payload).status_code == 200

    def test_batch_response_count(self):
        payload = {"customers": [SAMPLE_PAYLOAD] * 3}
        data = client.post("/predict/batch", json=payload).json()
        assert data["total"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_churn_rate_computed(self):
        payload = {"customers": [SAMPLE_PAYLOAD] * 3}
        data = client.post("/predict/batch", json=payload).json()
        assert 0.0 <= data["churn_rate"] <= 1.0

    def test_batch_vectorized_single_predict_proba_call(self, mock_deps):
        """predict_proba harus dipanggil SATU kali untuk seluruh batch (vectorized)."""
        payload = {"customers": [SAMPLE_PAYLOAD, SAMPLE_PAYLOAD, SAMPLE_PAYLOAD]}
        client.post("/predict/batch", json=payload)
        assert mock_deps.predict_proba.call_count == 1

    def test_batch_empty_returns_422(self):
        assert client.post("/predict/batch", json={"customers": []}).status_code == 422

    def test_batch_monitor_log_called_per_customer(self):
        mock_monitor = MagicMock()
        n = 3
        with patch("src.api.main._monitor", mock_monitor):
            payload = {"customers": [SAMPLE_PAYLOAD] * n}
            client.post("/predict/batch", json=payload)
            assert mock_monitor.log.call_count == n

    def test_batch_1000_customers(self):
        """Batch 1000 customer harus tidak crash dan return 1000 predictions."""
        mock = MagicMock()
        mock.predict_proba.return_value = np.column_stack([
            np.random.rand(1000),
            np.random.rand(1000),
        ])
        with patch("src.api.main._model", mock), \
             patch("src.api.main._feature_cols", MOCK_FEATURE_COLS), \
             patch("src.api.main._threshold", MOCK_THRESHOLD), \
             patch("src.api.main._monitor", MagicMock()):
            payload = {"customers": [SAMPLE_PAYLOAD] * 1000}
            response = client.post("/predict/batch", json=payload)
            assert response.status_code == 200
            assert response.json()["total"] == 1000
            assert mock.predict_proba.call_count == 1


# ---------------------------------------------------------------------------
# Tests: Rate Limiting
# ---------------------------------------------------------------------------

class TestRateLimit:
    @pytest.fixture(autouse=True)
    def mock_deps(self):
        mock = MagicMock()
        mock.predict_proba.return_value = np.array([[0.3, 0.7], [0.3, 0.7]])
        with patch("src.api.main._model", mock), \
             patch("src.api.main._feature_cols", MOCK_FEATURE_COLS), \
             patch("src.api.main._threshold", MOCK_THRESHOLD), \
             patch("src.api.main._monitor", MagicMock()):
            yield

    def test_batch_returns_429_when_rate_exceeded(self):
        """Setelah BATCH_RATE_LIMIT request, endpoint harus return 429."""
        import src.api.main as m

        m._rate_store.clear()

        with patch.object(m, "BATCH_RATE_LIMIT", 2):
            payload = {"customers": [SAMPLE_PAYLOAD, SAMPLE_PAYLOAD]}
            r1 = client.post("/predict/batch", json=payload)
            r2 = client.post("/predict/batch", json=payload)
            r3 = client.post("/predict/batch", json=payload)

            assert r1.status_code == 200
            assert r2.status_code == 200
            assert r3.status_code == 429
            assert "Too Many Requests" in r3.json()["detail"]

        m._rate_store.clear()


# ---------------------------------------------------------------------------
# Tests: /monitoring/drift & /monitoring/recent
# ---------------------------------------------------------------------------

class TestMonitoring:
    @pytest.fixture(autouse=True)
    def mock_monitor(self):
        mock = MagicMock()
        mock.drift_stats.return_value = {
            "window_hours": 24,
            "request_count": 10,
            "feature_means": {"avg_freight": 32.5, "avg_payment_value": 148.0},
            "churn_rate_recent": 0.72,
            "drift_flags": {},
            "flagged_features": [],
        }
        mock.recent_predictions.return_value = [
            {"ts": "2024-01-01T00:00:00", "customer_id": "c1", "churn_probability": 0.85, "is_churn": 1}
        ]
        with patch("src.api.main._monitor", mock):
            yield mock

    def test_drift_endpoint_returns_200(self):
        assert client.get("/monitoring/drift").status_code == 200

    def test_drift_schema(self):
        data = client.get("/monitoring/drift").json()
        assert "window_hours" in data
        assert "request_count" in data
        assert "feature_means" in data
        assert "churn_rate_recent" in data

    def test_drift_custom_window(self):
        assert client.get("/monitoring/drift?window_hours=48").status_code == 200

    def test_drift_window_hours_validation(self):
        assert client.get("/monitoring/drift?window_hours=0").status_code == 422
        assert client.get("/monitoring/drift?window_hours=721").status_code == 422

    def test_recent_endpoint_returns_200(self):
        assert client.get("/monitoring/recent").status_code == 200

    def test_recent_schema(self):
        data = client.get("/monitoring/recent").json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)

    def test_recent_custom_limit(self):
        assert client.get("/monitoring/recent?limit=10").status_code == 200


# ---------------------------------------------------------------------------
# Tests: /predict & /predict/batch — model/feature_cols belum loaded (503)
# ---------------------------------------------------------------------------

class TestPredict503:
    """Cover baris 503 saat model/feature_cols None, dan 500 saat exception."""

    def test_predict_503_when_model_not_loaded(self):
        """Harus return 503 jika model belum di-load."""
        with patch("src.api.main._model", None), \
             patch("src.api.main._feature_cols", MOCK_FEATURE_COLS):
            assert client.post("/predict", json=SAMPLE_PAYLOAD).status_code == 503

    def test_predict_503_when_feature_cols_missing(self):
        """Harus return 503 jika feature_cols belum tersedia."""
        with patch("src.api.main._model", MagicMock()), \
             patch("src.api.main._feature_cols", None):
            assert client.post("/predict", json=SAMPLE_PAYLOAD).status_code == 503

    def test_predict_500_on_unexpected_exception(self):
        """Harus return 500 jika predict_proba melempar exception tak terduga."""
        mock = MagicMock()
        mock.predict_proba.side_effect = RuntimeError("GPU out of memory")
        with patch("src.api.main._model", mock), \
             patch("src.api.main._feature_cols", MOCK_FEATURE_COLS), \
             patch("src.api.main._monitor", MagicMock()):
            response = client.post("/predict", json=SAMPLE_PAYLOAD)
            assert response.status_code == 500
            assert "Prediction failed" in response.json()["detail"]


class TestPredictBatch503:
    """Cover 503 saat model/feature_cols None, dan 500 saat exception."""

    def test_batch_503_when_model_not_loaded(self):
        with patch("src.api.main._model", None), \
             patch("src.api.main._feature_cols", MOCK_FEATURE_COLS):
            payload = {"customers": [SAMPLE_PAYLOAD]}
            assert client.post("/predict/batch", json=payload).status_code == 503

    def test_batch_503_when_feature_cols_missing(self):
        with patch("src.api.main._model", MagicMock()), \
             patch("src.api.main._feature_cols", None):
            payload = {"customers": [SAMPLE_PAYLOAD]}
            assert client.post("/predict/batch", json=payload).status_code == 503

    def test_batch_500_on_unexpected_exception(self):
        """Harus return 500 jika predict_proba melempar exception."""
        mock = MagicMock()
        mock.predict_proba.side_effect = ValueError("shape mismatch")
        with patch("src.api.main._model", mock), \
             patch("src.api.main._feature_cols", MOCK_FEATURE_COLS), \
             patch("src.api.main._monitor", MagicMock()):
            payload = {"customers": [SAMPLE_PAYLOAD]}
            response = client.post("/predict/batch", json=payload)
            assert response.status_code == 500
            assert "Batch prediction failed" in response.json()["detail"]


class TestMonitoring503:
    """Cover 503 saat monitor None, 500 saat error dict."""

    def test_drift_503_when_monitor_none(self):
        with patch("src.api.main._monitor", None):
            assert client.get("/monitoring/drift").status_code == 503

    def test_recent_503_when_monitor_none(self):
        with patch("src.api.main._monitor", None):
            assert client.get("/monitoring/recent").status_code == 503

    def test_drift_500_when_monitor_returns_error(self):
        """drift_stats() bisa return {"error": "..."} — harus diubah ke 500."""
        mock = MagicMock()
        mock.drift_stats.return_value = {"error": "database locked"}
        with patch("src.api.main._monitor", mock):
            response = client.get("/monitoring/drift")
            assert response.status_code == 500
            assert "database locked" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Tests: OHE columns di _request_to_dataframe
# ---------------------------------------------------------------------------

class TestOHEColumns:
    """Verifikasi bahwa OHE untuk state/payment_type/category berjalan benar."""

    @pytest.fixture(autouse=True)
    def mock_deps(self):
        mock = MagicMock()
        mock.predict_proba.return_value = np.array([[0.3, 0.7]])
        feature_cols = [
            "avg_freight", "avg_payment_value", "monetary", "frequency",
            "log_monetary", "log_frequency",
            "customer_state_SP", "customer_state_RJ",
            "dominant_payment_type_credit_card", "dominant_payment_type_boleto",
            "dominant_category_health_beauty", "dominant_category_toys",
        ]
        with patch("src.api.main._model", mock), \
             patch("src.api.main._feature_cols", feature_cols), \
             patch("src.api.main._threshold", MOCK_THRESHOLD), \
             patch("src.api.main._monitor", MagicMock()):
            yield mock

    def test_known_state_encoded_as_1(self):
        """customer_state='SP' harus encode customer_state_SP=1."""
        payload = {**SAMPLE_PAYLOAD, "customer_state": "SP"}
        assert client.post("/predict", json=payload).status_code == 200

    def test_known_payment_type_encoded(self):
        """dominant_payment_type='boleto' harus encode dominant_payment_type_boleto=1."""
        payload = {**SAMPLE_PAYLOAD, "dominant_payment_type": "boleto"}
        assert client.post("/predict", json=payload).status_code == 200

    def test_known_category_encoded(self):
        """dominant_category='toys' harus encode dominant_category_toys=1."""
        payload = {**SAMPLE_PAYLOAD, "dominant_category": "toys"}
        assert client.post("/predict", json=payload).status_code == 200


# ---------------------------------------------------------------------------
# Tests: CORS — konfigurasi via env var CORS_ORIGINS (changelog v1.3.0)
# ---------------------------------------------------------------------------

class TestCORS:
    """
    Verifikasi konfigurasi CORS sesuai changelog v1.3.0:
    allow_origins dibaca dari env var, bukan hardcode '*'.
    """

    def test_cors_wildcard_not_default(self):
        """'*' tidak boleh ada di CORS_ORIGINS default — ini security risk."""
        import src.api.main as m
        assert "*" not in m.CORS_ORIGINS, (
            "CORS default tidak boleh '*'. Set env var CORS_ORIGINS=* hanya untuk dev."
        )

    def test_cors_default_includes_streamlit_localhost(self):
        """Default CORS harus izinkan localhost:8501 (Streamlit local dev)."""
        import src.api.main as m
        assert any("localhost:8501" in o or "127.0.0.1:8501" in o for o in m.CORS_ORIGINS)

    def test_cors_env_var_parsed_correctly(self):
        """CORS_ORIGINS env var comma-separated harus di-parse jadi list tanpa spasi."""
        raw = "https://a.com, https://b.com ,https://c.com"
        parsed = [o.strip() for o in raw.split(",") if o.strip()]
        assert parsed == ["https://a.com", "https://b.com", "https://c.com"]

    def test_cors_env_var_empty_falls_back_to_default(self):
        """Env var kosong harus pakai default Streamlit local, bukan list kosong."""
        import src.api.main as m
        assert len(m.CORS_ORIGINS) > 0

    def test_cors_rejects_unlisted_origin_no_header(self):
        """Origin asing tidak boleh dapat Access-Control-Allow-Origin."""
        response = client.options(
            "/predict",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        allow_origin = response.headers.get("access-control-allow-origin", "")
        assert "evil.example.com" not in allow_origin

    def test_cors_configured_origin_gets_header(self):
        """Origin yang ada di CORS_ORIGINS harus mendapat Access-Control-Allow-Origin."""
        import src.api.main as m
        if not m.CORS_ORIGINS:
            pytest.skip("CORS_ORIGINS kosong")
        origin = m.CORS_ORIGINS[0]
        response = client.options(
            "/predict",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )
        allow_origin = response.headers.get("access-control-allow-origin", "")
        assert allow_origin == origin or allow_origin == "*"