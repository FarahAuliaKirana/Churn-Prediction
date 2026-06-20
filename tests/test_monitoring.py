"""
tests/test_monitoring.py
Unit tests untuk DriftMonitor.
"""
from unittest.mock import patch

import pytest

from src.monitoring.drift import DriftMonitor


@pytest.fixture
def monitor(tmp_path):
    """DriftMonitor dengan temporary SQLite database."""
    return DriftMonitor(db_path=tmp_path / "test_predictions.db")


SAMPLE_FEATURES = {
    "avg_payment_value": 150.0,
    "total_payment_value": 150.0,
    "avg_installments": 3.0,
    "avg_price": 120.0,
    "avg_freight": 30.0,
    "freight_ratio": 0.20,
    "avg_review_score": 4.0,
    "min_review_score": 4.0,
    "review_gap": 0.0,
    "review_count": 1,
    "total_items": 1,
    "total_orders": 1,
    "avg_order_value": 150.0,
    "frequency": 1,
    "monetary": 150.0,
    "customer_lifetime_days": 0,
    "log_monetary": 5.01,
    "log_frequency": 0.69,
}


class TestDriftMonitor:
    def test_log_and_retrieve(self, monitor):
        """Log satu prediksi dan pastikan bisa diambil kembali."""
        monitor.log(SAMPLE_FEATURES, churn_probability=0.85, is_churn=True, customer_id="c1")
        recent = monitor.recent_predictions(limit=10)
        assert len(recent) == 1
        assert recent[0]["customer_id"] == "c1"

    def test_log_multiple(self, monitor):
        for i in range(5):
            monitor.log(SAMPLE_FEATURES, churn_probability=0.7, is_churn=True, customer_id=f"c{i}")
        recent = monitor.recent_predictions(limit=10)
        assert len(recent) == 5

    def test_drift_stats_empty(self, monitor):
        """Stats saat tidak ada data harus return request_count=0."""
        stats = monitor.drift_stats(window_hours=24)
        assert stats["request_count"] == 0
        assert stats["feature_means"] == {}

    def test_drift_stats_with_data(self, monitor):
        """Stats setelah ada data harus return feature_means yang valid."""
        for _ in range(3):
            monitor.log(SAMPLE_FEATURES, churn_probability=0.85, is_churn=True)
        stats = monitor.drift_stats(window_hours=24)
        assert stats["request_count"] == 3
        assert "avg_freight" in stats["feature_means"]
        assert stats["feature_means"]["avg_freight"] == pytest.approx(30.0, abs=0.01)

    def test_churn_rate_computed(self, monitor):
        """churn_rate_recent harus dihitung dari is_churn."""
        monitor.log(SAMPLE_FEATURES, churn_probability=0.9, is_churn=True)
        monitor.log(SAMPLE_FEATURES, churn_probability=0.3, is_churn=False)
        stats = monitor.drift_stats(window_hours=24)
        assert stats["churn_rate_recent"] == pytest.approx(0.5, abs=0.01)

    def test_log_non_fatal_on_bad_features(self, monitor):
        """Log dengan fitur yang hilang tidak boleh raise Exception."""
        monitor.log({}, churn_probability=0.5, is_churn=False)

    def test_recent_predictions_limit(self, monitor):
        """Limit pada recent_predictions harus dihormati."""
        for i in range(10):
            monitor.log(SAMPLE_FEATURES, churn_probability=0.5, is_churn=False, customer_id=f"c{i}")
        recent = monitor.recent_predictions(limit=3)
        assert len(recent) == 3


# ---------------------------------------------------------------------------
# Tests: DriftMonitor — baseline comparison & error paths
# ---------------------------------------------------------------------------

class TestDriftMonitorBaseline:
    """Cover drift_flags dengan baseline_path, dan error handling paths."""

    @pytest.fixture
    def monitor_with_data(self, tmp_path):
        """Monitor berisi beberapa prediksi untuk drift stats."""
        m = DriftMonitor(db_path=tmp_path / "test.db")
        features = {
            "avg_payment_value": 200.0, "total_payment_value": 200.0,
            "avg_installments": 2.0, "avg_price": 150.0, "avg_freight": 40.0,
            "freight_ratio": 0.2, "avg_review_score": 4.0, "min_review_score": 3.0,
            "review_gap": 1.0, "review_count": 1, "total_items": 1,
            "total_orders": 1, "avg_order_value": 200.0, "frequency": 1,
            "monetary": 200.0, "customer_lifetime_days": 30,
        }
        for _ in range(5):
            m.log(features=features, churn_probability=0.8, is_churn=True)
        return m

    def test_drift_stats_with_baseline_path(self, monitor_with_data, tmp_path):
        """Drift stats harus mengandung drift_flags jika baseline_path valid."""
        import json
        baseline = {
            "feature_means_train": {
                "avg_payment_value": 100.0,  # 200 vs 100 = 100% drift → flagged
                "avg_freight": 38.0,          # ~5% drift → tidak flagged
            }
        }
        baseline_path = tmp_path / "metrics.json"
        baseline_path.write_text(json.dumps(baseline))

        stats = monitor_with_data.drift_stats(
            window_hours=720, baseline_path=str(baseline_path)
        )

        assert "drift_flags" in stats
        assert "flagged_features" in stats
        assert "avg_payment_value" in stats["drift_flags"]
        assert stats["drift_flags"]["avg_payment_value"]["flagged"] is True

    def test_drift_flags_not_flagged_when_within_threshold(self, monitor_with_data, tmp_path):
        """Fitur dengan drift < 20% tidak boleh masuk flagged_features."""
        import json
        baseline = {
            "feature_means_train": {
                "avg_freight": 39.0,  # 40 vs 39 = ~2.5% drift
            }
        }
        baseline_path = tmp_path / "metrics.json"
        baseline_path.write_text(json.dumps(baseline))

        stats = monitor_with_data.drift_stats(
            window_hours=720, baseline_path=str(baseline_path)
        )

        if "flagged_features" in stats:
            assert "avg_freight" not in stats["flagged_features"]

    def test_drift_stats_handles_bad_baseline_gracefully(self, monitor_with_data, tmp_path):
        """Baseline file rusak/tidak valid JSON tidak boleh crash."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("this is not json {{{")

        # Tidak boleh raise — hanya log warning
        stats = monitor_with_data.drift_stats(window_hours=720, baseline_path=str(bad_path))
        assert "window_hours" in stats  # result dasar tetap ada

    def test_drift_stats_db_error_returns_error_dict(self, tmp_path):
        """Jika DB tidak bisa dibaca, harus return dict dengan key 'error'."""
        m = DriftMonitor(db_path=tmp_path / "test.db")

        with patch.object(m, "_connect", side_effect=Exception("DB locked")):
            stats = m.drift_stats(window_hours=24)

        assert "error" in stats

    def test_recent_predictions_returns_empty_on_db_error(self, tmp_path):
        """recent_predictions harus return [] jika DB error, tidak raise."""
        m = DriftMonitor(db_path=tmp_path / "test.db")

        with patch.object(m, "_connect", side_effect=Exception("DB locked")):
            result = m.recent_predictions(limit=10)

        assert result == []

    def test_log_non_fatal_on_db_write_error(self, tmp_path):
        """log() harus tidak raise meskipun DB write gagal."""
        m = DriftMonitor(db_path=tmp_path / "test.db")

        with patch.object(m, "_connect", side_effect=Exception("disk full")):
            # Tidak boleh raise
            m.log(features={"avg_payment_value": 100.0}, churn_probability=0.5,
                  is_churn=False)