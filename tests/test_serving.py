"""Test modul scoring v2 bersama (src/serving/repeat.py) yang dipakai API dan dashboard."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import main as api
from src.features.snapshot import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.serving import repeat

SEGMENTS = [
    {"segment": "top1pct", "min_score": 0.02, "lift": 6.4, "precision": 0.036},
    {"segment": "top5pct", "min_score": 0.01, "lift": 1.9, "precision": 0.011},
    {"segment": "top10pct", "min_score": 0.008, "lift": 1.7, "precision": 0.0095},
]
ROW = {"customer_id": "c1", "recency_days": 20, "frequency": 2, "monetary": 250.0, "tenure_days": 180}
MODEL_FILE = Path("outputs/model_snapshot_rfm_logreg.pkl")
METRICS_FILE = Path("outputs/metrics_snapshot.json")


def _mock_model(probas):
    m = MagicMock()
    m.predict_proba.side_effect = lambda X: np.column_stack([1 - np.array(probas[:len(X)]),
                                                             probas[:len(X)]])
    return m


class TestBuildFrame:
    def test_columns_match_training_schema(self):
        df = repeat.build_frame([ROW])
        assert list(df.columns) == NUMERIC_FEATURES + CATEGORICAL_FEATURES

    def test_customer_id_dropped_and_unused_features_filled(self):
        df = repeat.build_frame([ROW])
        assert "customer_id" not in df.columns
        assert df.loc[0, "recency_days"] == 20
        assert df[CATEGORICAL_FEATURES].eq("unknown").all().all()
        unused = [c for c in NUMERIC_FEATURES if c not in ROW]
        assert df[unused].isna().all().all()


class TestAssignSegment:
    @pytest.mark.parametrize("score,expected", [
        (0.5, "top1pct"), (0.02, "top1pct"), (0.015, "top5pct"),
        (0.009, "top10pct"), (0.001, "rest"),
    ])
    def test_narrowest_segment_wins(self, score, expected):
        assert repeat.assign_segment(score, SEGMENTS)["segment"] == expected

    def test_no_segments_means_rest(self):
        assert repeat.assign_segment(0.9, []) == repeat.REST


class TestScoreRows:
    def test_output_shape_matches_api_response(self):
        out = repeat.score_rows(_mock_model([0.03]), SEGMENTS, [ROW])
        assert out == [{
            "customer_id": "c1", "repeat_probability": 0.03, "segment": "top1pct",
            "segment_lift": 6.4, "segment_precision": 0.036,
            "model_version": repeat.MODEL_VERSION,
        }]

    def test_rest_has_no_lift(self):
        out = repeat.score_rows(_mock_model([0.001]), SEGMENTS, [ROW])[0]
        assert out["segment"] == "rest" and out["segment_lift"] is None

    def test_probability_rounded_to_six_digits(self):
        out = repeat.score_rows(_mock_model([0.123456789]), SEGMENTS, [ROW])[0]
        assert out["repeat_probability"] == 0.123457


class TestLoadSegments:
    def test_sorted_from_highest_cutoff(self, tmp_path):
        fp = tmp_path / "m.json"
        fp.write_text(json.dumps({"segments": {"rfm_logreg": list(reversed(SEGMENTS))}}))
        assert [s["segment"] for s in repeat.load_segments(fp)] == ["top1pct", "top5pct", "top10pct"]

    def test_unknown_model_gives_empty_list(self, tmp_path):
        fp = tmp_path / "m.json"
        fp.write_text(json.dumps({"segments": {}}))
        assert repeat.load_segments(fp) == []


class TestScorerFromFiles:
    def test_missing_metrics_means_all_rest(self, tmp_path):
        with patch.object(repeat.joblib, "load", return_value=_mock_model([0.9])):
            scorer = repeat.RepeatScorer.from_files(tmp_path / "m.pkl", tmp_path / "missing.json")
        assert scorer.segments == []
        assert scorer.score([ROW])[0]["segment"] == "rest"


@pytest.mark.skipif(not (MODEL_FILE.exists() and METRICS_FILE.exists()),
                    reason="model v2 belum dilatih")
class TestCommittedModel:
    """Model yang di-commit untuk dashboard harus memberi hasil yang sama dengan API."""

    def test_dashboard_and_api_give_identical_results(self):
        scorer = repeat.RepeatScorer.from_files(MODEL_FILE, METRICS_FILE)
        rows = [ROW, {**ROW, "customer_id": "c2", "recency_days": 300, "frequency": 1,
                      "tenure_days": 300, "monetary": 40.0}]
        local = scorer.score(rows)

        with patch.object(api, "_repeat_model", scorer.model), \
             patch.object(api, "_repeat_segments", scorer.segments):
            client = TestClient(api.app)
            r = client.post("/v2/repeat/score/batch", json={"customers": rows})
        assert r.status_code == 200
        assert r.json()["predictions"] == local

    def test_recent_repeat_buyer_scores_higher(self):
        scorer = repeat.RepeatScorer.from_files(MODEL_FILE, METRICS_FILE)
        recent, old = scorer.score([
            ROW, {**ROW, "recency_days": 330, "tenure_days": 330, "frequency": 1},
        ])
        assert recent["repeat_probability"] > old["repeat_probability"]
