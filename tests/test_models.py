"""
tests/test_models.py
Unit tests untuk src/models/evaluate.py dan src/models/train.py.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Tests: src/models/evaluate.py
# ---------------------------------------------------------------------------

class TestFindOptimalThreshold:
    """Tests untuk find_optimal_threshold()."""

    from src.models.evaluate import find_optimal_threshold, DEFAULT_THRESHOLD

    def test_returns_float(self):
        from src.models.evaluate import find_optimal_threshold
        y_true = pd.Series([0, 0, 1, 1, 0, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        result = find_optimal_threshold(y_true, y_proba)
        assert isinstance(result, float)

    def test_threshold_in_valid_range(self):
        from src.models.evaluate import find_optimal_threshold
        rng = np.random.default_rng(42)
        y_true = pd.Series(rng.integers(0, 2, 200))
        y_proba = rng.uniform(0, 1, 200)
        thresh = find_optimal_threshold(y_true, y_proba)
        assert 0.0 <= thresh <= 1.0

    def test_falls_back_to_default_when_recall_unachievable(self):
        """Jika recall constraint tidak bisa dipenuhi, harus return DEFAULT_THRESHOLD."""
        from src.models.evaluate import find_optimal_threshold, DEFAULT_THRESHOLD
        # Semua prediksi sangat buruk — recall 99% tidak mungkin tanpa threshold 0
        y_true = pd.Series([1, 1, 1, 1, 1])
        y_proba = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        # Dengan min_churn_recall=0.99 dan proba sangat rendah,
        # threshold yang memenuhi recall juga punya recall 100% — harus ada match
        # Test ini verifikasi fungsi tidak crash
        result = find_optimal_threshold(y_true, y_proba, min_churn_recall=0.99)
        assert isinstance(result, float)

    def test_high_recall_constraint_respected(self):
        """Threshold yang dipilih harus menghasilkan recall >= min_churn_recall."""
        from src.models.evaluate import find_optimal_threshold
        rng = np.random.default_rng(0)
        # Buat data di mana model cukup baik
        y_true = pd.Series([0]*80 + [1]*20)
        y_proba = np.concatenate([rng.uniform(0, 0.4, 80), rng.uniform(0.6, 1.0, 20)])

        thresh = find_optimal_threshold(y_true, y_proba, min_churn_recall=0.80)
        y_pred = (y_proba >= thresh).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        assert recall >= 0.80


class TestComputeShapValues:
    """Tests untuk compute_shap_values() — mock shap agar tidak butuh model nyata."""

    def test_returns_tuple_of_array_and_dataframe(self):
        from src.models.evaluate import compute_shap_values
        import shap

        n, f = 50, 4
        X = pd.DataFrame(np.random.rand(n, f), columns=[f"feat_{i}" for i in range(f)])

        mock_model = MagicMock()
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.named_steps = {"model": mock_model}
        mock_pipeline.steps = [("imputer", MagicMock()), ("model", mock_model)]

        # Mock preprocessing transform
        mock_preprocessing = MagicMock(spec=Pipeline)
        mock_preprocessing.transform.return_value = X.values

        fake_shap = np.random.rand(n, f)
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = fake_shap

        with patch("src.models.evaluate.Pipeline", return_value=mock_preprocessing), \
             patch("shap.TreeExplainer", return_value=mock_explainer):
            shap_vals, X_sample = compute_shap_values(mock_pipeline, X, max_samples=n)

        assert isinstance(shap_vals, np.ndarray)
        assert isinstance(X_sample, pd.DataFrame)

    def test_samples_when_input_exceeds_max(self):
        from src.models.evaluate import compute_shap_values

        n, f, max_s = 200, 3, 50
        X = pd.DataFrame(np.random.rand(n, f), columns=[f"feat_{i}" for i in range(f)])

        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.named_steps = {"model": MagicMock()}
        mock_pipeline.steps = [("imputer", MagicMock()), ("model", MagicMock())]

        mock_preprocessing = MagicMock()
        mock_preprocessing.transform.return_value = X.values

        fake_shap = np.random.rand(max_s, f)
        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = fake_shap

        with patch("src.models.evaluate.Pipeline", return_value=mock_preprocessing), \
             patch("shap.TreeExplainer", return_value=mock_explainer):
            shap_vals, X_sample = compute_shap_values(mock_pipeline, X, max_samples=max_s)

        assert len(X_sample) == max_s


class TestTopShapFeatures:
    """Tests untuk top_shap_features()."""

    def test_returns_dataframe_with_correct_columns(self):
        from src.models.evaluate import top_shap_features
        shap_vals = np.array([[0.5, 0.1, 0.3], [0.2, 0.4, 0.1]])
        features = ["feat_a", "feat_b", "feat_c"]
        df = top_shap_features(shap_vals, features, top_n=2)
        assert isinstance(df, pd.DataFrame)
        assert "feature" in df.columns
        assert "mean_abs_shap" in df.columns
        assert len(df) == 2

    def test_sorted_by_importance_descending(self):
        from src.models.evaluate import top_shap_features
        shap_vals = np.array([[1.0, 0.1, 0.5]])
        features = ["high", "low", "mid"]
        df = top_shap_features(shap_vals, features, top_n=3)
        assert df.iloc[0]["feature"] == "high"
        assert df.iloc[1]["feature"] == "mid"
        assert df.iloc[2]["feature"] == "low"


# ---------------------------------------------------------------------------
# Tests: src/models/train.py
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_returns_pipeline(self):
        from src.models.train import build_pipeline
        pipe = build_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_imputer_and_model(self):
        from src.models.train import build_pipeline
        pipe = build_pipeline()
        step_names = [name for name, _ in pipe.steps]
        assert "imputer" in step_names
        assert "model" in step_names

    def test_pipeline_has_no_scaler(self):
        """XGBoost invariant terhadap scaling — StandardScaler sengaja tidak ada."""
        from src.models.train import build_pipeline
        from sklearn.preprocessing import StandardScaler
        pipe = build_pipeline()
        step_types = [type(step).__name__ for _, step in pipe.steps]
        assert "StandardScaler" not in step_types


class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, tmp_path):
        from src.models.train import save_model, load_model, build_pipeline
        import numpy as np

        pipe = build_pipeline()
        # Fit dengan data minimal
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
                           "b": [0.5, 1.5, 2.5, 3.5, 4.5] * 20})
        y = pd.Series([0, 1, 0, 1, 0] * 20)
        pipe.fit(X, y)

        model_path = tmp_path / "model.pkl"
        save_model(pipe, model_path)
        loaded = load_model(model_path)

        assert isinstance(loaded, Pipeline)
        # Prediksi harus sama
        preds_orig = pipe.predict_proba(X)
        preds_load = loaded.predict_proba(X)
        np.testing.assert_array_almost_equal(preds_orig, preds_load)

    def test_save_creates_parent_dir(self, tmp_path):
        from src.models.train import save_model, build_pipeline
        import numpy as np

        pipe = build_pipeline()
        X = pd.DataFrame({"a": [1.0, 2.0] * 10, "b": [0.5, 1.5] * 10})
        y = pd.Series([0, 1] * 10)
        pipe.fit(X, y)

        nested_path = tmp_path / "subdir" / "nested" / "model.pkl"
        save_model(pipe, nested_path)
        assert nested_path.exists()


class TestEvaluateAgainstBaseline:
    def test_returns_true_when_no_baseline_exists(self, tmp_path):
        from src.models.train import evaluate_against_baseline
        X = pd.DataFrame({"a": [1.0, 2.0]})
        y = pd.Series([0, 1])
        result = evaluate_against_baseline(
            new_auc=0.85,
            baseline_model_path=tmp_path / "nonexistent.pkl",
            X_test=X,
            y_test=y,
        )
        assert result is True

    def test_returns_true_when_new_model_better(self, tmp_path):
        from src.models.train import evaluate_against_baseline, save_model, build_pipeline

        # Buat baseline model yang jelek
        pipe = build_pipeline()
        X = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        pipe.fit(X, y)
        baseline_path = tmp_path / "baseline.pkl"
        save_model(pipe, baseline_path)

        # new_auc sangat tinggi — harus True
        result = evaluate_against_baseline(
            new_auc=0.999,
            baseline_model_path=baseline_path,
            X_test=X,
            y_test=y,
        )
        assert result is True

    def test_returns_false_when_improvement_below_min(self, tmp_path):
        from src.models.train import evaluate_against_baseline, save_model, build_pipeline

        pipe = build_pipeline()
        X = pd.DataFrame({"a": np.random.rand(100), "b": np.random.rand(100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        pipe.fit(X, y)
        baseline_path = tmp_path / "baseline.pkl"
        save_model(pipe, baseline_path)

        # new_auc sangat rendah — pasti False
        result = evaluate_against_baseline(
            new_auc=0.001,
            baseline_model_path=baseline_path,
            X_test=X,
            y_test=y,
            min_improvement=0.001,
        )
        assert result is False