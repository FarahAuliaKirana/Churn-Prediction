"""
tests/test_models.py
Unit tests untuk src/models/evaluate.py dan src/models/train.py.
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Tests: src/models/evaluate.py
# ---------------------------------------------------------------------------

class TestFindOptimalThreshold:
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
        """Jika tidak ada threshold yang memenuhi constraint, harus tidak crash."""
        from src.models.evaluate import find_optimal_threshold
        y_true = pd.Series([1, 1, 1, 1, 1])
        y_proba = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        result = find_optimal_threshold(y_true, y_proba, min_churn_recall=0.99)
        assert isinstance(result, float)

    def test_high_recall_constraint_respected(self):
        """Threshold yang dipilih harus menghasilkan recall >= min_churn_recall."""
        from src.models.evaluate import find_optimal_threshold
        rng = np.random.default_rng(0)
        y_true = pd.Series([0] * 80 + [1] * 20)
        y_proba = np.concatenate([rng.uniform(0, 0.4, 80), rng.uniform(0.6, 1.0, 20)])

        thresh = find_optimal_threshold(y_true, y_proba, min_churn_recall=0.80)
        y_pred = (y_proba >= thresh).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        assert recall >= 0.80


class TestComputeShapValues:
    def test_returns_tuple_of_array_and_dataframe(self):
        from src.models.evaluate import compute_shap_values

        n, f = 50, 4
        X = pd.DataFrame(np.random.rand(n, f), columns=[f"feat_{i}" for i in range(f)])

        mock_model = MagicMock()
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.named_steps = {"model": mock_model}
        mock_pipeline.steps = [("imputer", MagicMock()), ("model", mock_model)]

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
        pipe = build_pipeline()
        step_types = [type(step).__name__ for _, step in pipe.steps]
        assert "StandardScaler" not in step_types


class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, tmp_path):
        from src.models.train import save_model, load_model, build_pipeline

        pipe = build_pipeline()
        X = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0] * 20,
            "b": [0.5, 1.5, 2.5, 3.5, 4.5] * 20,
        })
        y = pd.Series([0, 1, 0, 1, 0] * 20)
        pipe.fit(X, y)

        model_path = tmp_path / "model.pkl"
        save_model(pipe, model_path)
        loaded = load_model(model_path)

        assert isinstance(loaded, Pipeline)
        preds_orig = pipe.predict_proba(X)
        preds_load = loaded.predict_proba(X)
        np.testing.assert_array_almost_equal(preds_orig, preds_load)

    def test_save_creates_parent_dir(self, tmp_path):
        from src.models.train import save_model, build_pipeline

        pipe = build_pipeline()
        X = pd.DataFrame({"a": [1.0, 2.0] * 10, "b": [0.5, 1.5] * 10})
        y = pd.Series([0, 1] * 10)
        pipe.fit(X, y)

        nested_path = tmp_path / "subdir" / "nested" / "model.pkl"
        save_model(pipe, nested_path)
        assert nested_path.exists()


class TestEvaluateAgainstBaseline:
    """
    Mock joblib.load langsung — paling reliable karena evaluate_against_baseline
    memanggil joblib.load(baseline_path) lalu .predict_proba() pada hasilnya.
    Patch di class Pipeline tidak reliable karena instance yang di-load adalah
    objek nyata yang sudah ada sebelum patch diterapkan.
    """

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
        """Baseline AUC di-mock = 0.5 (proba konstan), new_auc = 0.95 → harus True."""
        from src.models.train import evaluate_against_baseline, save_model, build_pipeline

        # Buat file baseline agar path.exists() = True
        pipe = build_pipeline()
        rng = np.random.default_rng(42)
        a = rng.uniform(0, 1, 100)
        X = pd.DataFrame({"a": a, "b": rng.uniform(0, 1, 100)})
        y = pd.Series((a > 0.5).astype(int))
        pipe.fit(X, y)
        baseline_path = tmp_path / "baseline.pkl"
        save_model(pipe, baseline_path)

        # joblib.load di-mock → return mock dengan predict_proba AUC = 0.5
        mock_baseline = MagicMock()
        mock_baseline.predict_proba.return_value = np.column_stack([
            np.full(len(X), 0.5),
            np.full(len(X), 0.5),
        ])

        with patch("src.models.train.joblib.load", return_value=mock_baseline):
            result = evaluate_against_baseline(
                new_auc=0.95,
                baseline_model_path=baseline_path,
                X_test=X,
                y_test=y,
            )
        assert result is True

    def test_returns_false_when_improvement_below_min(self, tmp_path):
        """Baseline AUC di-mock = 1.0 (sempurna), new_auc = 0.001 → harus False."""
        from src.models.train import evaluate_against_baseline, save_model, build_pipeline

        pipe = build_pipeline()
        rng = np.random.default_rng(0)
        a = rng.uniform(0, 1, 100)
        X = pd.DataFrame({"a": a, "b": rng.uniform(0, 1, 100)})
        y = pd.Series((a > 0.5).astype(int))
        pipe.fit(X, y)
        baseline_path = tmp_path / "baseline.pkl"
        save_model(pipe, baseline_path)

        # joblib.load di-mock → baseline predict_proba = label sempurna (AUC = 1.0)
        mock_baseline = MagicMock()
        perfect = y.values.astype(float)
        mock_baseline.predict_proba.return_value = np.column_stack([
            1 - perfect,
            perfect,
        ])

        with patch("src.models.train.joblib.load", return_value=mock_baseline):
            result = evaluate_against_baseline(
                new_auc=0.001,
                baseline_model_path=baseline_path,
                X_test=X,
                y_test=y,
                min_improvement=0.001,
            )
        assert result is False


# ---------------------------------------------------------------------------
# Tests: threshold & ranking reports, class weighting (v1.3.0)
# ---------------------------------------------------------------------------

class TestThresholdSelection:
    def test_majority_positive_does_not_flag_everyone(self):
        """
        Regresi: dulu F1 kelas churn + recall >= 0.99 memilih threshold yang
        menandai hampir semua customer. Youden's J harus memisahkan kelas.
        """
        from src.models.evaluate import find_optimal_threshold
        rng = np.random.default_rng(1)
        y = np.array([1] * 900 + [0] * 100)
        proba = np.concatenate([rng.uniform(0.4, 1.0, 900), rng.uniform(0.0, 0.6, 100)])
        thresh = find_optimal_threshold(y, proba)
        flagged = (proba >= thresh).mean()
        assert flagged < 0.95

    def test_single_class_returns_default(self):
        from src.models.evaluate import DEFAULT_THRESHOLD, find_optimal_threshold
        assert find_optimal_threshold(np.ones(10), np.linspace(0, 1, 10)) == DEFAULT_THRESHOLD


class TestThresholdReport:
    def test_confusion_counts_and_baseline(self):
        from src.models.evaluate import threshold_report
        y = np.array([1, 1, 1, 0])
        proba = np.array([0.9, 0.8, 0.2, 0.1])
        rep = threshold_report(y, proba, threshold=0.5)
        assert (rep["tp"], rep["fn"], rep["fp"], rep["tn"]) == (2, 1, 0, 1)
        assert rep["churn_precision"] == 1.0
        assert rep["non_churn_recall"] == 1.0
        assert rep["baseline_all_churn_precision"] == 0.75
        assert rep["churn_precision_lift"] == pytest.approx(1 / 0.75, abs=1e-3)


class TestRankingReport:
    def test_perfect_ranking_reaches_max_lift(self):
        from src.models.evaluate import ranking_report
        y = np.array([1] * 90 + [0] * 10)
        proba = np.concatenate([np.linspace(1.0, 0.5, 90), np.linspace(0.4, 0.0, 10)])
        rep = ranking_report(y, proba, top_frac=0.1)
        assert rep["top_decile_churn_rate"] == 1.0
        assert rep["top_decile_churn_lift"] == rep["max_possible_churn_lift"]
        assert rep["bottom_decile_return_rate"] == 1.0
        assert rep["bottom_decile_return_lift"] == pytest.approx(10.0)


class TestClassWeighting:
    def test_scale_pos_weight_is_neg_over_pos(self):
        from src.models.train import compute_scale_pos_weight
        y = pd.Series([1] * 90 + [0] * 10)
        assert compute_scale_pos_weight(y) == pytest.approx(10 / 90)

    def test_default_pipeline_has_no_reweighting(self):
        from src.models.train import build_pipeline
        model = build_pipeline().named_steps["model"]
        assert model.get_params()["scale_pos_weight"] == 1.0


class TestTrainSplits:
    def test_threshold_from_val_and_metrics_keys(self):
        from src.models.train import train
        rng = np.random.default_rng(0)
        n = 600
        a = rng.uniform(0, 1, n)
        X = pd.DataFrame({"monetary": a * 100, "avg_freight": rng.uniform(0, 50, n)})
        y = pd.Series((a + rng.normal(0, 0.3, n) > 0.2).astype(int))
        _, metrics = train(X, y, n_cv_folds=3)

        assert metrics["n_train"] + metrics["n_val"] + metrics["n_test"] == n
        assert metrics["threshold_method"] == "youden_j_on_validation"
        for key in ["test_threshold_report", "test_ranking_report", "feature_means_train"]:
            assert key in metrics
        assert set(metrics["feature_means_train"]) == {"monetary", "avg_freight"}
