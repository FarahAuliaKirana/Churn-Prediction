"""
train_pipeline.py  — Entry point CLI untuk training pipeline end-to-end.

Usage:
    python train_pipeline.py --raw-path data/raw --output-dir outputs

Steps yang dijalankan:
    1. Load & merge 7 tabel raw CSV
    2. Label churn (90-day window)
    3. Feature engineering
    4. Train XGBoost + cross-validation
    5. Evaluasi & SHAP
    6. Compare vs baseline model (jika ada)
    7. Simpan model baru jika lebih baik
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

# Pastikan src/ ada di path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.load import build_master
from src.features.engineer import build_features, label_churn, split_X_y
from src.models.evaluate import compute_shap_values, plot_shap_summary, top_shap_features
from src.models.train import evaluate_against_baseline, save_model, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Olist Churn Prediction Training Pipeline")
    parser.add_argument("--raw-path",       default="data/raw",          help="Path ke folder CSV mentah")
    parser.add_argument("--processed-path", default="data/processed",    help="Path output processed data")
    parser.add_argument("--output-dir",     default="outputs",           help="Path output model & figures")
    parser.add_argument("--baseline-model", default=None,               help="Path model baseline (opsional)")
    parser.add_argument("--force-deploy",   action="store_true",         help="Bypass compare, langsung simpan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_path       = Path(args.raw_path)
    processed_path = Path(args.processed_path)
    output_dir     = Path(args.output_dir)
    figures_dir    = output_dir / "figures"

    processed_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load & merge
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Load & merge raw tables")
    logger.info("=" * 60)
    master = build_master(raw_path)
    master.to_csv(processed_path / "master.csv", index=False)
    logger.info("master.csv disimpan: %s", master.shape)

    # ------------------------------------------------------------------
    # Step 2: Churn labeling
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Churn labeling (90-day window)")
    logger.info("=" * 60)
    rfm = label_churn(master)
    rfm.to_csv(processed_path / "rfm_labeled.csv", index=False)
    logger.info("rfm_labeled.csv disimpan: %s", rfm.shape)

    # ------------------------------------------------------------------
    # Step 3: Feature engineering
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Feature engineering")
    logger.info("=" * 60)
    features = build_features(master, rfm)
    features.to_csv(processed_path / "features.csv", index=False)
    logger.info("features.csv disimpan: %s", features.shape)

    # ------------------------------------------------------------------
    # Step 4: Train
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Train XGBoost")
    logger.info("=" * 60)
    X, y = split_X_y(features)
    pipeline, metrics = train(X, y)

    # Simpan metrics
    metrics_path = output_dir / "metrics.json"
    # classification_report tidak bisa di-serialize langsung
    metrics_serializable = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(metrics_path, "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    logger.info("Metrics disimpan: %s", metrics_path)

    # ------------------------------------------------------------------
    # Step 5: SHAP
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: SHAP interpretability")
    logger.info("=" * 60)
    shap_values, X_sample = compute_shap_values(pipeline, X)
    plot_shap_summary(shap_values, X_sample, output_dir=figures_dir)
    top_features = top_shap_features(shap_values, X.columns.tolist())
    top_features.to_csv(output_dir / "shap_top_features.csv", index=False)
    logger.info("Top 10 SHAP features:\n%s", top_features.to_string())

    # ------------------------------------------------------------------
    # Step 6: Compare vs baseline & deploy
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6: Compare vs baseline")
    logger.info("=" * 60)
    model_output_path = output_dir / "model_xgb.pkl"

    if args.force_deploy:
        should_deploy = True
        logger.info("--force-deploy aktif, skip comparison.")
    elif args.baseline_model:
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        pipeline.predict_proba(X_test)[:, 1]
        new_auc = metrics["roc_auc"]
        should_deploy = evaluate_against_baseline(
            new_auc, args.baseline_model, X_test, y_test
        )
    else:
        should_deploy = True
        logger.info("Tidak ada baseline model path — langsung deploy.")

    if should_deploy:
        save_model(pipeline, model_output_path)
        logger.info("✅ Model disimpan ke: %s", model_output_path)

        # Simpan feature_cols.json — wajib ada agar API bisa startup
        feature_cols = X.columns.tolist()
        with open(output_dir / "feature_cols.json", "w") as f:
            json.dump(feature_cols, f)
        logger.info("feature_cols.json disimpan: %d kolom", len(feature_cols))
    else:
        logger.warning("⛔ Model baru tidak di-deploy. Baseline lebih baik.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE SELESAI")
    logger.info("ROC-AUC : %.4f", metrics["roc_auc"])
    logger.info("CV AUC  : %.4f ± %.4f", metrics["cv_auc_mean"], metrics["cv_auc_std"])
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
