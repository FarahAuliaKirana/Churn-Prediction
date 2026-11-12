"""
train_snapshot_pipeline.py — Training model repeat purchase berbasis snapshot.

Usage:
    python train_snapshot_pipeline.py --raw-path data/raw --output-dir outputs
    python train_snapshot_pipeline.py --backtest          # + backtest 3 periode test

Desain (lihat src/features/snapshot.py):
    train      : snapshot 2017-06-01, 2017-09-01, 2017-12-01
    validation : snapshot 2018-03-01   (pemilihan hyperparameter)
    test       : snapshot 2018-06-01   (out-of-time; label window s/d 2018-08-30,
                                         data order praktis berhenti akhir Agustus 2018)

Backtest (lihat src/models/backtest.py): prosedur yang sama diulang dengan test
snapshot 2017-12, 2018-03, dan 2018-06 (expanding window).

Output:
    outputs/model_snapshot_xgboost.pkl
    outputs/model_snapshot_rfm_logreg.pkl
    outputs/metrics_snapshot.json
    outputs/backtest_snapshot.csv               (dengan --backtest)
    outputs/backtest_snapshot_significance.csv  (dengan --backtest)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).parent))

from src.data.load import build_master  # noqa: E402
from src.features.snapshot import build_snapshots, load_reviews  # noqa: E402
from src.models.backtest import (  # noqa: E402
    QUARTERLY_CUTOFFS,
    build_snapshot_cache,
    make_folds,
    run_backtest,
    summarize_backtest,
)
from src.models.snapshot_model import train_and_evaluate  # noqa: E402

TRAIN_CUTOFFS = ["2017-06-01", "2017-09-01", "2017-12-01"]
VAL_CUTOFF = "2018-03-01"
TEST_CUTOFF = "2018-06-01"
DEMO_SAMPLE_SIZE = 5000
DEMO_FILE = "demo_customers_2018-06.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Olist Repeat Purchase — Snapshot Pipeline")
    parser.add_argument("--raw-path", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--backtest", action="store_true",
                        help="Jalankan juga backtest di beberapa periode test")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    master = build_master(args.raw_path)
    reviews = load_reviews(args.raw_path)

    train_df = build_snapshots(master, TRAIN_CUTOFFS, reviews=reviews)
    val_df = build_snapshots(master, [VAL_CUTOFF], reviews=reviews)
    test_df = build_snapshots(master, [TEST_CUTOFF], reviews=reviews)

    models, metrics, _ = train_and_evaluate(train_df, val_df, test_df, n_boot=args.n_boot)

    for name, model in models.items():
        joblib.dump(model, out / f"model_snapshot_{name}.pkl")
    with open(out / "metrics_snapshot.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Sampel customer asli dari test snapshot (belum pernah dilihat model) untuk mode demo
    # dashboard. Hanya fitur RFM + hasil aktual; customer_unique_id Olist sudah anonim.
    demo = test_df.sample(n=min(DEMO_SAMPLE_SIZE, len(test_df)), random_state=42)
    demo = demo.rename(columns={"customer_unique_id": "customer_id",
                                "is_repeat": "ordered_again_90d"})
    demo[["customer_id", "recency_days", "frequency", "monetary", "tenure_days",
          "ordered_again_90d"]].to_csv(out / DEMO_FILE, index=False)

    logger.info("Disimpan: %s", [f"model_snapshot_{n}.pkl" for n in models]
                + ["metrics_snapshot.json", DEMO_FILE])

    if args.backtest:
        snapshots = build_snapshot_cache(master, QUARTERLY_CUTOFFS, reviews=reviews)
        fold_metrics, significance = run_backtest(snapshots, make_folds(), n_boot=args.n_boot)
        fold_metrics.to_csv(out / "backtest_snapshot.csv", index=False)
        significance.to_csv(out / "backtest_snapshot_significance.csv", index=False)
        logger.info("Ringkasan backtest:\n%s", summarize_backtest(fold_metrics).to_string())
        logger.info("Disimpan: backtest_snapshot.csv, backtest_snapshot_significance.csv")


if __name__ == "__main__":
    main()
