"""
train_snapshot_pipeline.py — Training model repeat purchase berbasis snapshot.

Usage:
    python train_snapshot_pipeline.py --raw-path data/raw --output-dir outputs

Desain (lihat src/features/snapshot.py):
    train      : snapshot 2017-06-01, 2017-09-01, 2017-12-01
    validation : snapshot 2018-03-01   (pemilihan hyperparameter)
    test       : snapshot 2018-06-01   (out-of-time; label window s/d 2018-08-30,
                                         data order praktis berhenti akhir Agustus 2018)

Output:
    outputs/model_snapshot_xgb.pkl
    outputs/model_snapshot_rfm_logreg.pkl
    outputs/metrics_snapshot.json
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
from src.models.snapshot_model import train_and_evaluate  # noqa: E402

TRAIN_CUTOFFS = ["2017-06-01", "2017-09-01", "2017-12-01"]
VAL_CUTOFF = "2018-03-01"
TEST_CUTOFF = "2018-06-01"

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

    logger.info("Disimpan: %s", [f"model_snapshot_{n}.pkl" for n in models] + ["metrics_snapshot.json"])


if __name__ == "__main__":
    main()
