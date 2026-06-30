"""Command-line runner for the geospatial clustering pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TRAIN_FILE_PATH  # noqa: E402
from src.pipeline import GeospatialClusteringPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run geospatial clustering pipeline")
    parser.add_argument("--data-path", type=str, default=str(TRAIN_FILE_PATH))
    parser.add_argument(
        "--algorithms",
        nargs="*",
        default=["kmeans", "minibatch_kmeans", "dbscan", "hdbscan", "agglomerative", "gmm"],
    )
    parser.add_argument("--keep-outliers", action="store_true", help="Do not remove outliers before clustering")
    parser.add_argument("--run-automl", action="store_true", help="Run downstream AutoML benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pipeline = GeospatialClusteringPipeline(
        data_path=Path(args.data_path),
        remove_outliers=not args.keep_outliers,
        algorithms=args.algorithms,
        run_downstream_automl=args.run_automl,
    )
    report = pipeline.run()

    logger.info("Pipeline best algorithm: %s", report.best_algorithm)
    logger.info("Artifacts: %s", json.dumps(report.artifact_paths, indent=2))
    logger.info("Reports: %s", json.dumps(report.report_paths, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
