"""Execute all tutorial notebooks end-to-end using nbconvert."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


NOTEBOOK_ORDER = [
    "01_geospatial_foundations.ipynb",
    "02_dataset_profile_and_eda.ipynb",
    "03_distance_engineering.ipynb",
    "04_clustering_algorithms_and_k_selection.ipynb",
    "05_outliers_business_zones_and_interpretability.ipynb",
    "06_advanced_spatial_analysis.ipynb",
    "07_downstream_automl_benchmark.ipynb",
    "08_end_to_end_pipeline_and_streamlit_demo.ipynb",
]


def run_notebook(path: Path) -> None:
    logger.info("Executing notebook: %s", path.name)
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        str(path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Notebook failed: %s", path.name)
        logger.error(result.stderr[-4000:])
        raise RuntimeError(f"Notebook execution failed: {path.name}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    notebooks_dir = project_root / "notebooks"

    for name in NOTEBOOK_ORDER:
        nb_path = notebooks_dir / name
        if not nb_path.exists():
            raise FileNotFoundError(f"Notebook missing: {nb_path}")
        run_notebook(nb_path)

    logger.info("All notebooks executed successfully")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        logger.exception("Notebook execution pipeline failed")
        raise SystemExit(1)
