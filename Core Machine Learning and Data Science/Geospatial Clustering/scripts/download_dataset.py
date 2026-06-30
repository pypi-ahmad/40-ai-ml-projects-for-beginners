"""Download Food Delivery dataset for this project.

Primary path for this repository workflow:
1. Call Kaggle MCP `download_dataset` to obtain signed URL for `train.csv`.
2. Export that URL as `KAGGLE_MCP_SIGNED_URL`.
3. Run this script to persist file under `data/raw/train.csv`.

Fallback path (if MCP URL is not provided): Kaggle CLI download.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    KAGGLE_DATASET_SLUG,
    KAGGLE_OWNER,
    RAW_DATA_DIR,
    REPORTS_DIR,
    TRAIN_FILE_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _download_from_signed_url(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset file from Kaggle MCP signed URL")
    with urlopen(url) as response:  # nosec B310 - trusted short-lived URL from Kaggle MCP
        total = int(response.headers.get("Content-Length", "0"))
        chunk_size = 1024 * 1024
        downloaded = 0

        with destination.open("wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    pct = (downloaded / total) * 100
                    logger.info("Download progress: %.1f%%", pct)

    logger.info("Saved dataset file to %s", destination)


def _download_with_kaggle_cli(destination: Path) -> None:
    logger.info("Signed URL not found. Falling back to Kaggle CLI.")

    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        f"{KAGGLE_OWNER}/{KAGGLE_DATASET_SLUG}",
        "-p",
        str(RAW_DATA_DIR),
        "--unzip",
    ]

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Kaggle CLI download failed: %s", result.stderr.strip())
        raise RuntimeError("Dataset download failed via Kaggle CLI")

    logger.info("Kaggle CLI download succeeded")
    if not destination.exists():
        raise FileNotFoundError(f"Expected file missing after download: {destination}")


def main() -> int:
    destination = TRAIN_FILE_PATH
    signed_url = os.getenv("KAGGLE_MCP_SIGNED_URL", "").strip()

    start = time.time()
    if signed_url:
        _download_from_signed_url(signed_url, destination)
        source = "kaggle_mcp_signed_url"
    else:
        _download_with_kaggle_cli(destination)
        source = "kaggle_cli"

    metadata = {
        "dataset": f"{KAGGLE_OWNER}/{KAGGLE_DATASET_SLUG}",
        "file": destination.name,
        "download_source": source,
        "downloaded_at_epoch": int(time.time()),
        "elapsed_seconds": round(time.time() - start, 3),
        "output_path": str(destination),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    provenance_path = REPORTS_DIR / "dataset_provenance.json"
    provenance_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote dataset provenance to %s", provenance_path)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logger.exception("Dataset download failed: %s", exc)
        raise SystemExit(1)
