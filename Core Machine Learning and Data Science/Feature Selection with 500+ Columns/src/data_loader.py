"""Data loading utilities for project datasets.

This module centralizes dataset access so notebooks and scripts share the same
cache location and metadata contract.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_DIR = PROJECT_ROOT / "data" / "real"

ISOLET_DATA_ID = 44010
ISOLET_CACHE_FILE = REAL_DATA_DIR / "isolet.parquet"
ISOLET_METADATA_FILE = REAL_DATA_DIR / "isolet_metadata.json"


def _ensure_data_dir(base_dir: Path | None = None) -> Path:
    data_dir = base_dir if base_dir is not None else REAL_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def load_isolet_dataset(
    *,
    force_refresh: bool = False,
    data_id: int = ISOLET_DATA_ID,
    base_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Load ISOLET from local cache, downloading once from OpenML when needed.

    Args:
        force_refresh: If True, ignore local cache and redownload.
        data_id: OpenML data id for ISOLET.
        base_dir: Optional override for cached dataset location.

    Returns:
        Tuple of (features, target, metadata).
    """
    data_dir = _ensure_data_dir(base_dir=base_dir)
    cache_file = data_dir / ISOLET_CACHE_FILE.name
    metadata_file = data_dir / ISOLET_METADATA_FILE.name

    if cache_file.exists() and metadata_file.exists() and not force_refresh:
        LOGGER.info("Loading ISOLET from cache: %s", cache_file)
        df = pd.read_parquet(cache_file)
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        X = df.drop(columns=["target"])
        y = df["target"].astype(int)
        return X, y, metadata

    LOGGER.info("Fetching ISOLET from OpenML (data_id=%s)", data_id)
    dataset = fetch_openml(data_id=data_id, as_frame=True, parser="auto", cache=True)

    X = dataset["data"].copy()
    X.columns = [f"f_{idx:03d}" for idx in range(X.shape[1])]
    y = pd.Series(dataset["target"]).astype(int)
    y.name = "target"

    df = X.copy()
    df["target"] = y
    df.to_parquet(cache_file, index=False)

    metadata: dict[str, Any] = {
        "name": "ISOLET",
        "source": "OpenML",
        "data_id": int(data_id),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_classes": int(y.nunique()),
        "missing_values": int(X.isna().sum().sum()),
        "cache_file": str(cache_file),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info(
        "Cached ISOLET to %s with %s rows and %s features",
        cache_file,
        metadata["n_rows"],
        metadata["n_features"],
    )
    return X, y, metadata
