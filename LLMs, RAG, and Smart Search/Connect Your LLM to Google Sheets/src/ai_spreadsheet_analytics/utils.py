"""General helpers."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def utc_timestamp() -> str:
    """Return UTC timestamp string."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def stable_df_hash(df: pd.DataFrame) -> str:
    """Generate stable hash for dataframe contents."""
    normalized = df.fillna("<NA>").astype(str)
    payload = "\n".join("|".join(row) for row in normalized.to_numpy())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ensure_parent(path: Path) -> None:
    """Ensure parent path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any) -> float | None:
    """Convert value to float when possible."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
