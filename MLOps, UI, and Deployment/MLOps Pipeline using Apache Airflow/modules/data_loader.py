"""Data loading/saving helpers for raw, processed, and artifact datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .settings import ensure_parent, get_project_root, load_config, resolve_path


def load_csv(path: str | Path, parse_dates: bool = True) -> pd.DataFrame:
    """Load CSV file with optional Date parsing."""
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    df = pd.read_csv(p)
    if parse_dates and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist DataFrame as CSV and return absolute path."""
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    ensure_parent(p)
    df.to_csv(p, index=False)
    return p


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load parquet file."""
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")
    return pd.read_parquet(p)


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    """Persist DataFrame as parquet and return absolute path."""
    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    ensure_parent(p)
    df.to_parquet(p, index=False)
    return p


def load_json(path: str | Path) -> dict[str, Any]:
    """Load JSON file as dictionary."""
    import json

    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Save dictionary as JSON."""
    import json

    p = Path(path)
    if not p.is_absolute():
        p = get_project_root() / p
    ensure_parent(p)
    p.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return p


def load_dataset(phase: str, config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load dataset by logical phase key.

    Supported phases:
    - raw
    - synthetic
    - validated
    - features
    - train
    - test
    """
    config = config or load_config()
    phase_to_key = {
        "raw": "data_raw",
        "synthetic": "data_synthetic",
        "validated": "data_validated",
        "features": "data_features",
        "train": "data_train",
        "test": "data_test",
    }
    if phase not in phase_to_key:
        raise ValueError(f"Unsupported phase '{phase}'. Options: {sorted(phase_to_key)}")

    path = resolve_path(config, phase_to_key[phase])
    if path.suffix == ".csv":
        return load_csv(path)
    if path.suffix == ".parquet":
        return load_parquet(path)
    raise ValueError(f"Unsupported file extension for {path}")


def save_dataset(df: pd.DataFrame, phase: str, config: dict[str, Any] | None = None) -> Path:
    """Save dataset by logical phase key."""
    config = config or load_config()
    phase_to_key = {
        "synthetic": "data_synthetic",
        "validated": "data_validated",
        "features": "data_features",
        "train": "data_train",
        "test": "data_test",
    }
    if phase not in phase_to_key:
        raise ValueError(f"Unsupported save phase '{phase}'. Options: {sorted(phase_to_key)}")

    path = resolve_path(config, phase_to_key[phase])
    if path.suffix == ".csv":
        return save_csv(df, path)
    if path.suffix == ".parquet":
        return save_parquet(df, path)
    raise ValueError(f"Unsupported file extension for {path}")
