"""Data loading and splitting utilities for Ames Housing training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_api.training.feature_spec import ALL_FEATURES, TARGET_COLUMN


@dataclass(frozen=True)
class SplitData:
    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


@dataclass(frozen=True)
class DatasetBundle:
    frame: pd.DataFrame
    dataset_hash: str
    split: SplitData


def load_dataset(path: Path, random_seed: int) -> DatasetBundle:
    """Load dataset and return deterministic train/val/test splits."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    _validate_columns(frame)

    csv_hash = hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    x = frame[ALL_FEATURES].copy()
    y = frame[TARGET_COLUMN].copy()

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.4,
        random_state=random_seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=random_seed,
    )

    split = SplitData(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
    return DatasetBundle(frame=frame, dataset_hash=csv_hash, split=split)


def _validate_columns(frame: pd.DataFrame) -> None:
    expected = set(ALL_FEATURES + [TARGET_COLUMN])
    missing = sorted(expected.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    null_counts = frame[ALL_FEATURES + [TARGET_COLUMN]].isna().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"Dataset contains null values in required columns: {bad.to_dict()}")
