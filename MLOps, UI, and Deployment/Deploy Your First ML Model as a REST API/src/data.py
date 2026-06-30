"""Dataset loading and deterministic train/validation/test splitting."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from src.constants import FEATURE_NAMES, TARGET_NAME


@dataclass(slots=True)
class DatasetSplit:
    """Container holding train/validation/test partitions."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_california_housing() -> tuple[pd.DataFrame, pd.Series]:
    """Load California Housing with feature ordering normalized.

    Returns:
        Tuple of dataframe and target series.
    """
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X = X[FEATURE_NAMES].copy()
    y = y.rename(TARGET_NAME)
    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> DatasetSplit:
    """Split into train/validation/test with deterministic seeds.

    The split is 70% train, 15% validation, 15% test.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
    )

    return DatasetSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
