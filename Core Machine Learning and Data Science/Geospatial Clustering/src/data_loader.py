"""Data loading, profiling hooks, and cleaning routines.

The project intentionally keeps raw ingestion and deterministic cleaning in one
module so notebooks, tests, Streamlit, and pipeline runs use identical logic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import (
    COL_AGE,
    COL_DELIVERY_LAT,
    COL_DELIVERY_LON,
    COL_MULTI_DELIVERY,
    COL_ORDER_DATE,
    COL_RATINGS,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
    COL_TIME_ORDERED,
    COL_TIME_PICKED,
    COL_TIME_TAKEN,
    COL_VEHICLE_COND,
    COL_WEATHER,
    INDIA_BOUNDS,
    REQUIRED_COLUMNS,
    TRAIN_FILE_PATH,
)

logger = logging.getLogger(__name__)


@dataclass
class CleaningSummary:
    """Structured cleaning metadata attached to each cleaned dataframe."""

    n_raw_rows: int
    n_clean_rows: int
    dropped_missing_critical: int
    dropped_out_of_bounds: int
    dropped_zero_coordinates: int
    invalid_date_rows: int


def load_raw_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load raw CSV as-is.

    Args:
        path: Optional explicit CSV path. Defaults to project train file.

    Returns:
        Raw dataframe with no mutation.
    """
    csv_path = Path(path) if path is not None else TRAIN_FILE_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {csv_path}")

    logger.info("Loading raw dataset from %s", csv_path)
    return pd.read_csv(csv_path, low_memory=False)


def load_and_clean_data(
    path: Optional[str | Path] = None,
    *,
    validate: bool = True,
    drop_out_of_bounds: bool = True,
) -> pd.DataFrame:
    """Load dataset and apply deterministic production-safe cleaning.

    Cleaning policy:
    1. Standardize whitespace and common null placeholders.
    2. Parse date/time and convert numeric-like columns.
    3. Remove rows with missing critical fields.
    4. Remove impossible or out-of-country coordinate rows.

    Args:
        path: Optional CSV path.
        validate: If True, attach cleaning report in ``df.attrs``.
        drop_out_of_bounds: If True, remove coordinates outside India bounds.

    Returns:
        Cleaned dataframe ready for feature engineering.
    """
    df = load_raw_data(path)
    n_raw = len(df)

    _ensure_required_columns(df)
    _strip_and_normalize_strings(df)

    invalid_date_rows = _parse_dates(df)
    _parse_times(df)
    _clean_weather(df)
    _clean_time_taken(df)
    _cast_numeric_columns(df)

    critical_cols = [
        COL_AGE,
        COL_RATINGS,
        COL_RESTAURANT_LAT,
        COL_RESTAURANT_LON,
        COL_DELIVERY_LAT,
        COL_DELIVERY_LON,
        COL_ORDER_DATE,
        COL_TIME_TAKEN,
    ]
    before_missing_drop = len(df)
    df = df.dropna(subset=critical_cols).copy()
    dropped_missing_critical = before_missing_drop - len(df)

    before_zero_drop = len(df)
    df = _drop_zero_coordinate_rows(df)
    dropped_zero_coordinates = before_zero_drop - len(df)

    dropped_out_of_bounds = 0
    if drop_out_of_bounds:
        before_bounds_drop = len(df)
        df = _filter_india_bounds(df)
        dropped_out_of_bounds = before_bounds_drop - len(df)

    df = df.reset_index(drop=True)

    if validate:
        summary = CleaningSummary(
            n_raw_rows=n_raw,
            n_clean_rows=len(df),
            dropped_missing_critical=dropped_missing_critical,
            dropped_out_of_bounds=dropped_out_of_bounds,
            dropped_zero_coordinates=dropped_zero_coordinates,
            invalid_date_rows=invalid_date_rows,
        )
        df.attrs["validation_report"] = {
            "n_raw_rows": summary.n_raw_rows,
            "n_clean_rows": summary.n_clean_rows,
            "rows_removed_total": summary.n_raw_rows - summary.n_clean_rows,
            "steps": {
                "dropped_missing_critical": summary.dropped_missing_critical,
                "dropped_zero_coordinates": summary.dropped_zero_coordinates,
                "dropped_out_of_bounds": summary.dropped_out_of_bounds,
                "invalid_date_rows": summary.invalid_date_rows,
            },
        }

    logger.info(
        "Data cleaning complete: %d -> %d rows (removed=%d)",
        n_raw,
        len(df),
        n_raw - len(df),
    )
    return df


def explain_dataset_fields() -> dict[str, str]:
    """Return beginner-friendly explanations for key dataset fields."""
    return {
        "ID": "Unique order identifier.",
        "Delivery_person_ID": "Unique delivery partner identifier.",
        "Delivery_person_Age": "Age of delivery partner in years.",
        "Delivery_person_Ratings": "Historical rating score of delivery partner.",
        "Restaurant_latitude": "Latitude of restaurant pickup location.",
        "Restaurant_longitude": "Longitude of restaurant pickup location.",
        "Delivery_location_latitude": "Latitude of customer drop location.",
        "Delivery_location_longitude": "Longitude of customer drop location.",
        "Order_Date": "Date when the order was placed.",
        "Time_Orderd": "Clock time when order was placed.",
        "Time_Order_picked": "Clock time when order was picked from restaurant.",
        "Weatherconditions": "Weather context during delivery.",
        "Road_traffic_density": "Traffic congestion level.",
        "Vehicle_condition": "Vehicle quality/condition score.",
        "Type_of_order": "Food order category.",
        "Type_of_vehicle": "Delivery vehicle type.",
        "multiple_deliveries": "How many concurrent deliveries were carried.",
        "Festival": "Whether it was a festival day.",
        "City": "City category (urban/metropolitan/semi-urban).",
        "Time_taken(min)": "Observed delivery completion time in minutes.",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def _strip_and_normalize_strings(df: pd.DataFrame) -> None:
    """Normalize object columns by trimming whitespace and null tokens."""
    object_cols = df.select_dtypes(include=["object", "string"]).columns
    null_tokens = {"", "nan", "none", "null", "na", "n/a"}

    for col in object_cols:
        normalized = df[col].astype(str).str.strip()
        normalized = normalized.mask(normalized.str.lower().isin(null_tokens), other=pd.NA)
        df[col] = normalized


def _parse_dates(df: pd.DataFrame) -> int:
    """Parse order date and return number of invalid parsed rows."""
    parsed = pd.to_datetime(df[COL_ORDER_DATE], format="%d-%m-%Y", errors="coerce")
    invalid = int(parsed.isna().sum())
    df[COL_ORDER_DATE] = parsed
    return invalid


def _parse_times(df: pd.DataFrame) -> None:
    """Parse time columns while preserving missing values for downstream fallback."""
    for col in [COL_TIME_ORDERED, COL_TIME_PICKED]:
        parsed = pd.to_datetime(df[col], format="%H:%M:%S", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(df[col], format="%H:%M", errors="coerce")
        df[col] = parsed.dt.strftime("%H:%M:%S")


def _clean_weather(df: pd.DataFrame) -> None:
    if COL_WEATHER in df.columns:
        df[COL_WEATHER] = (
            df[COL_WEATHER]
            .astype(str)
            .str.replace(r"^conditions\s+", "", regex=True)
            .str.strip()
        )


def _clean_time_taken(df: pd.DataFrame) -> None:
    cleaned = (
        df[COL_TIME_TAKEN]
        .astype(str)
        .str.replace(r"^\(min\)\s*", "", regex=True)
        .str.strip()
    )
    df[COL_TIME_TAKEN] = pd.to_numeric(cleaned, errors="coerce")


def _cast_numeric_columns(df: pd.DataFrame) -> None:
    numeric_columns = [
        COL_AGE,
        COL_RATINGS,
        COL_VEHICLE_COND,
        COL_MULTI_DELIVERY,
        COL_RESTAURANT_LAT,
        COL_RESTAURANT_LON,
        COL_DELIVERY_LAT,
        COL_DELIVERY_LON,
        COL_TIME_TAKEN,
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _drop_zero_coordinate_rows(df: pd.DataFrame) -> pd.DataFrame:
    zero_mask = (
        (df[COL_RESTAURANT_LAT] == 0)
        | (df[COL_RESTAURANT_LON] == 0)
        | (df[COL_DELIVERY_LAT] == 0)
        | (df[COL_DELIVERY_LON] == 0)
    )
    return df.loc[~zero_mask].copy()


def _filter_india_bounds(df: pd.DataFrame) -> pd.DataFrame:
    bounds = INDIA_BOUNDS
    mask = (
        (df[COL_RESTAURANT_LAT] >= bounds["lat_min"])
        & (df[COL_RESTAURANT_LAT] <= bounds["lat_max"])
        & (df[COL_RESTAURANT_LON] >= bounds["lon_min"])
        & (df[COL_RESTAURANT_LON] <= bounds["lon_max"])
        & (df[COL_DELIVERY_LAT] >= bounds["lat_min"])
        & (df[COL_DELIVERY_LAT] <= bounds["lat_max"])
        & (df[COL_DELIVERY_LON] >= bounds["lon_min"])
        & (df[COL_DELIVERY_LON] <= bounds["lon_max"])
    )
    return df.loc[mask].copy()
