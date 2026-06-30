"""Validation helpers for Streamlit upload workflows."""

from __future__ import annotations

import pandas as pd

from src.config import (
    COL_DELIVERY_LAT,
    COL_DELIVERY_LON,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
    REQUIRED_COLUMNS,
)


def validate_upload_dataframe(df: pd.DataFrame) -> list[str]:
    """Validate uploaded dataframe and return a list of user-facing errors."""
    errors: list[str] = []

    if df.empty:
        errors.append("Uploaded file is empty.")
        return errors

    missing_cols = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return errors

    lat_lon_cols = [COL_RESTAURANT_LAT, COL_RESTAURANT_LON, COL_DELIVERY_LAT, COL_DELIVERY_LON]
    for column in lat_lon_cols:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.isna().all():
            errors.append(f"Column '{column}' has no valid numeric coordinates.")

    if errors:
        return errors

    rest_lat = pd.to_numeric(df[COL_RESTAURANT_LAT], errors="coerce")
    rest_lon = pd.to_numeric(df[COL_RESTAURANT_LON], errors="coerce")
    drop_lat = pd.to_numeric(df[COL_DELIVERY_LAT], errors="coerce")
    drop_lon = pd.to_numeric(df[COL_DELIVERY_LON], errors="coerce")

    invalid_global = (
        (rest_lat < -90)
        | (rest_lat > 90)
        | (drop_lat < -90)
        | (drop_lat > 90)
        | (rest_lon < -180)
        | (rest_lon > 180)
        | (drop_lon < -180)
        | (drop_lon > 180)
    )

    if bool(invalid_global.fillna(False).any()):
        errors.append("File contains coordinates outside global lat/lon bounds.")

    return errors
