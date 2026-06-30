"""Dataset quality profiling and readiness checks.

This module provides a reusable quality gate that can be called from notebooks,
pipeline runs, and CI tests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    COL_DELIVERY_LAT,
    COL_DELIVERY_LON,
    COL_ID,
    COL_RESTAURANT_LAT,
    COL_RESTAURANT_LON,
    INDIA_BOUNDS,
    PROFILE_REPORT_PATH,
    REQUIRED_COLUMNS,
)
from src.distance import haversine_vectorized


@dataclass
class QualityGateResult:
    """Structured dataset-quality report."""

    n_rows: int
    n_columns: int
    missing_by_column_pct: dict[str, float]
    duplicate_rows: int
    duplicate_id_rows: int
    coordinate_ranges: dict[str, dict[str, float]]
    rows_outside_global_bounds: int
    rows_outside_india_bounds: int
    rows_with_zero_coordinates: int
    potential_coordinate_swaps: int
    distance_summary_km: dict[str, float]
    schema_pass: bool
    blocking_issues: list[str]
    warnings: list[str]


_NULL_TOKENS = {"", "nan", "none", "null", "na", "n/a"}


def _normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    object_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        s = out[col].astype(str).str.strip()
        out[col] = s.mask(s.str.lower().isin(_NULL_TOKENS), other=pd.NA)
    return out


def run_quality_gate(
    df: pd.DataFrame,
    *,
    required_columns: list[str] | None = None,
    id_column: str = COL_ID,
) -> QualityGateResult:
    """Run schema and quality checks for geospatial food-delivery data."""
    req_cols = required_columns or REQUIRED_COLUMNS

    missing_cols = [c for c in req_cols if c not in df.columns]
    blocking_issues: list[str] = []
    warnings: list[str] = []

    if missing_cols:
        blocking_issues.append(f"Missing required columns: {missing_cols}")

    normalized = _normalize_missing(df)
    missing_pct = (normalized.isna().mean() * 100).sort_values(ascending=False)
    missing_map = {k: float(v) for k, v in missing_pct.items()}

    duplicate_rows = int(df.duplicated().sum())
    duplicate_id_rows = int(df[id_column].duplicated().sum()) if id_column in df.columns else 0

    coord_cols = [COL_RESTAURANT_LAT, COL_RESTAURANT_LON, COL_DELIVERY_LAT, COL_DELIVERY_LON]
    numeric: dict[str, pd.Series] = {}
    coordinate_ranges: dict[str, dict[str, float]] = {}
    for col in coord_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        numeric[col] = values
        coordinate_ranges[col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "median": float(values.median()),
            "na_count": float(values.isna().sum()),
        }

    global_invalid_mask = (
        (numeric[COL_RESTAURANT_LAT] < -90)
        | (numeric[COL_RESTAURANT_LAT] > 90)
        | (numeric[COL_DELIVERY_LAT] < -90)
        | (numeric[COL_DELIVERY_LAT] > 90)
        | (numeric[COL_RESTAURANT_LON] < -180)
        | (numeric[COL_RESTAURANT_LON] > 180)
        | (numeric[COL_DELIVERY_LON] < -180)
        | (numeric[COL_DELIVERY_LON] > 180)
    )
    rows_outside_global_bounds = int(global_invalid_mask.sum())

    bounds = INDIA_BOUNDS
    out_of_india_mask = (
        (numeric[COL_RESTAURANT_LAT] < bounds["lat_min"])
        | (numeric[COL_RESTAURANT_LAT] > bounds["lat_max"])
        | (numeric[COL_RESTAURANT_LON] < bounds["lon_min"])
        | (numeric[COL_RESTAURANT_LON] > bounds["lon_max"])
        | (numeric[COL_DELIVERY_LAT] < bounds["lat_min"])
        | (numeric[COL_DELIVERY_LAT] > bounds["lat_max"])
        | (numeric[COL_DELIVERY_LON] < bounds["lon_min"])
        | (numeric[COL_DELIVERY_LON] > bounds["lon_max"])
    )
    rows_outside_india_bounds = int(out_of_india_mask.sum())

    zero_coordinates = (
        (numeric[COL_RESTAURANT_LAT] == 0)
        | (numeric[COL_RESTAURANT_LON] == 0)
        | (numeric[COL_DELIVERY_LAT] == 0)
        | (numeric[COL_DELIVERY_LON] == 0)
    )
    rows_with_zero_coordinates = int(zero_coordinates.sum())

    # Heuristic: latitude unrealistic for India while longitude looks like latitude magnitude.
    potential_swaps = (
        (numeric[COL_RESTAURANT_LAT].abs() > 40)
        & (numeric[COL_RESTAURANT_LON].abs() < 40)
    ) | (
        (numeric[COL_DELIVERY_LAT].abs() > 40)
        & (numeric[COL_DELIVERY_LON].abs() < 40)
    )
    potential_coordinate_swaps = int(potential_swaps.sum())

    lat_r = numeric[COL_RESTAURANT_LAT].to_numpy(dtype=float)
    lon_r = numeric[COL_RESTAURANT_LON].to_numpy(dtype=float)
    lat_d = numeric[COL_DELIVERY_LAT].to_numpy(dtype=float)
    lon_d = numeric[COL_DELIVERY_LON].to_numpy(dtype=float)
    distances = haversine_vectorized(lat_r, lon_r, lat_d, lon_d)
    finite_distances = distances[np.isfinite(distances)]

    if len(finite_distances) == 0:
        distance_summary = {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
        warnings.append("Distance summary unavailable because all coordinate pairs were invalid.")
    else:
        distance_summary = {
            "p50": float(np.percentile(finite_distances, 50)),
            "p90": float(np.percentile(finite_distances, 90)),
            "p99": float(np.percentile(finite_distances, 99)),
            "max": float(np.max(finite_distances)),
            "mean": float(np.mean(finite_distances)),
        }

    if rows_outside_global_bounds > 0:
        warnings.append(
            f"{rows_outside_global_bounds} rows violate global lat/lon bounds and are invalid coordinates."
        )
    if rows_outside_india_bounds > 0:
        warnings.append(
            f"{rows_outside_india_bounds} rows are outside India bounds and should be cleaned."
        )
    if rows_with_zero_coordinates > 0:
        warnings.append(
            f"{rows_with_zero_coordinates} rows include zero coordinates and may be GPS errors."
        )
    if potential_coordinate_swaps > 0:
        warnings.append(
            f"{potential_coordinate_swaps} rows may contain swapped latitude/longitude values."
        )
    if distance_summary["max"] > 500:
        warnings.append(
            "Extreme delivery distances detected (>500 km); likely geocoding or data-entry anomalies."
        )

    return QualityGateResult(
        n_rows=len(df),
        n_columns=df.shape[1],
        missing_by_column_pct=missing_map,
        duplicate_rows=duplicate_rows,
        duplicate_id_rows=duplicate_id_rows,
        coordinate_ranges=coordinate_ranges,
        rows_outside_global_bounds=rows_outside_global_bounds,
        rows_outside_india_bounds=rows_outside_india_bounds,
        rows_with_zero_coordinates=rows_with_zero_coordinates,
        potential_coordinate_swaps=potential_coordinate_swaps,
        distance_summary_km=distance_summary,
        schema_pass=len(blocking_issues) == 0,
        blocking_issues=blocking_issues,
        warnings=warnings,
    )


def save_quality_report(
    report: QualityGateResult,
    *,
    path: str | Path | None = None,
) -> Path:
    """Save quality-gate report to JSON."""
    out_path = Path(path) if path is not None else PROFILE_REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return out_path


def quality_gate_to_markdown(report: QualityGateResult) -> str:
    """Render compact markdown summary used in notebooks and Streamlit."""
    lines: list[str] = []
    lines.append("## Dataset Quality Gate")
    lines.append(f"- Rows: {report.n_rows:,}")
    lines.append(f"- Columns: {report.n_columns}")
    lines.append(f"- Duplicate rows: {report.duplicate_rows:,}")
    lines.append(f"- Duplicate IDs: {report.duplicate_id_rows:,}")
    lines.append(f"- Global-bound violations: {report.rows_outside_global_bounds:,}")
    lines.append(f"- Out-of-India rows: {report.rows_outside_india_bounds:,}")
    lines.append(f"- Zero-coordinate rows: {report.rows_with_zero_coordinates:,}")
    lines.append(f"- Potential coordinate swaps: {report.potential_coordinate_swaps:,}")
    lines.append(f"- Distance p90 (km): {report.distance_summary_km['p90']:.2f}")

    if report.blocking_issues:
        lines.append("- Blocking issues:")
        for issue in report.blocking_issues:
            lines.append(f"  - {issue}")

    if report.warnings:
        lines.append("- Warnings:")
        for warning in report.warnings[:8]:
            lines.append(f"  - {warning}")

    return "\n".join(lines)
