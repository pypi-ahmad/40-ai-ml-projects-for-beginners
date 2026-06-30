"""Geospatial math and CRS validation helpers.

These checks are used by tests and pipeline verification reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.distance import (
    euclidean_projected_km,
    geodesic_distance,
    haversine_vectorized,
    project_wgs84_to_web_mercator,
)


@dataclass
class DistanceValidationRow:
    """One known-coordinate validation case."""

    name: str
    haversine_km: float
    geodesic_km: float
    euclidean_projected_km: float
    expected_km: float
    tolerance_km: float


@dataclass
class GeospatialValidationReport:
    """Structured geospatial validation report."""

    all_distance_cases_pass: bool
    all_crs_checks_pass: bool
    distance_cases: list[dict[str, Any]]
    crs_checks: dict[str, Any]


KNOWN_DISTANCE_CASES: list[dict[str, Any]] = [
    {
        "name": "Bengaluru-Chennai",
        "p1": (12.9716, 77.5946),
        "p2": (13.0827, 80.2707),
        "expected_km": 290.5,
        "tolerance_km": 3.0,
    },
    {
        "name": "Mumbai-Delhi",
        "p1": (19.0760, 72.8777),
        "p2": (28.6139, 77.2090),
        "expected_km": 1150.0,
        "tolerance_km": 15.0,
    },
    {
        "name": "NYC-London",
        "p1": (40.7128, -74.0060),
        "p2": (51.5074, -0.1278),
        "expected_km": 5570.0,
        "tolerance_km": 30.0,
    },
]


def run_distance_validation() -> tuple[bool, list[DistanceValidationRow]]:
    """Validate distance implementations against known coordinate pairs."""
    rows: list[DistanceValidationRow] = []

    for case in KNOWN_DISTANCE_CASES:
        lat1, lon1 = case["p1"]
        lat2, lon2 = case["p2"]

        hav = float(
            haversine_vectorized(
                np.array([lat1]),
                np.array([lon1]),
                np.array([lat2]),
                np.array([lon2]),
            )[0]
        )
        geo = float(geodesic_distance(lat1, lon1, lat2, lon2))
        eu = float(
            euclidean_projected_km(
                np.array([lat1]),
                np.array([lon1]),
                np.array([lat2]),
                np.array([lon2]),
            )[0]
        )

        rows.append(
            DistanceValidationRow(
                name=str(case["name"]),
                haversine_km=hav,
                geodesic_km=geo,
                euclidean_projected_km=eu,
                expected_km=float(case["expected_km"]),
                tolerance_km=float(case["tolerance_km"]),
            )
        )

    all_pass = True
    for row in rows:
        if abs(row.geodesic_km - row.expected_km) > row.tolerance_km:
            all_pass = False

    return all_pass, rows


def run_crs_validation() -> tuple[bool, dict[str, Any]]:
    """Validate CRS transform behavior and axis ordering assumptions."""
    # Simple India coordinate near Bengaluru.
    lat, lon = 12.9716, 77.5946
    x, y = project_wgs84_to_web_mercator(np.array([lat]), np.array([lon]))

    # EPSG:3857 should be in meter-scale magnitudes.
    meter_scale_ok = abs(float(x[0])) > 1_000_000 and abs(float(y[0])) > 1_000_000

    # Axis-order check: if swapped, values differ drastically.
    x_swapped, y_swapped = project_wgs84_to_web_mercator(np.array([lon]), np.array([lat]))
    axis_order_ok = abs(float(x[0]) - float(x_swapped[0])) > 1_000_000

    checks = {
        "crs_source": "EPSG:4326",
        "crs_projected": "EPSG:3857",
        "projected_coordinate_example": {"x_m": float(x[0]), "y_m": float(y[0])},
        "meter_scale_ok": meter_scale_ok,
        "axis_order_ok": axis_order_ok,
    }
    return meter_scale_ok and axis_order_ok, checks


def build_geospatial_validation_report() -> GeospatialValidationReport:
    """Run all geospatial checks and return one report object."""
    dist_ok, dist_rows = run_distance_validation()
    crs_ok, crs_checks = run_crs_validation()

    return GeospatialValidationReport(
        all_distance_cases_pass=dist_ok,
        all_crs_checks_pass=crs_ok,
        distance_cases=[asdict(r) for r in dist_rows],
        crs_checks=crs_checks,
    )


def save_geospatial_validation_report(report: GeospatialValidationReport, path: Path) -> Path:
    """Persist geospatial validation report as JSON."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path
