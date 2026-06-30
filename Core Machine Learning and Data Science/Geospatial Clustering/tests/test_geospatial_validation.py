"""Tests for geospatial math and CRS validation helpers."""

from __future__ import annotations

from src.geospatial_validation import build_geospatial_validation_report


def test_geospatial_validation_report_passes_core_checks() -> None:
    report = build_geospatial_validation_report()

    assert report.all_distance_cases_pass is True
    assert report.all_crs_checks_pass is True
    assert len(report.distance_cases) >= 3
    assert report.crs_checks["crs_source"] == "EPSG:4326"
    assert report.crs_checks["crs_projected"] == "EPSG:3857"
