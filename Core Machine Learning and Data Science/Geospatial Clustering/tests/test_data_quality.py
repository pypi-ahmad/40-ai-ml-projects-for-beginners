"""Tests for dataset quality gate."""

from __future__ import annotations

from src.config import TRAIN_FILE_PATH
from src.data_loader import load_raw_data
from src.data_quality import run_quality_gate


def test_quality_gate_has_core_sections() -> None:
    raw = load_raw_data(TRAIN_FILE_PATH)
    report = run_quality_gate(raw)

    assert report.n_rows == len(raw)
    assert report.n_columns == raw.shape[1]
    assert "Restaurant_latitude" in report.coordinate_ranges
    assert "Restaurant_longitude" in report.coordinate_ranges
    assert "Delivery_location_latitude" in report.coordinate_ranges
    assert "Delivery_location_longitude" in report.coordinate_ranges
    assert report.distance_summary_km["max"] >= report.distance_summary_km["p99"]
    assert report.rows_outside_global_bounds == 0
    assert report.rows_outside_india_bounds >= 0
    assert report.rows_with_zero_coordinates >= 0
