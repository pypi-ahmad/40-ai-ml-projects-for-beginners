"""Data validation and drift utilities for MLOps quality gates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .data_loader import save_json
from .settings import load_config, resolve_path


def _check_schema(df: pd.DataFrame, expected_columns: list[str]) -> dict[str, Any]:
    missing_cols = [col for col in expected_columns if col not in df.columns]
    unexpected_cols = [col for col in df.columns if col not in expected_columns and col != "Synthetic"]
    return {
        "missing_columns": missing_cols,
        "unexpected_columns": unexpected_cols,
        "passed": len(missing_cols) == 0,
    }


def _check_missing(df: pd.DataFrame) -> dict[str, Any]:
    counts = df.isna().sum().to_dict()
    pct = ((df.isna().sum() / max(len(df), 1)) * 100).round(3).to_dict()
    return {
        "missing_counts": {k: int(v) for k, v in counts.items() if int(v) > 0},
        "missing_pct": {k: float(v) for k, v in pct.items() if float(v) > 0},
    }


def _check_domain(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    invalid_rows = {
        "negative_usage": int((df[target_col] < 0).sum()) if target_col in df.columns else 0,
        "negative_notifications": int((df["Notifications"] < 0).sum()) if "Notifications" in df.columns else 0,
        "negative_times_opened": int((df["Times Opened"] < 0).sum()) if "Times Opened" in df.columns else 0,
    }
    return invalid_rows


def _check_duplicates(df: pd.DataFrame) -> dict[str, Any]:
    duplicated = int(df.duplicated().sum())
    return {"duplicate_rows": duplicated, "passed": duplicated == 0}


def _check_basic_stats(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number])
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "numeric_summary": numeric.describe().to_dict() if not numeric.empty else {},
    }


def compute_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute PSI between expected and actual distribution."""
    expected = expected.dropna().to_numpy()
    actual = actual.dropna().to_numpy()
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    min_val = min(float(expected.min()), float(actual.min()))
    max_val = max(float(expected.max()), float(actual.max()))
    if min_val == max_val:
        return 0.0

    breakpoints = np.linspace(min_val, max_val, bins + 1)
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)

    expected_pct = np.clip(expected_pct, 1e-6, 1.0)
    actual_pct = np.clip(actual_pct, 1e-6, 1.0)
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def detect_drift(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: list[str],
    psi_threshold: float,
    ks_pvalue_threshold: float,
) -> dict[str, Any]:
    """Run PSI + KS drift checks across numeric features."""
    report: dict[str, Any] = {}
    drifted = []

    for col in numeric_cols:
        if col not in baseline_df.columns or col not in current_df.columns:
            continue
        psi = compute_psi(baseline_df[col], current_df[col])
        ks_stat, ks_pvalue = ks_2samp(
            baseline_df[col].dropna().to_numpy(),
            current_df[col].dropna().to_numpy(),
            alternative="two-sided",
            mode="auto",
        )
        psi_flag = psi > psi_threshold
        ks_flag = ks_pvalue < ks_pvalue_threshold
        if psi_flag or ks_flag:
            drifted.append(col)
        report[col] = {
            "psi": round(float(psi), 6),
            "psi_drift": bool(psi_flag),
            "ks_stat": round(float(ks_stat), 6),
            "ks_pvalue": round(float(ks_pvalue), 6),
            "ks_drift": bool(ks_flag),
        }

    return {
        "drift_detected": len(drifted) > 0,
        "drifted_columns": drifted,
        "feature_report": report,
    }


def run_full_validation(df: pd.DataFrame, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run full data quality checks and return serializable report."""
    config = config or load_config()
    validation_cfg = config["validation"]
    project_cfg = config["project"]

    schema = _check_schema(df, expected_columns=list(validation_cfg["expected_columns"]))
    missing = _check_missing(df)
    domain = _check_domain(df, target_col=str(project_cfg["target_col"]))
    duplicate = _check_duplicates(df)
    stats = _check_basic_stats(df)

    max_missing_pct = float(validation_cfg["max_missing_pct"])
    missing_violations = {
        col: pct for col, pct in missing["missing_pct"].items() if float(pct) > max_missing_pct
    }

    min_rows = int(validation_cfg["min_rows"])
    checks_passed = (
        schema["passed"]
        and duplicate["passed"]
        and len(missing_violations) == 0
        and all(count == 0 for count in domain.values())
        and stats["rows"] >= min_rows
    )

    return {
        "checks_passed": bool(checks_passed),
        "schema": schema,
        "missing": missing,
        "missing_violations": missing_violations,
        "domain": domain,
        "duplicates": duplicate,
        "stats": stats,
    }


def save_validation_report(report: dict[str, Any], config: dict[str, Any] | None = None) -> str:
    """Persist validation report JSON and return file path."""
    config = config or load_config()
    out_path = resolve_path(config, "reports_dir") / "data_validation_report.json"
    save_json(report, out_path)
    return str(out_path)
