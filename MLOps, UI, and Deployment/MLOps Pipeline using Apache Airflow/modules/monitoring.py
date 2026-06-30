"""Monitoring and alerting utilities for data/model/pipeline health."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import save_json
from .data_validator import compute_psi, detect_drift
from .settings import load_config, resolve_path


def prediction_drift(reference_pred: np.ndarray, current_pred: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute prediction PSI drift status."""
    psi = compute_psi(pd.Series(reference_pred), pd.Series(current_pred), bins=10)
    return {
        "psi": float(psi),
        "drift": bool(psi > threshold),
        "threshold": float(threshold),
    }


def pipeline_runtime_report(task_runtime_seconds: dict[str, float]) -> dict[str, Any]:
    """Compute runtime totals and bottleneck task."""
    total = float(sum(task_runtime_seconds.values()))
    worst_task = max(task_runtime_seconds, key=task_runtime_seconds.get) if task_runtime_seconds else ""
    return {
        "task_runtime_seconds": task_runtime_seconds,
        "total_runtime_seconds": total,
        "slowest_task": worst_task,
        "slowest_task_seconds": float(task_runtime_seconds.get(worst_task, 0.0)),
    }


def generate_alerts(
    metrics: dict[str, Any],
    drift_report: dict[str, Any],
    runtime_report: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate operational alerts from KPI thresholds."""
    config = config or load_config()
    monitoring_cfg = config["monitoring"]

    alerts: list[dict[str, Any]] = []

    if float(metrics.get("mae", 0.0)) > float(monitoring_cfg["error_threshold_mae"]):
        alerts.append(
            {
                "severity": "warning",
                "message": f"MAE {metrics['mae']:.4f} above threshold {monitoring_cfg['error_threshold_mae']}",
            }
        )

    if bool(drift_report.get("drift_detected", False)):
        alerts.append(
            {
                "severity": "critical",
                "message": f"Feature drift detected: {drift_report.get('drifted_columns', [])}",
            }
        )

    runtime_limit = float(monitoring_cfg["runtime_warning_sec"])
    if float(runtime_report.get("total_runtime_seconds", 0.0)) > runtime_limit:
        alerts.append(
            {
                "severity": "warning",
                "message": f"Pipeline runtime exceeded threshold: {runtime_report['total_runtime_seconds']:.2f}s > {runtime_limit:.2f}s",
            }
        )

    return alerts


def save_monitoring_snapshot(
    metrics: dict[str, Any],
    drift_report: dict[str, Any],
    runtime_report: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str:
    """Persist timestamped monitoring snapshot with alerts."""
    config = config or load_config()
    alerts = generate_alerts(metrics=metrics, drift_report=drift_report, runtime_report=runtime_report, config=config)
    payload = {
        "created_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "drift_report": drift_report,
        "runtime_report": runtime_report,
        "alerts": alerts,
    }

    output_dir = resolve_path(config, "monitoring_dir")
    filename = f"monitoring_snapshot_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = output_dir / filename
    save_json(payload, output_path)
    return str(output_path)


def run_data_drift_monitoring(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: list[str],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run configured drift checks for numeric features."""
    config = config or load_config()
    val_cfg = config["validation"]
    return detect_drift(
        baseline_df=baseline_df,
        current_df=current_df,
        numeric_cols=numeric_cols,
        psi_threshold=float(val_cfg["psi_threshold"]),
        ks_pvalue_threshold=float(val_cfg["ks_pvalue_threshold"]),
    )
