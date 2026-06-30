"""Optional MLflow tracking for run metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class MLflowTracker:
    """Best-effort local MLflow logger."""

    def __init__(self, tracking_dir: str = "artifacts/mlruns") -> None:
        self.tracking_dir = tracking_dir

    def log_run(self, run_name: str, params: dict[str, Any], metrics: dict[str, float]) -> None:
        try:
            import mlflow

            root = Path(self.tracking_dir).resolve()
            root.mkdir(parents=True, exist_ok=True)

            db_path = root / "mlflow.db"
            mlflow.set_tracking_uri(f"sqlite:///{db_path}")
            mlflow.set_experiment("crew-platform-runs")
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({key: str(value) for key, value in params.items()})
                mlflow.log_metrics({key: float(value) for key, value in metrics.items()})
        except Exception:
            # Never break primary workflow if tracking backend is unavailable.
            return
