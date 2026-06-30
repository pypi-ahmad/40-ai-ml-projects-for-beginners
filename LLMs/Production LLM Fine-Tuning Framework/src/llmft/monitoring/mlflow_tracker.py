"""Optional MLflow tracking wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from llmft.utils.io import write_json
from llmft.utils.logging import get_logger


class MLflowTracker:
    """Track runs with MLflow when available, fallback to local JSON logs."""

    def __init__(self, artifacts_dir: str | Path, experiment_name: str = "llmft") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.experiment_name = experiment_name
        self.logger = get_logger("llmft.mlflow", self.artifacts_dir / "logs" / "mlflow.log")
        self._mlflow = None
        try:
            import mlflow  # type: ignore

            self._mlflow = mlflow
            mlflow.set_experiment(experiment_name)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("mlflow unavailable, using local tracker: %s", exc)

    def log_run(self, run_name: str, params: dict[str, Any], metrics: dict[str, float], artifacts: dict[str, str]) -> None:
        """Log run metadata."""
        if self._mlflow is not None:
            with self._mlflow.start_run(run_name=run_name):
                self._mlflow.log_params(params)
                self._mlflow.log_metrics(metrics)
                for key, value in artifacts.items():
                    if Path(value).exists():
                        self._mlflow.log_artifact(value, artifact_path=key)
            return

        write_json(
            self.artifacts_dir / "mlflow" / f"{run_name}.json",
            {
                "run_name": run_name,
                "params": params,
                "metrics": metrics,
                "artifacts": artifacts,
            },
        )
