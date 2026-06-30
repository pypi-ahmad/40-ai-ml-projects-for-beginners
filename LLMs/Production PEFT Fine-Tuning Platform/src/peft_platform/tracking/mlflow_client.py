"""MLflow tracking wrapper."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from peft_platform.utils.io import ensure_dir


class TrackingClient:
    """Thin MLflow wrapper with no-op fallback."""

    def __init__(self, tracking_uri: str, experiment: str) -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.enabled = False
        self._mlflow = None

        try:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment)
            self._mlflow = mlflow
            self.enabled = True
        except Exception:
            self.enabled = False

    def log_run(
        self,
        run_name: str,
        params: dict[str, Any],
        metrics: dict[str, float],
        artifacts: list[Path] | None = None,
    ) -> str:
        if not self.enabled or self._mlflow is None:
            return "mlflow-disabled"

        with self._mlflow.start_run(run_name=run_name) as run:
            self._mlflow.log_params(params)
            self._mlflow.log_metrics(metrics)
            if artifacts:
                for artifact in artifacts:
                    if artifact.exists():
                        self._mlflow.log_artifact(str(artifact))
            return run.info.run_id


def save_manifest(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    serialized = asdict(payload) if hasattr(payload, "__dataclass_fields__") else payload
    path.write_text(str(serialized), encoding="utf-8")
