"""Filesystem model registry with versioning and metadata tracking."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib


def _model_root(base_dir: str | Path, model_name: str) -> Path:
    root = Path(base_dir) / model_name
    root.mkdir(parents=True, exist_ok=True)
    return root


def _latest_version(base_dir: str | Path, model_name: str) -> int:
    root = _model_root(base_dir, model_name)
    versions: list[int] = []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith("v"):
            try:
                versions.append(int(item.name[1:]))
            except ValueError:
                continue
    return max(versions) if versions else 0


def register_model(
    model: Any,
    model_name: str,
    base_dir: str | Path,
    metrics: dict[str, Any],
    hyperparameters: dict[str, Any],
    feature_names: list[str],
    stage: str = "staging",
    mlflow_run_id: str | None = None,
) -> dict[str, Any]:
    """Register model with auto-incremented version and metadata."""
    next_version = _latest_version(base_dir=base_dir, model_name=model_name) + 1
    version_dir = _model_root(base_dir, model_name) / f"v{next_version}"
    version_dir.mkdir(parents=True, exist_ok=True)

    model_path = version_dir / "model.joblib"
    metadata_path = version_dir / "metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_name": model_name,
        "version": next_version,
        "registered_at": datetime.utcnow().isoformat(),
        "model_type": type(model).__name__,
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "feature_names": feature_names,
        "stage": stage,
        "mlflow_run_id": mlflow_run_id,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    set_stage(base_dir=base_dir, model_name=model_name, stage=stage, version=next_version)

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "version": next_version,
        "stage": stage,
    }


def set_stage(base_dir: str | Path, model_name: str, stage: str, version: int) -> None:
    """Update stage mapping for model."""
    root = _model_root(base_dir, model_name)
    stage_path = root / "stages.json"
    payload: dict[str, int] = {}
    if stage_path.exists():
        payload = json.loads(stage_path.read_text(encoding="utf-8"))
    payload[stage] = version
    stage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_model(base_dir: str | Path, model_name: str, version: int | None = None, stage: str | None = None) -> tuple[Any, dict[str, Any]]:
    """Load registered model by version or stage alias."""
    root = _model_root(base_dir, model_name)

    if version is None and stage is not None:
        stage_path = root / "stages.json"
        if not stage_path.exists():
            raise FileNotFoundError(f"stages.json missing for model {model_name}")
        stage_map = json.loads(stage_path.read_text(encoding="utf-8"))
        if stage not in stage_map:
            raise KeyError(f"Stage '{stage}' not found for model {model_name}")
        version = int(stage_map[stage])

    if version is None:
        version = _latest_version(base_dir=base_dir, model_name=model_name)
        if version == 0:
            raise FileNotFoundError(f"No versions found for model {model_name}")

    version_dir = root / f"v{version}"
    model = joblib.load(version_dir / "model.joblib")
    metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8"))
    return model, metadata


def list_versions(base_dir: str | Path, model_name: str) -> list[dict[str, Any]]:
    """List all versions with metadata."""
    root = _model_root(base_dir, model_name)
    versions = []
    for item in sorted(root.iterdir()):
        if item.is_dir() and item.name.startswith("v"):
            metadata_path = item / "metadata.json"
            if metadata_path.exists():
                versions.append(json.loads(metadata_path.read_text(encoding="utf-8")))
    return versions
