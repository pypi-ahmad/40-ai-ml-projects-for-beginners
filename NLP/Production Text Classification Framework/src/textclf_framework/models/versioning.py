"""Model version metadata and local registry helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(slots=True)
class ModelVersion:
    name: str
    version: str
    dataset: str
    strategy: str
    metrics: dict[str, float]
    artifact_path: str
    created_at_utc: str


def save_model_version(version: ModelVersion, registry_path: str | Path) -> None:
    """Append model version metadata to local registry file."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    registry: list[dict[str, object]] = []
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))

    registry.append(asdict(version))
    path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def new_version(
    name: str,
    dataset: str,
    strategy: str,
    metrics: dict[str, float],
    artifact_path: str,
) -> ModelVersion:
    """Create timestamp-based model version record."""
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    return ModelVersion(
        name=name,
        version=f"v{stamp}",
        dataset=dataset,
        strategy=strategy,
        metrics=metrics,
        artifact_path=artifact_path,
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
    )
