import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ModelVersion:
    """Represents a single versioned model snapshot.

    Each version captures:
    - The model artifact path
    - Training metadata (accuracy, parameters, feature count)
    - Timestamp and author
    - Change description

    This enables model rollback, audit trails, and production monitoring.
    """

    def __init__(
        self,
        version_id: str,
        model_path: str,
        metrics: dict | None = None,
        description: str = "",
        author: str = "ml_package",
        *,
        artifact_sha256: str | None = None,
        dataset_fingerprint: str | None = None,
        parent_version: str | None = None,
        tags: list[str] | None = None,
    ):
        self.version_id = version_id
        self.model_path = model_path
        self.metrics = metrics or {}
        self.description = description
        self.author = author
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status = "created"
        self.artifact_sha256 = artifact_sha256
        self.dataset_fingerprint = dataset_fingerprint
        self.parent_version = parent_version
        self.tags = tags or []

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "model_path": str(self.model_path),
            "metrics": self.metrics,
            "description": self.description,
            "author": self.author,
            "created_at": self.created_at,
            "status": self.status,
            "artifact_sha256": self.artifact_sha256,
            "dataset_fingerprint": self.dataset_fingerprint,
            "parent_version": self.parent_version,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelVersion":
        version = cls(
            version_id=payload["version_id"],
            model_path=payload["model_path"],
            metrics=payload.get("metrics", {}),
            description=payload.get("description", ""),
            author=payload.get("author", "ml_package"),
            artifact_sha256=payload.get("artifact_sha256"),
            dataset_fingerprint=payload.get("dataset_fingerprint"),
            parent_version=payload.get("parent_version"),
            tags=payload.get("tags", []),
        )
        version.created_at = payload.get("created_at", version.created_at)
        version.status = payload.get("status", "created")
        return version

    def mark_active(self) -> None:
        self.status = "active"

    def mark_archived(self) -> None:
        self.status = "archived"

    def mark_failed(self) -> None:
        self.status = "failed"


class VersionRegistry:
    """Manages model version history with load/save to disk.

    Provides a simple registry file (JSON) that maps version IDs to
    model metadata. Supports promotion, rollback, and audit queries.

    Usage:
        registry = VersionRegistry("models/registry.json")
        registry.register("v1", "models/model_v1.pkl", {"accuracy": 0.95})
        registry.activate("v1")
    """

    SCHEMA_VERSION = 1

    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._versions: dict[str, ModelVersion] = {}
        self._active_version: str | None = None
        self._load()

    def _load(self) -> None:
        """Load registry from disk, creating if missing."""
        if self.registry_path.exists():
            with open(self.registry_path, encoding="utf-8") as f:
                data = json.load(f)
            for vid, vdata in data.get("versions", {}).items():
                payload = dict(vdata)
                payload["version_id"] = vid
                mv = ModelVersion.from_dict(payload)
                self._versions[vid] = mv
            self._active_version = data.get("active_version")

    def _save(self) -> None:
        """Persist registry to disk as JSON."""
        data = {
            "schema_version": self.SCHEMA_VERSION,
            "active_version": self._active_version,
            "versions": {
                vid: mv.to_dict() for vid, mv in self._versions.items()
            },
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        version_id: str,
        model_path: str,
        metrics: dict | None = None,
        description: str = "",
        author: str = "ml_package",
        *,
        artifact_sha256: str | None = None,
        dataset_fingerprint: str | None = None,
        parent_version: str | None = None,
        tags: list[str] | None = None,
        allow_overwrite: bool = False,
    ) -> ModelVersion:
        """Register a new model version in the registry.

        Args:
            version_id: Unique version identifier (e.g., "v1", "v2")
            model_path: Path to the serialized model artifact
            metrics: Dict of evaluation metrics
            description: Human-readable change description
            author: Who/what created this version

        Returns:
            The registered ModelVersion
        """
        if version_id in self._versions and not allow_overwrite:
            raise ValueError(
                f"Version '{version_id}' already exists. "
                "Set allow_overwrite=True to replace it."
            )
        if parent_version is not None and parent_version not in self._versions:
            raise ValueError(
                f"Parent version '{parent_version}' is not registered."
            )

        mv = ModelVersion(
            version_id=version_id,
            model_path=model_path,
            metrics=metrics,
            description=description,
            author=author,
            artifact_sha256=artifact_sha256,
            dataset_fingerprint=dataset_fingerprint,
            parent_version=parent_version,
            tags=tags,
        )
        self._versions[version_id] = mv
        self._save()
        return mv

    def collect_artifact_digests(self) -> set[str]:
        """Return all non-empty artifact digests from registered versions."""
        digests: set[str] = set()
        for version in self._versions.values():
            if version.artifact_sha256:
                digests.add(version.artifact_sha256)
        return digests

    def next_version_id(self, prefix: str = "v") -> str:
        """Return next numeric version id for the given prefix."""
        max_index = 0
        for version_id in self._versions:
            if version_id.startswith(prefix):
                suffix = version_id[len(prefix) :]
                if suffix.isdigit():
                    max_index = max(max_index, int(suffix))
        return f"{prefix}{max_index + 1}"

    def activate(self, version_id: str) -> None:
        """Promote a version to active (production) status.

        Automatically deactivates the previously active version.
        """
        if version_id not in self._versions:
            raise KeyError(f"Version '{version_id}' not found in registry")
        if self._active_version:
            self._versions[self._active_version].mark_archived()
        self._versions[version_id].mark_active()
        self._active_version = version_id
        self._save()

    def get_active(self) -> ModelVersion | None:
        """Return the currently active (production) version."""
        if self._active_version and self._active_version in self._versions:
            return self._versions[self._active_version]
        return None

    def get(self, version_id: str) -> ModelVersion:
        """Get a specific version by ID."""
        if version_id not in self._versions:
            raise KeyError(f"Version '{version_id}' not found")
        return self._versions[version_id]

    def list_versions(self) -> list[dict]:
        """Return all registered versions sorted by creation time (newest first)."""
        versions = [v.to_dict() for v in self._versions.values()]
        versions.sort(key=lambda x: x["created_at"], reverse=True)
        return versions

    def rollback_to(self, version_id: str) -> dict:
        """Rollback to a previous version. Returns the rolled-back version info."""
        if version_id not in self._versions:
            raise KeyError(f"Version '{version_id}' not found")
        old_active = self._active_version
        self.activate(version_id)
        return {
            "previous_active": old_active,
            "current_active": version_id,
        }
