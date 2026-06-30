"""Environment-driven runtime settings for CLI/API workflows."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


def _to_bool(raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(raw: str | None, default: list[str]) -> list[str]:
    if raw is None:
        return default
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return items or default


def _split_set(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {item.strip() for item in raw.split(",") if item.strip()}


@dataclass(frozen=True)
class PackageSettings:
    model_path: Path
    registry_path: Path
    background_data_path: Path
    log_dir: Path
    cors_origins: list[str]
    verify_artifacts: bool
    allow_unsafe_deserialization: bool
    trusted_digests: set[str]
    api_host: str
    api_port: int

    @classmethod
    def from_env(cls) -> "PackageSettings":
        return cls(
            model_path=Path(os.getenv("ML_MODEL_PATH", "models/iris_model.pkl")),
            registry_path=Path(os.getenv("ML_REGISTRY_PATH", "models/registry.json")),
            background_data_path=Path(
                os.getenv("ML_BACKGROUND_DATA_PATH", "models/background_data.npy")
            ),
            log_dir=Path(os.getenv("ML_LOG_DIR", "logs")),
            cors_origins=_split_csv(
                os.getenv("ML_API_CORS_ORIGINS"),
                ["http://localhost", "http://127.0.0.1"],
            ),
            verify_artifacts=_to_bool(os.getenv("ML_VERIFY_ARTIFACTS"), True),
            allow_unsafe_deserialization=_to_bool(
                os.getenv("ML_ALLOW_UNSAFE_DESERIALIZATION"), False
            ),
            trusted_digests=_split_set(os.getenv("ML_TRUSTED_DIGESTS")),
            api_host=os.getenv("ML_API_HOST", "0.0.0.0"),
            api_port=int(os.getenv("ML_API_PORT", "8000")),
        )

    def resolved_trusted_digests(self, model_path: str | Path | None = None) -> set[str]:
        """Return trusted digest allow-list from env + registry metadata."""
        digests = set(self.trusted_digests)
        if not self.registry_path.exists():
            return digests

        try:
            with self.registry_path.open(encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return digests

        versions = data.get("versions", {})
        active_version = data.get("active_version")

        if active_version:
            active_record = versions.get(active_version, {})
            active_digest = active_record.get("artifact_sha256")
            if isinstance(active_digest, str) and active_digest:
                digests.add(active_digest)

        if model_path is None:
            return digests

        target = Path(model_path).resolve(strict=False)
        for version in versions.values():
            version_path = version.get("model_path")
            version_digest = version.get("artifact_sha256")
            if not isinstance(version_path, str) or not isinstance(version_digest, str):
                continue

            resolved = Path(version_path).resolve(strict=False)
            if resolved == target:
                digests.add(version_digest)

        return digests
