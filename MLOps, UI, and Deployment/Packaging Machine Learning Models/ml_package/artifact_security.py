"""Artifact integrity helpers for serialized model files."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml_package.exceptions import ArtifactVerificationError


def compute_sha256(path: str | Path, chunk_size: int = 65536) -> str:
    """Compute SHA256 digest for a file."""
    file_path = Path(path)
    hasher = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def manifest_path_for(artifact_path: str | Path) -> Path:
    artifact = Path(artifact_path)
    return artifact.with_suffix(f"{artifact.suffix}.manifest.json")


def write_manifest(
    artifact_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write sidecar manifest with checksum and metadata."""
    artifact = Path(artifact_path)
    digest = compute_sha256(artifact)
    payload = {
        "artifact": artifact.name,
        "sha256": digest,
        "size_bytes": artifact.stat().st_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload["metadata"] = metadata

    manifest_path = manifest_path_for(artifact)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return manifest_path


def read_manifest(artifact_path: str | Path) -> dict[str, Any] | None:
    """Read sidecar manifest if present."""
    manifest_path = manifest_path_for(artifact_path)
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_manifest_checksum(
    artifact_path: str | Path,
    *,
    required: bool = False,
) -> str:
    """Verify artifact checksum against manifest and return computed digest."""
    artifact = Path(artifact_path)
    manifest = read_manifest(artifact)
    digest = compute_sha256(artifact)

    if manifest is None:
        if required:
            raise ArtifactVerificationError(
                f"Missing manifest for artifact: {artifact}"
            )
        return digest

    expected = manifest.get("sha256")
    if expected is None:
        raise ArtifactVerificationError(
            f"Manifest missing sha256 field: {manifest_path_for(artifact)}"
        )
    if digest != expected:
        raise ArtifactVerificationError(
            "Artifact checksum mismatch. "
            f"expected={expected}, actual={digest}, artifact={artifact}"
        )
    return digest
