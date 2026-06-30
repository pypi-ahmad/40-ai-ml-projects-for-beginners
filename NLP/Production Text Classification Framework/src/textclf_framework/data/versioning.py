"""Dataset version manifest utilities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from datasets import DatasetDict

from .schemas import DatasetVersionManifest


def _fingerprint_for_split(dataset_dict: DatasetDict, split: str) -> str:
    if split not in dataset_dict:
        return "missing"
    return str(getattr(dataset_dict[split], "_fingerprint", "unknown"))


def _preprocess_hash(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_manifest(
    dataset_name: str,
    source: str,
    dataset_dict: DatasetDict,
    split_seed: int,
    preprocessing_config: dict[str, object],
    label_names: list[str],
    revision: str | None = None,
) -> DatasetVersionManifest:
    """Create immutable dataset metadata manifest."""
    return DatasetVersionManifest(
        dataset_name=dataset_name,
        source=source,
        revision=revision,
        train_fingerprint=_fingerprint_for_split(dataset_dict, "train"),
        validation_fingerprint=_fingerprint_for_split(dataset_dict, "validation"),
        test_fingerprint=_fingerprint_for_split(dataset_dict, "test"),
        split_seed=split_seed,
        preprocessing_hash=_preprocess_hash(preprocessing_config),
        label_names=label_names,
        created_at_utc=datetime.now(tz=timezone.utc).isoformat(),
    )


def save_manifest(manifest: DatasetVersionManifest, output_path: str | Path) -> None:
    """Persist manifest to JSON path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(asdict(manifest), file, indent=2)
