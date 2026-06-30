"""Persistent index manifest management for incremental indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_rag.types import LoadedDocument
from local_rag.utils import json_dump, json_load


@dataclass(slots=True)
class ManifestDiff:
    """Diff between current corpus and persisted manifest."""

    added_or_changed_doc_ids: list[str]
    removed_doc_ids: list[str]
    compatible: bool


class IndexManifestManager:
    """Track corpus and indexing config across runs."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

    def exists(self) -> bool:
        """Check whether manifest file exists."""

        return self.manifest_path.exists()

    def load(self) -> dict[str, Any]:
        """Load persisted manifest."""

        if not self.exists():
            return {}
        return json_load(self.manifest_path)

    def diff(
        self,
        docs: list[LoadedDocument],
        *,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> ManifestDiff:
        """Compute incremental indexing diff against saved manifest."""

        current = {
            doc.doc_id: f"{doc.metadata.get('hash', '')}::{doc.metadata.get('version_id', '')}"
            for doc in docs
        }
        previous = self.load()

        previous_docs = previous.get("documents", {})
        previous_model = previous.get("embedding_model")
        previous_chunk_size = previous.get("chunk_size")
        previous_chunk_overlap = previous.get("chunk_overlap")

        compatible = (
            previous_model == embedding_model
            and previous_chunk_size == chunk_size
            and previous_chunk_overlap == chunk_overlap
        )

        if not previous:
            return ManifestDiff(
                added_or_changed_doc_ids=sorted(current.keys()),
                removed_doc_ids=[],
                compatible=False,
            )

        added_or_changed = [
            doc_id for doc_id, doc_hash in current.items() if previous_docs.get(doc_id) != doc_hash
        ]
        removed = [doc_id for doc_id in previous_docs if doc_id not in current]

        if not compatible:
            return ManifestDiff(
                added_or_changed_doc_ids=sorted(current.keys()),
                removed_doc_ids=sorted(previous_docs.keys()),
                compatible=False,
            )

        return ManifestDiff(
            added_or_changed_doc_ids=sorted(added_or_changed),
            removed_doc_ids=sorted(removed),
            compatible=True,
        )

    def save(
        self,
        docs: list[LoadedDocument],
        *,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        collection_name: str,
    ) -> None:
        """Persist index manifest after successful upsert."""

        payload = {
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "collection_name": collection_name,
            "documents": {
                doc.doc_id: f"{doc.metadata.get('hash', '')}::{doc.metadata.get('version_id', '')}"
                for doc in docs
            },
        }
        json_dump(self.manifest_path, payload)
