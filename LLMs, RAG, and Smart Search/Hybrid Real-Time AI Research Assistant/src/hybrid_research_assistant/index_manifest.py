"""Manifest tracking for incremental indexing and document versioning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hybrid_research_assistant.schemas import DocumentRecord
from hybrid_research_assistant.utils import json_dump, json_load


@dataclass(slots=True)
class ManifestDiff:
    """Diff between current corpus and persisted manifest."""

    added_or_changed_doc_ids: list[str]
    removed_doc_ids: list[str]
    compatible: bool


class IndexManifestManager:
    """Track corpus and index config across runs."""

    def __init__(self, manifest_path: Path) -> None:
        self.manifest_path = manifest_path

    def exists(self) -> bool:
        return self.manifest_path.exists()

    def load(self) -> dict[str, Any]:
        if not self.exists():
            return {}
        return json_load(self.manifest_path)

    def diff(
        self,
        docs: list[DocumentRecord],
        *,
        schema_version: str,
        collection_name: str,
        namespace: str,
        embedding_model: str,
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_dimension: int,
    ) -> ManifestDiff:
        current = {doc.doc_id: str(doc.metadata.get("document_hash", "")) for doc in docs}
        previous = self.load()
        previous_docs = previous.get("documents", {})

        compatible = (
            previous.get("schema_version") == schema_version
            and previous.get("collection_name") == collection_name
            and previous.get("namespace") == namespace
            and previous.get("embedding_model") == embedding_model
            and previous.get("chunking_strategy") == chunking_strategy
            and int(previous.get("chunk_size", -1)) == chunk_size
            and int(previous.get("chunk_overlap", -1)) == chunk_overlap
            and int(previous.get("embedding_dimension", -1)) == embedding_dimension
        )

        if not previous:
            return ManifestDiff(added_or_changed_doc_ids=sorted(current.keys()), removed_doc_ids=[], compatible=False)

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
        docs: list[DocumentRecord],
        *,
        schema_version: str,
        collection_name: str,
        namespace: str,
        embedding_model: str,
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        embedding_dimension: int,
    ) -> None:
        payload = {
            "schema_version": schema_version,
            "collection_name": collection_name,
            "namespace": namespace,
            "embedding_model": embedding_model,
            "chunking_strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "embedding_dimension": embedding_dimension,
            "total_documents": len(docs),
            "documents": {doc.doc_id: doc.metadata.get("document_hash", "") for doc in docs},
        }
        json_dump(self.manifest_path, payload)
