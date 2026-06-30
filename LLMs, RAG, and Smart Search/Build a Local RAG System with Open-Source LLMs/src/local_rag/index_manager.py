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
        manifest_schema_version: str,
        corpus_profile: str,
        embedding_model: str,
        normalize_embeddings: bool,
        embedding_dimension: int,
        chunk_size: int,
        chunk_overlap: int,
        collection_name: str,
    ) -> ManifestDiff:
        """Compute incremental indexing diff against saved manifest."""

        current = {doc.doc_id: str(doc.metadata.get("hash", "")) for doc in docs}
        previous = self.load()

        previous_docs = previous.get("documents", {})
        previous_schema_version = previous.get("schema_version")
        previous_corpus_profile = previous.get("corpus_profile")
        previous_model = previous.get("embedding_model")
        previous_normalize = previous.get("normalize_embeddings")
        previous_dimension = previous.get("embedding_dimension")
        previous_chunk_size = previous.get("chunk_size")
        previous_chunk_overlap = previous.get("chunk_overlap")
        previous_collection = previous.get("collection_name")

        compatible = (
            previous_schema_version == manifest_schema_version
            and previous_corpus_profile == corpus_profile
            and previous_model == embedding_model
            and previous_normalize == normalize_embeddings
            and previous_dimension == embedding_dimension
            and previous_chunk_size == chunk_size
            and previous_chunk_overlap == chunk_overlap
            and previous_collection == collection_name
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
        manifest_schema_version: str,
        corpus_profile: str,
        embedding_model: str,
        normalize_embeddings: bool,
        embedding_dimension: int,
        chunk_size: int,
        chunk_overlap: int,
        collection_name: str,
    ) -> None:
        """Persist index manifest after successful upsert."""

        payload = {
            "schema_version": manifest_schema_version,
            "corpus_profile": corpus_profile,
            "embedding_model": embedding_model,
            "normalize_embeddings": normalize_embeddings,
            "embedding_dimension": embedding_dimension,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "collection_name": collection_name,
            "total_documents": len(docs),
            "documents": {doc.doc_id: doc.metadata.get("hash", "") for doc in docs},
        }
        json_dump(self.manifest_path, payload)
