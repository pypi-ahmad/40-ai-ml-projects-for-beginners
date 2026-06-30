from __future__ import annotations

from pathlib import Path

from hybrid_research_assistant.index_manifest import IndexManifestManager
from hybrid_research_assistant.schemas import DocumentRecord


def test_manifest_diff_and_save(tmp_path: Path) -> None:
    manager = IndexManifestManager(tmp_path / "manifest.json")
    docs = [
        DocumentRecord(doc_id="d1", text="hello", metadata={"document_hash": "h1"}),
        DocumentRecord(doc_id="d2", text="world", metadata={"document_hash": "h2"}),
    ]

    diff = manager.diff(
        docs,
        schema_version="1.0",
        collection_name="col",
        namespace="ns",
        embedding_model="emb",
        chunking_strategy="recursive",
        chunk_size=512,
        chunk_overlap=50,
        embedding_dimension=384,
    )
    assert diff.compatible is False
    assert len(diff.added_or_changed_doc_ids) == 2

    manager.save(
        docs,
        schema_version="1.0",
        collection_name="col",
        namespace="ns",
        embedding_model="emb",
        chunking_strategy="recursive",
        chunk_size=512,
        chunk_overlap=50,
        embedding_dimension=384,
    )
    assert manager.exists()
