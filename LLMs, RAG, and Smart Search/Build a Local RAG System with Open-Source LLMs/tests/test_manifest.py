from __future__ import annotations

from pathlib import Path

from local_rag.index_manager import IndexManifestManager
from local_rag.types import LoadedDocument


def _docs() -> list[LoadedDocument]:
    return [
        LoadedDocument(doc_id="d1", text="a", metadata={"hash": "h1"}),
        LoadedDocument(doc_id="d2", text="b", metadata={"hash": "h2"}),
    ]


def test_manifest_diff_and_save(tmp_path: Path) -> None:
    manager = IndexManifestManager(tmp_path / "manifest.json")
    docs = _docs()

    first = manager.diff(
        docs,
        manifest_schema_version="1.0",
        corpus_profile="full",
        embedding_model="qwen",
        normalize_embeddings=True,
        embedding_dimension=3,
        chunk_size=256,
        chunk_overlap=32,
        collection_name="c_full",
    )
    assert first.compatible is False
    assert sorted(first.added_or_changed_doc_ids) == ["d1", "d2"]

    manager.save(
        docs,
        manifest_schema_version="1.0",
        corpus_profile="full",
        embedding_model="qwen",
        normalize_embeddings=True,
        embedding_dimension=3,
        chunk_size=256,
        chunk_overlap=32,
        collection_name="c_full",
    )

    second = manager.diff(
        docs,
        manifest_schema_version="1.0",
        corpus_profile="full",
        embedding_model="qwen",
        normalize_embeddings=True,
        embedding_dimension=3,
        chunk_size=256,
        chunk_overlap=32,
        collection_name="c_full",
    )
    assert second.compatible is True
    assert second.added_or_changed_doc_ids == []

    third = manager.diff(
        docs,
        manifest_schema_version="1.0",
        corpus_profile="full",
        embedding_model="qwen",
        normalize_embeddings=True,
        embedding_dimension=3,
        chunk_size=512,
        chunk_overlap=32,
        collection_name="c_full",
    )
    assert third.compatible is False
    assert sorted(third.added_or_changed_doc_ids) == ["d1", "d2"]
