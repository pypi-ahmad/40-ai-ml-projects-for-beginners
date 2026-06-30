from __future__ import annotations

from pathlib import Path

from local_rag.types import ChunkRecord
from local_rag.vectordb import ChromaVectorStore


def _chunk(chunk_id: str, doc_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        metadata={
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "source_path": f"{doc_id}.txt",
            "source_type": "txt",
        },
    )


def test_chroma_upsert_query_delete(tmp_path: Path) -> None:
    store = ChromaVectorStore(db_path=tmp_path / "chroma", collection_name="test")

    chunks = [
        _chunk("c1", "doc1", "alpha"),
        _chunk("c2", "doc2", "beta"),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    store.upsert_chunks(chunks, embeddings)
    assert store.count() == 2
    integrity = store.integrity_report()
    assert integrity["total_vectors"] == 2
    assert integrity["missing_required_metadata"] == 0

    hits = store.query(query_embedding=[1.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "doc1"

    store.delete_by_doc_ids(["doc1"])
    assert store.count() == 1
