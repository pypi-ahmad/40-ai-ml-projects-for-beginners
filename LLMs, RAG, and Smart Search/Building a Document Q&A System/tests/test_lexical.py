from __future__ import annotations

from pathlib import Path

from local_rag.lexical import BM25LexicalIndex
from local_rag.types import ChunkRecord


def _chunk(chunk_id: str, doc_id: str, text: str) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        metadata={"doc_id": doc_id, "source_path": f"{doc_id}.txt", "source_type": "txt"},
    )


def test_bm25_build_query_and_load(tmp_path: Path) -> None:
    index_path = tmp_path / "bm25.json"
    index = BM25LexicalIndex(index_path)
    chunks = [
        _chunk("c1", "d1", "network encryption policy and controls"),
        _chunk("c2", "d2", "kernel scheduler details"),
    ]
    stats = index.build(chunks)
    assert stats.chunk_count == 2

    hits = index.query("encryption policy", top_k=1)
    assert len(hits) == 1
    assert hits[0].chunk_id == "c1"
    assert hits[0].strategy == "keyword"

    reloaded = BM25LexicalIndex(index_path)
    assert reloaded.load() is True
    hits2 = reloaded.query("scheduler", top_k=1)
    assert len(hits2) == 1
    assert hits2[0].chunk_id == "c2"
