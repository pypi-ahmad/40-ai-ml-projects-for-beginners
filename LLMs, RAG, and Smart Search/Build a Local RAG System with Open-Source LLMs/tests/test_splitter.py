from __future__ import annotations

from local_rag.splitter import TextChunker
from local_rag.types import LoadedDocument


def test_chunking_generates_overlap_chunks() -> None:
    text = " ".join(["token"] * 300)
    doc = LoadedDocument(
        doc_id="doc1",
        text=text,
        metadata={"doc_id": "doc1", "source_path": "x.txt", "hash": "abc"},
    )

    chunker = TextChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.split_documents([doc])

    assert len(chunks) > 1
    assert all(chunk.metadata["chunk_size"] == 120 for chunk in chunks)
    assert all(chunk.metadata["chunk_overlap"] == 30 for chunk in chunks)
    assert all(chunk.doc_id == "doc1" for chunk in chunks)
