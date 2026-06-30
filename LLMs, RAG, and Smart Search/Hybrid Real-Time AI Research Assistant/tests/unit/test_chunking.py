from __future__ import annotations

from hybrid_research_assistant.chunking import Chunker
from hybrid_research_assistant.schemas import ChunkingStrategy, DocumentRecord


def _sample_doc() -> DocumentRecord:
    return DocumentRecord(
        doc_id="doc-1",
        text=("Sentence one. Sentence two. Sentence three. " * 20).strip(),
        metadata={"doc_id": "doc-1", "source": "sample.txt"},
    )


def test_recursive_chunking_produces_chunks() -> None:
    chunker = Chunker(chunk_size=120, chunk_overlap=20, strategy=ChunkingStrategy.RECURSIVE)
    chunks = chunker.split_documents([_sample_doc()])
    assert len(chunks) > 1
    assert all(chunk.metadata["chunking_strategy"] == "recursive" for chunk in chunks)


def test_token_chunking_produces_chunks() -> None:
    chunker = Chunker(chunk_size=64, chunk_overlap=8, strategy=ChunkingStrategy.TOKEN)
    chunks = chunker.split_documents([_sample_doc()])
    assert len(chunks) > 1
    assert all(chunk.metadata["chunking_strategy"] == "token" for chunk in chunks)


def test_semantic_chunking_produces_chunks() -> None:
    chunker = Chunker(chunk_size=64, chunk_overlap=0, strategy=ChunkingStrategy.SEMANTIC)
    chunks = chunker.split_documents([_sample_doc()])
    assert len(chunks) >= 1
    assert all(chunk.metadata["chunking_strategy"] == "semantic" for chunk in chunks)
