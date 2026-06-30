from rag_system.chunking import ChunkingStrategy, TextChunker
from rag_system.types import DocumentRecord


def sample_doc() -> DocumentRecord:
    text = " ".join([f"Sentence {i}." for i in range(1, 80)])
    return DocumentRecord(doc_id="doc_1", text=text, metadata={"title": "Sample"})


def test_fixed_chunking_produces_multiple_chunks() -> None:
    chunker = TextChunker(strategy=ChunkingStrategy.FIXED, chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk_document(sample_doc())

    assert len(chunks) > 1
    assert all(chunk.metadata["strategy"] == "fixed" for chunk in chunks)


def test_recursive_chunking_metadata_present() -> None:
    chunker = TextChunker(strategy=ChunkingStrategy.RECURSIVE, chunk_size=180, chunk_overlap=40)
    chunks = chunker.chunk_document(sample_doc())

    assert len(chunks) >= 1
    assert all("chunk_index" in chunk.metadata for chunk in chunks)


def test_parent_child_chunking_contains_parent_links() -> None:
    chunker = TextChunker(
        strategy=ChunkingStrategy.PARENT_CHILD,
        chunk_size=120,
        chunk_overlap=20,
        parent_chunk_size=260,
        parent_chunk_overlap=40,
    )
    chunks = chunker.chunk_document(sample_doc())

    parent_chunks = [chunk for chunk in chunks if chunk.metadata.get("level") == "parent"]
    child_chunks = [chunk for chunk in chunks if chunk.metadata.get("level") == "child"]

    assert parent_chunks
    assert child_chunks
    assert all(child.parent_id is not None for child in child_chunks)
