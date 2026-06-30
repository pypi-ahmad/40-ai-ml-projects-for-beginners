from semantic_search.chunking import Chunker, ChunkingParams
from semantic_search.schemas import DocumentRecord
from semantic_search.utils import hash_text


def _doc(text: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id="d1",
        source="test",
        text=text,
        document_hash=hash_text(text),
    )


def test_recursive_chunking_produces_multiple_chunks():
    text = "word " * 400
    chunker = Chunker(ChunkingParams(strategy="recursive", chunk_size=100, chunk_overlap=10))
    chunks = chunker.chunk_documents([_doc(text)])
    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0


def test_token_chunking_respects_size():
    text = " ".join([f"token{i}" for i in range(40)])
    chunker = Chunker(ChunkingParams(strategy="token", chunk_size=10, chunk_overlap=2))
    chunks = chunker.chunk_documents([_doc(text)])
    assert len(chunks) >= 4


def test_semantic_chunking_returns_text():
    text = "Sentence one about AI. Sentence two about embeddings. Sentence three about search."
    chunker = Chunker(ChunkingParams(strategy="semantic", chunk_size=25, chunk_overlap=2))
    chunks = chunker.chunk_documents([_doc(text)])
    assert len(chunks) >= 1
    assert all(chunk.text for chunk in chunks)
