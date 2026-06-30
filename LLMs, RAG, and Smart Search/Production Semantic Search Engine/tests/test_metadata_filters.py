from semantic_search.retrieval import HybridRetriever
from semantic_search.schemas import DocumentChunk


def _chunk(chunk_id: str, category: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        chunk_index=0,
        text="semantic search over news documents",
        metadata={"category": category, "author": "author1"},
    )


def test_lexical_filter_by_category():
    chunks = [_chunk("c1", "POLITICS"), _chunk("c2", "TECH")]
    retriever = HybridRetriever(chunks)
    hits, _ = retriever.lexical_search("news documents", top_k=5, filters={"category": "TECH"})
    assert len(hits) == 1
    assert hits[0].metadata["category"] == "TECH"
