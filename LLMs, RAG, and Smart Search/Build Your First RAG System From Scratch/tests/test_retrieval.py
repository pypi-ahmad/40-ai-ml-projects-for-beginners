from rag_system.retrieval import RetrievalEngine
from rag_system.types import ChunkRecord


class DummyEmbeddingEngine:
    def __init__(self):
        self.model_name = "dummy"

    def embed(self, text: str) -> list[float]:
        # Tiny deterministic embedding for test stability.
        if "cat" in text:
            return [1.0, 0.0, 0.0]
        if "dog" in text:
            return [0.0, 1.0, 0.0]
        return [0.2, 0.2, 0.6]

    def embed_batch(self, texts, batch_size=32):
        return [self.embed(text) for text in texts]


def test_retrieval_query_returns_ranked_chunks() -> None:
    engine = RetrievalEngine(
        collection_name="test_retrieval_docs",
        persist_directory=None,
        embedding_engine=DummyEmbeddingEngine(),
        default_top_k=2,
    )
    engine.clear()

    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="d1", text="cat facts", metadata={"title": "Cats"}),
        ChunkRecord(chunk_id="c2", doc_id="d2", text="dog facts", metadata={"title": "Dogs"}),
    ]
    engine.index_chunks(chunks)

    hits = engine.query("tell me about cat", top_k=2)
    assert len(hits) == 2
    assert hits[0].chunk_id == "c1"
    assert hits[0].score >= hits[1].score


def test_retrieval_classification() -> None:
    engine = RetrievalEngine(
        collection_name="test_retrieval_classify",
        persist_directory=None,
        embedding_engine=DummyEmbeddingEngine(),
        default_top_k=2,
    )
    engine.clear()
    engine.index_chunks(
        [
            ChunkRecord(chunk_id="c1", doc_id="d1", text="cat facts", metadata={"title": "Cats"}),
            ChunkRecord(chunk_id="c2", doc_id="d2", text="dog facts", metadata={"title": "Dogs"}),
        ]
    )

    hits = engine.query("cat", top_k=2, dedupe_by_doc=True)
    bucket = engine.classify_retrieval(chunks=hits, gold_doc_ids=["d1"], min_relevance_score=0.1)
    assert bucket == "hit_top1"
