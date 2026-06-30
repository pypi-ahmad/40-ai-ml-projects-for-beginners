from __future__ import annotations

from local_rag.retriever import Retriever
from local_rag.types import RetrievalResult


class FakeEmbedder:
    def embed_texts(self, texts):
        assert len(texts) == 1
        return [[0.1, 0.2, 0.3]]


class FakeStore:
    def __init__(self) -> None:
        self.last_top_k = None
        self.last_where = None

    def query(self, query_embedding, top_k, where=None):
        self.last_top_k = top_k
        self.last_where = where
        return [
            RetrievalResult(
                chunk_id="c1",
                doc_id="d1",
                text="hello",
                score=0.9,
                metadata={"source_path": "x"},
            )
        ]


def test_retrieve_passes_k_and_filter() -> None:
    store = FakeStore()
    retriever = Retriever(vector_store=store, embedder=FakeEmbedder())

    hits, latency = retriever.retrieve("q", top_k=7, filters={"source_type": "txt"})
    assert len(hits) == 1
    assert store.last_top_k == 7
    assert store.last_where == {"source_type": "txt"}
    assert latency >= 0.0
