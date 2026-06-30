from __future__ import annotations

from hybrid_research_assistant.retrieval import IntentRouter, RetrievalService
from hybrid_research_assistant.schemas import RetrievalMode, RetrievedContext


class _DummyEmbedder:
    model_name = "dummy"

    def embed_texts(self, texts, batch_size=32):  # noqa: ANN001, ANN201
        return [[0.1, 0.2, 0.3] for _ in texts]

    def embedding_dimension(self) -> int:
        return 3


class _DummyStore:
    def query(self, query_embedding, top_k, where=None):  # noqa: ANN001, ANN201
        return [
            RetrievedContext(
                chunk_id="c1",
                doc_id="d1",
                text="local chunk",
                score=0.9,
                metadata={"source": "local.md", "doc_id": "d1"},
                source="local",
            )
        ]


class _DummyWeb:
    async def search(self, query, k, provider=None):  # noqa: ANN001, ANN201
        return []


def test_router_modes() -> None:
    router = IntentRouter()
    assert router.route("What happened in AI yesterday?", RetrievalMode.AUTO).mode == RetrievalMode.WEB
    assert router.route("Compare latest updates with internal policy", RetrievalMode.AUTO).mode == RetrievalMode.HYBRID
    assert router.route("What is LangGraph?", RetrievalMode.AUTO).mode == RetrievalMode.LOCAL


def test_hybrid_dedup_merge() -> None:
    rows = [
        RetrievedContext("c1", "d1", "a", 0.6, {"source": "x", "url": ""}, "local"),
        RetrievedContext("c2", "d2", "b", 0.8, {"source": "x", "url": ""}, "web"),
    ]
    merged = RetrievalService.merge_and_deduplicate(rows)
    assert len(merged) == 1
    assert merged[0].chunk_id == "c2"


def test_mmr_select() -> None:
    rows = [
        RetrievedContext("c1", "d1", "a", 0.9, {"source": "s1"}, "local"),
        RetrievedContext("c2", "d1", "b", 0.8, {"source": "s1"}, "local"),
        RetrievedContext("c3", "d2", "c", 0.7, {"source": "s2"}, "local"),
    ]
    selected = RetrievalService.mmr_select(rows, top_k=2, lambda_mult=0.5)
    assert len(selected) == 2
    assert selected[0].chunk_id == "c1"
