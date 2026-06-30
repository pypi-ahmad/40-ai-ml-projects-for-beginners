from __future__ import annotations

from hybrid_research_assistant.cache import SemanticResponseCache


class _FixedEmbedder:
    model_name = "fixed"

    def embed_texts(self, texts, batch_size=32):  # noqa: ANN001, ANN201
        out = []
        for text in texts:
            length = float(len(text))
            out.append([length, 1.0, 0.5])
        return out

    def embedding_dimension(self) -> int:
        return 3


def test_semantic_cache_hit() -> None:
    cache = SemanticResponseCache(
        embedder=_FixedEmbedder(),
        similarity_threshold=0.9,
        ttl_local=3600,
        ttl_web=600,
    )
    cache.put("What is LangGraph?", "Answer", "local")
    hit = cache.get("What is LangGraph?")
    assert hit is not None
    assert hit.response == "Answer"
