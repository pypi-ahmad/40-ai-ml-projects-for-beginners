"""Cross-encoder reranking."""

from __future__ import annotations

from time import perf_counter

from semantic_search.logging_utils import get_logger
from semantic_search.schemas import SearchHit

logger = get_logger()


class CrossEncoderReranker:
    """Rerank hits using sentence-transformers cross encoder."""

    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        try:
            self.model = CrossEncoder(model_name, local_files_only=True, device="cpu")
        except Exception:  # noqa: BLE001
            self.model = CrossEncoder(model_name, device="cpu")

    def rerank(self, query: str, hits: list[SearchHit], top_n: int | None = None) -> tuple[list[SearchHit], float]:
        """Rerank candidate hits and return latency."""
        if not hits:
            return hits, 0.0

        start = perf_counter()
        pairs = [[query, hit.text] for hit in hits]
        scores = self.model.predict(pairs)

        reranked = []
        for hit, score in zip(hits, scores, strict=False):
            reranked.append(
                hit.model_copy(
                    update={
                        "rerank_score": float(score),
                        "score": float(score),
                    }
                )
            )
        reranked.sort(key=lambda x: x.score, reverse=True)

        if top_n is not None:
            reranked = reranked[:top_n]

        output = [h.model_copy(update={"rank": idx + 1}) for idx, h in enumerate(reranked)]
        latency_ms = (perf_counter() - start) * 1000
        logger.info("rerank_complete", candidates=len(hits), latency_ms=latency_ms)
        return output, latency_ms
