"""Cross-encoder reranker and fallback reranking."""

from __future__ import annotations

import time
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from hybrid_research_assistant.schemas import RetrievedContext


@dataclass(slots=True)
class RerankReport:
    """Rerank timing and quality summary."""

    before_scores: list[float]
    after_scores: list[float]
    latency_ms: float


class Reranker:
    """Cross-encoder reranker with safe lexical fallback."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self._disabled = False

    def _get_model(self) -> CrossEncoder:
        if self._disabled:
            raise RuntimeError("reranker_model_unavailable")
        if self._model is None:
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as err:  # noqa: BLE001
                self._disabled = True
                raise RuntimeError("reranker_model_unavailable") from err
        return self._model

    def rerank(self, query: str, rows: list[RetrievedContext], top_k: int) -> tuple[list[RetrievedContext], RerankReport]:
        """Rerank contexts and return sorted rows."""

        started = time.perf_counter()
        before = [row.score for row in rows]

        try:
            model = self._get_model()
            pairs = [[query, row.text] for row in rows]
            scores = list(model.predict(pairs))
        except Exception:  # noqa: BLE001
            query_terms = {term for term in query.lower().split() if term}
            scores = [
                float(sum(1 for term in query_terms if term in row.text.lower()))
                for row in rows
            ]

        rescored: list[RetrievedContext] = []
        for row, score in zip(rows, scores, strict=False):
            rescored.append(
                RetrievedContext(
                    chunk_id=row.chunk_id,
                    doc_id=row.doc_id,
                    text=row.text,
                    score=float(score),
                    metadata=dict(row.metadata),
                    source=row.source,
                )
            )

        ranked = sorted(rescored, key=lambda item: item.score, reverse=True)[:top_k]
        latency_ms = (time.perf_counter() - started) * 1000
        report = RerankReport(before_scores=before, after_scores=[row.score for row in ranked], latency_ms=latency_ms)
        return ranked, report
