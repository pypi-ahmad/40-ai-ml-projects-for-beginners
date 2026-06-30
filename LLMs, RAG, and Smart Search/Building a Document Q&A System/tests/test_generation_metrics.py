from __future__ import annotations

from local_rag.evaluator import GenerationEvaluator
from local_rag.types import EvalExample, RetrievalResult, TimingBreakdown


class FakePipeline:
    def ask(self, query: str, *, top_k: int, strategy="hybrid"):
        del query, top_k, strategy

        class _Response:
            answer = "Encryption policy requires annual key rotation."
            citations = [{"chunk_id": "c1"}]
            retrieved = [
                RetrievalResult(
                    chunk_id="c1",
                    doc_id="d1",
                    text="Annual key rotation is required by encryption policy.",
                    score=0.9,
                    metadata={"source_path": "policy.txt", "document_name": "policy.txt"},
                    strategy="hybrid",
                    vector_score=0.9,
                    keyword_score=0.8,
                )
            ]
            timings = TimingBreakdown(
                embedding_ms=1.0,
                retrieval_ms=2.0,
                prompt_ms=1.0,
                generation_ms=3.0,
                total_ms=7.0,
            )

        return _Response()


def test_generation_evaluator_outputs_metrics() -> None:
    evaluator = GenerationEvaluator(FakePipeline())
    metric = evaluator.evaluate(
        [
            EvalExample(
                query="What does policy require?",
                relevant_doc_ids=["d1"],
                answer="Policy requires annual key rotation.",
            )
        ],
        top_k=5,
    )
    assert metric.strategy == "hybrid"
    assert metric.avg_generation_latency_ms > 0
    assert metric.avg_citation_count >= 1
