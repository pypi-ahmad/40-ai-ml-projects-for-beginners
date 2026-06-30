from __future__ import annotations

from local_rag.evaluator import RetrievalEvaluator
from local_rag.types import EvalExample, RetrievalResult


class FakeRetriever:
    def retrieve(self, query: str, top_k: int, filters=None):
        del query, top_k, filters
        return [
            RetrievalResult(
                chunk_id="chunk-a",
                doc_id="doc-a",
                text="a",
                score=0.9,
                metadata={},
            ),
            RetrievalResult(
                chunk_id="chunk-b",
                doc_id="doc-b",
                text="b",
                score=0.8,
                metadata={},
            ),
        ], 5.0


class DuplicateDocRetriever:
    def retrieve(self, query: str, top_k: int, filters=None):
        del query, top_k, filters
        return [
            RetrievalResult(
                chunk_id="chunk-a1",
                doc_id="doc-a",
                text="a1",
                score=0.95,
                metadata={},
            ),
            RetrievalResult(
                chunk_id="chunk-a2",
                doc_id="doc-a",
                text="a2",
                score=0.9,
                metadata={},
            ),
            RetrievalResult(
                chunk_id="chunk-x",
                doc_id="doc-x",
                text="x",
                score=0.4,
                metadata={},
            ),
        ], 4.0


def test_retrieval_evaluator_supports_chunk_level_relevance() -> None:
    evaluator = RetrievalEvaluator(FakeRetriever())
    metrics = evaluator.evaluate(
        [
            EvalExample(
                query="q",
                relevant_doc_ids=[],
                relevant_chunk_ids=["chunk-b"],
            )
        ],
        ks=(2,),
    )
    assert len(metrics) == 1
    assert metrics[0].mrr == 0.5


def test_retrieval_metrics_bound_with_duplicate_doc_hits() -> None:
    evaluator = RetrievalEvaluator(DuplicateDocRetriever())
    metrics = evaluator.evaluate(
        [
            EvalExample(
                query="q",
                relevant_doc_ids=["doc-a"],
                relevant_chunk_ids=[],
            )
        ],
        ks=(3,),
    )
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.precision_at_k == 1 / 3
    assert metric.recall_at_k == 1.0
    assert metric.ndcg == 1.0
    assert metric.mrr == 1.0
