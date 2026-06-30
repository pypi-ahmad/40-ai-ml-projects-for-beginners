from __future__ import annotations

from local_rag.rag import RAGPipeline
from local_rag.types import RetrievalResult


class FakeRetriever:
    def retrieve(self, query: str, *, top_k: int, filters=None):
        del query, top_k, filters
        hits = [
            RetrievalResult(
                chunk_id="chunk-1",
                doc_id="doc-1",
                text="context text",
                score=0.93,
                metadata={"source_path": "docs/a.txt"},
            )
        ]
        return hits, 3.0


class FakeGenerator:
    model = "qwen3.5:4b"

    def generate(self, messages, *, temperature=None, max_tokens=None):
        del messages, temperature, max_tokens
        return "answer", 12.0

    def stream_generate(self, messages, *, temperature=None, max_tokens=None):
        del messages, temperature, max_tokens
        yield from ["ans", "wer"]


def test_stream_path_returns_finalized_response() -> None:
    pipeline = RAGPipeline(retriever=FakeRetriever(), generator=FakeGenerator())
    stream_iter, session = pipeline.ask_stream("question", top_k=5)
    answer = "".join(stream_iter)
    response = pipeline.finalize_stream(session=session, answer=answer, generation_ms=10.0)

    assert response.answer == "answer"
    assert response.model == "qwen3.5:4b"
    assert response.top_k == 5
    assert len(response.citations) == 1
    assert response.citations[0]["source_path"] == "docs/a.txt"
