from types import SimpleNamespace

from rag_system.generation import GenerationEngine, RAGPipeline
from rag_system.types import RetrievedChunk


class FakeClient:
    def __init__(self) -> None:
        self.last_think = None

    def chat(self, **kwargs):
        self.last_think = kwargs.get("think")
        message = SimpleNamespace(content="Grounded answer", thinking="")
        return SimpleNamespace(
            message=message,
            done_reason="stop",
            prompt_eval_count=10,
            eval_count=12,
        )


class DummyRetriever:
    def query(self, query: str, top_k: int = 6, metadata_filter=None):
        return [
            RetrievedChunk(
                chunk_id="c1",
                doc_id="d1",
                text="weak context",
                score=0.1,
                distance=1.8,
                metadata={"title": "T"},
            )
        ]

    def format_context(self, chunks):
        return ("[1] weak context", ["[1]"])


def test_generation_engine_uses_think_false_by_default() -> None:
    engine = GenerationEngine(model_name="qwen3.5:4b")
    fake = FakeClient()
    engine.client = fake

    out = engine.generate(messages=[{"role": "user", "content": "Hello"}])

    assert fake.last_think is False
    assert out["text"] == "Grounded answer"


def test_pipeline_abstains_on_low_relevance() -> None:
    engine = GenerationEngine(model_name="qwen3.5:4b")
    fake = FakeClient()
    engine.client = fake
    pipeline = RAGPipeline(retrieval_engine=DummyRetriever(), generation_engine=engine, abstain_threshold=0.2)

    out = pipeline.answer("what is this?")
    assert out.abstained is True
    assert "cannot find enough evidence" in out.answer.lower()
