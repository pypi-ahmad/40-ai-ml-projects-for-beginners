from __future__ import annotations

from types import SimpleNamespace

import pytest

from rag_system.embeddings import EmbeddingEngine


class _FailLargeBatchClient:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed(self, model: str, input):  # noqa: A002
        if isinstance(input, list):
            self.batch_sizes.append(len(input))
            if len(input) > 2:
                raise TimeoutError("timed out")
            vectors = [[float(len(text))] for text in input]
            return SimpleNamespace(embeddings=vectors)

        return SimpleNamespace(embeddings=[[float(len(input))]])


class _AlwaysFailClient:
    def embed(self, model: str, input):  # noqa: A002
        raise TimeoutError("timed out")


def test_embed_batch_falls_back_by_splitting_large_batches() -> None:
    engine = EmbeddingEngine(model_name="qwen3-embedding:4b", max_retries=1)
    client = _FailLargeBatchClient()
    engine.client = client

    vectors = engine.embed_batch(["a", "bb", "ccc", "dddd"], batch_size=4)

    assert vectors == [[1.0], [2.0], [3.0], [4.0]]
    assert client.batch_sizes == [4, 2, 2]


def test_embed_batch_raises_when_single_item_keeps_failing() -> None:
    engine = EmbeddingEngine(model_name="qwen3-embedding:4b", max_retries=1)
    engine.client = _AlwaysFailClient()

    with pytest.raises(RuntimeError, match="Batch embedding failed"):
        engine.embed_batch(["only one"], batch_size=1)
