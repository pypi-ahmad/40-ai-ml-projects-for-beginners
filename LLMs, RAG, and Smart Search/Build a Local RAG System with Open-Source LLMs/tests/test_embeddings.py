from __future__ import annotations

import pytest

from local_rag.embeddings import OllamaEmbeddingClient


class _StubEmbedClient:
    def __init__(self, responses: list[object]) -> None:
        self._responses = responses
        self.calls = 0

    def embed(self, *, model: str, input: list[str]) -> dict[str, list[list[float]]]:
        response = self._responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return {"embeddings": response}


def test_embed_retries_and_succeeds_after_transient_error() -> None:
    client = OllamaEmbeddingClient(
        model="qwen3-embedding:4b",
        host="http://127.0.0.1:11434",
        normalize=False,
        max_retries=3,
        retry_backoff_seconds=0.0,
    )
    stub = _StubEmbedClient(
        responses=[
            RuntimeError("EOF"),
            [[0.1, 0.2], [0.3, 0.4]],
        ]
    )
    client.client = stub  # type: ignore[assignment]

    vectors = client.embed_texts(["a", "b"], batch_size=2)
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert stub.calls == 2


def test_embed_raises_after_retry_exhaustion() -> None:
    client = OllamaEmbeddingClient(
        model="qwen3-embedding:4b",
        host="http://127.0.0.1:11434",
        normalize=False,
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    stub = _StubEmbedClient(
        responses=[
            RuntimeError("EOF"),
            RuntimeError("EOF"),
        ]
    )
    client.client = stub  # type: ignore[assignment]

    with pytest.raises(RuntimeError):
        client.embed_texts(["a"], batch_size=1)
    assert stub.calls == 2

