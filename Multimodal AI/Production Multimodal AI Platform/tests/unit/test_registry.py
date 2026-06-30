"""Adapter registry tests."""

from __future__ import annotations

from multimodal_ai.adapters.base import EmbeddingAdapter
from multimodal_ai.adapters.registry import AdapterRegistry


class DummyEmbedding(EmbeddingAdapter):
    name = "dummy"

    def health(self) -> dict:
        return {"ok": True}

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 2.0]

    def embed_image(self, image_path: str) -> list[float]:
        return [3.0, 4.0]


def test_registry_register_and_create() -> None:
    registry = AdapterRegistry()
    registry.register_embedding("dummy", DummyEmbedding)

    adapter = registry.create_embedding("dummy")
    assert adapter.embed_text("hello") == [1.0, 2.0]
    assert "dummy" in registry.available()["embedding"]
