"""Multimodal embedding index and search pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from multimodal_ai.adapters.registry import AdapterRegistry
from multimodal_ai.domain import RetrievalHit
from multimodal_ai.storage.chroma_store import ChromaStore


class RetrievalPipeline:
    """Pipeline for indexing and semantic retrieval."""

    def __init__(
        self, registry: AdapterRegistry, chroma_store: ChromaStore, embedding_name: str
    ) -> None:
        self._registry = registry
        self._chroma = chroma_store
        self._embedding_name = embedding_name

    def index_image(self, image_path: str, metadata: dict[str, Any] | None = None) -> str:
        """Index image embedding in chroma."""

        embedding = self._registry.create_embedding(self._embedding_name)
        vector = embedding.embed_image(image_path)
        record_id = str(uuid4())
        payload = metadata or {}
        payload.setdefault("path", image_path)
        payload.setdefault("type", "image")
        self._chroma.upsert("image", record_id, vector, Path(image_path).name, payload)
        return record_id

    def index_text(
        self, text: str, modality: str = "document", metadata: dict[str, Any] | None = None
    ) -> str:
        """Index text embedding in chroma."""

        embedding = self._registry.create_embedding(self._embedding_name)
        vector = embedding.embed_text(text)
        record_id = str(uuid4())
        payload = metadata or {}
        payload.setdefault("type", modality)
        self._chroma.upsert(modality, record_id, vector, text, payload)
        return record_id

    def search(self, query: str, modality: str = "image", top_k: int = 5) -> list[RetrievalHit]:
        """Run semantic search by query text."""

        embedding = self._registry.create_embedding(self._embedding_name)
        vector = embedding.embed_text(query)
        return self._chroma.search(modality, vector, top_k=top_k)
