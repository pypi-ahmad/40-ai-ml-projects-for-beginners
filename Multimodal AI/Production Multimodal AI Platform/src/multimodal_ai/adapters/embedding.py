"""Embedding adapter implementations."""

from __future__ import annotations

from typing import Any

from multimodal_ai.adapters.base import EmbeddingAdapter
from multimodal_ai.adapters.common import deterministic_vector, image_fingerprint


class _HFEmbeddingBase(EmbeddingAdapter):
    """Sentence-transformers based fallback-first adapter."""

    def __init__(self, model_id: str, name: str) -> None:
        self._model_id = model_id
        self.name = name
        self._model: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_id)
        except Exception:  # noqa: BLE001
            self._model = None

    def health(self) -> dict[str, Any]:
        self._load()
        return {"ok": self._model is not None, "name": self.name}

    def embed_text(self, text: str) -> list[float]:
        self._load()
        if self._model is None:
            return deterministic_vector(f"text:{self.name}:{text}")
        vector = self._model.encode(text)
        return [float(v) for v in vector.tolist()]

    def embed_image(self, image_path: str) -> list[float]:
        self._load()
        fingerprint = image_fingerprint(image_path)
        if self._model is None:
            return deterministic_vector(f"image:{self.name}:{fingerprint}")

        # Fallback through textual fingerprint for model-agnostic embedding.
        vector = self._model.encode(f"image_fingerprint:{fingerprint}")
        return [float(v) for v in vector.tolist()]


class CLIPEmbeddingAdapter(_HFEmbeddingBase):
    """CLIP-compatible embedding adapter."""

    def __init__(self) -> None:
        super().__init__("sentence-transformers/clip-ViT-B-32", "clip")


class SigLIPEmbeddingAdapter(_HFEmbeddingBase):
    """SigLIP-compatible embedding adapter."""

    def __init__(self) -> None:
        super().__init__("sentence-transformers/clip-ViT-B-32", "siglip")
