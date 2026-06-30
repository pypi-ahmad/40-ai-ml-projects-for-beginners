"""Base adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAdapter(ABC):
    """Common adapter contract."""

    name: str

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Return adapter health status."""


class VisionModelAdapter(BaseAdapter):
    """Vision-language model adapter contract."""

    @abstractmethod
    def caption(self, image_path: str, style: str = "detailed") -> dict[str, Any]:
        """Generate image caption."""

    @abstractmethod
    def vqa(self, image_path: str, question: str) -> dict[str, Any]:
        """Answer question from image."""


class EmbeddingAdapter(BaseAdapter):
    """Embedding adapter contract."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed text."""

    @abstractmethod
    def embed_image(self, image_path: str) -> list[float]:
        """Embed image path or bytes."""


class OCRAdapter(BaseAdapter):
    """OCR adapter contract."""

    @abstractmethod
    def extract(self, path: str) -> dict[str, Any]:
        """Extract OCR content."""


class DetectionAdapter(BaseAdapter):
    """Object detection adapter contract."""

    @abstractmethod
    def detect(self, image_path: str) -> list[dict[str, Any]]:
        """Detect objects."""


class LLMAdapter(BaseAdapter):
    """LLM adapter contract."""

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """Complete prompt."""
