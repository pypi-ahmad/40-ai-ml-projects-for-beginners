"""Adapter registry and plugin loading."""

from __future__ import annotations

from collections.abc import Callable

from multimodal_ai.adapters.base import (
    BaseAdapter,
    DetectionAdapter,
    EmbeddingAdapter,
    LLMAdapter,
    OCRAdapter,
    VisionModelAdapter,
)

AdapterFactory = Callable[[], BaseAdapter]


class AdapterRegistry:
    """In-memory adapter registry for plugin architecture."""

    def __init__(self) -> None:
        self._vision_factories: dict[str, Callable[[], VisionModelAdapter]] = {}
        self._embedding_factories: dict[str, Callable[[], EmbeddingAdapter]] = {}
        self._ocr_factories: dict[str, Callable[[], OCRAdapter]] = {}
        self._detector_factories: dict[str, Callable[[], DetectionAdapter]] = {}
        self._llm_factories: dict[str, Callable[[], LLMAdapter]] = {}

    def register_vision(self, name: str, factory: Callable[[], VisionModelAdapter]) -> None:
        self._vision_factories[name] = factory

    def register_embedding(self, name: str, factory: Callable[[], EmbeddingAdapter]) -> None:
        self._embedding_factories[name] = factory

    def register_ocr(self, name: str, factory: Callable[[], OCRAdapter]) -> None:
        self._ocr_factories[name] = factory

    def register_detector(self, name: str, factory: Callable[[], DetectionAdapter]) -> None:
        self._detector_factories[name] = factory

    def register_llm(self, name: str, factory: Callable[[], LLMAdapter]) -> None:
        self._llm_factories[name] = factory

    def create_vision(self, name: str) -> VisionModelAdapter:
        return self._vision_factories[name]()

    def create_embedding(self, name: str) -> EmbeddingAdapter:
        return self._embedding_factories[name]()

    def create_ocr(self, name: str) -> OCRAdapter:
        return self._ocr_factories[name]()

    def create_detector(self, name: str) -> DetectionAdapter:
        return self._detector_factories[name]()

    def create_llm(self, name: str) -> LLMAdapter:
        return self._llm_factories[name]()

    def available(self) -> dict[str, list[str]]:
        """Return available adapters by class."""

        return {
            "vision": sorted(self._vision_factories),
            "embedding": sorted(self._embedding_factories),
            "ocr": sorted(self._ocr_factories),
            "detector": sorted(self._detector_factories),
            "llm": sorted(self._llm_factories),
        }
