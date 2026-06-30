"""Service bootstrap and dependency graph setup."""

from __future__ import annotations

import os

from multimodal_ai.adapters.detection import GroundingDINODetectionAdapter, YOLODetectionAdapter
from multimodal_ai.adapters.embedding import CLIPEmbeddingAdapter, SigLIPEmbeddingAdapter
from multimodal_ai.adapters.llm import HFTextGenerationAdapter, OllamaLLMAdapter
from multimodal_ai.adapters.ocr import (
    EasyOCRAdapter,
    GLMOcrAdapter,
    PaddleOCRAdapter,
    TesseractOCRAdapter,
)
from multimodal_ai.adapters.registry import AdapterRegistry
from multimodal_ai.adapters.vision import (
    BLIP2Adapter,
    Florence2Adapter,
    LlamaVisionAdapter,
    MiniCPMVAdapter,
    OllamaVisionAdapter,
    Qwen25VLAdapter,
    SmolVLMAdapter,
)
from multimodal_ai.analytics.metrics import MetricsCollector
from multimodal_ai.config.settings import PlatformConfig, load_config
from multimodal_ai.mcp.external import ExternalMCPHookRegistry
from multimodal_ai.observability.logging import get_logger
from multimodal_ai.pipelines.document_pipeline import DocumentPipeline
from multimodal_ai.pipelines.rag_pipeline import MultimodalRAGPipeline
from multimodal_ai.pipelines.retrieval_pipeline import RetrievalPipeline
from multimodal_ai.services.platform_service import PlatformService
from multimodal_ai.storage.chroma_store import ChromaStore
from multimodal_ai.storage.sqlite_store import SQLiteStore


def build_registry(config: PlatformConfig) -> AdapterRegistry:
    """Build adapter registry with built-in plugins."""

    registry = AdapterRegistry()

    use_ollama_vision = os.getenv("MM_USE_OLLAMA_VISION", "false").lower() == "true"

    registry.register_embedding("clip", CLIPEmbeddingAdapter)
    registry.register_embedding("siglip", SigLIPEmbeddingAdapter)

    registry.register_ocr("glm_ocr", GLMOcrAdapter)
    registry.register_ocr("easyocr", EasyOCRAdapter)
    registry.register_ocr("paddleocr", PaddleOCRAdapter)
    registry.register_ocr("tesseract", TesseractOCRAdapter)

    registry.register_detector("yolo", YOLODetectionAdapter)
    registry.register_detector("grounding_dino", GroundingDINODetectionAdapter)

    registry.register_llm("ollama", lambda: OllamaLLMAdapter(model=config.default_llm_model))
    registry.register_llm("hf", HFTextGenerationAdapter)

    if use_ollama_vision:
        registry.register_vision("qwen2_5_vl", lambda: OllamaVisionAdapter("qwen2.5vl:7b"))
        registry.register_vision("llama_vision", lambda: OllamaVisionAdapter("llama3.2-vision"))
    else:
        registry.register_vision("qwen2_5_vl", Qwen25VLAdapter)
        registry.register_vision("llama_vision", LlamaVisionAdapter)

    registry.register_vision("florence_2", Florence2Adapter)
    registry.register_vision("minicpm_v", MiniCPMVAdapter)
    registry.register_vision("blip2", BLIP2Adapter)
    registry.register_vision("smolvlm", SmolVLMAdapter)
    registry.register_vision("clip", Florence2Adapter)
    registry.register_vision("siglip", Florence2Adapter)

    return registry


def build_platform_service(config_path: str = "configs/config.yaml") -> PlatformService:
    """Build full platform service graph."""

    config = load_config(config_path)
    logger = get_logger()

    registry = build_registry(config)

    sqlite_store = SQLiteStore(config.sqlite_url)
    sqlite_store.create_tables()

    chroma_store = ChromaStore(config.chroma_path)

    document_pipeline = DocumentPipeline(
        registry=registry,
        primary_engine=config.ocr_primary_engine,
        min_text_chars=40,
    )
    retrieval_pipeline = RetrievalPipeline(
        registry=registry,
        chroma_store=chroma_store,
        embedding_name=config.default_embedding_model,
    )
    rag_pipeline = MultimodalRAGPipeline(
        registry=registry,
        document_pipeline=document_pipeline,
        retrieval_pipeline=retrieval_pipeline,
        llm_name=config.default_llm_backend,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )

    metrics = MetricsCollector()
    external_hooks = ExternalMCPHookRegistry()

    logger.info("Platform service bootstrapped")
    return PlatformService(
        config=config,
        registry=registry,
        sqlite_store=sqlite_store,
        retrieval_pipeline=retrieval_pipeline,
        document_pipeline=document_pipeline,
        rag_pipeline=rag_pipeline,
        metrics=metrics,
        external_mcp_hooks=external_hooks,
    )
