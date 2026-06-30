"""Main orchestrator service for all platform capabilities."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from multimodal_ai.adapters.registry import AdapterRegistry
from multimodal_ai.analytics.metrics import MetricsCollector
from multimodal_ai.analytics.system_monitor import collect_system_stats
from multimodal_ai.config.settings import PlatformConfig
from multimodal_ai.domain import RequestEnvelope, ResponseEnvelope
from multimodal_ai.mcp.external import ExternalMCPHookRegistry
from multimodal_ai.observability.logging import get_logger
from multimodal_ai.pipelines.document_pipeline import DocumentPipeline
from multimodal_ai.pipelines.rag_pipeline import MultimodalRAGPipeline
from multimodal_ai.pipelines.retrieval_pipeline import RetrievalPipeline
from multimodal_ai.storage.sqlite_store import SQLiteStore
from multimodal_ai.utils.files import infer_media_type


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class PlatformService:
    """Top-level multimodal service used by API, CLI, UI, and MCP."""

    def __init__(
        self,
        config: PlatformConfig,
        registry: AdapterRegistry,
        sqlite_store: SQLiteStore,
        retrieval_pipeline: RetrievalPipeline,
        document_pipeline: DocumentPipeline,
        rag_pipeline: MultimodalRAGPipeline,
        metrics: MetricsCollector,
        external_mcp_hooks: ExternalMCPHookRegistry | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._sqlite = sqlite_store
        self._retrieval = retrieval_pipeline
        self._docs = document_pipeline
        self._rag = rag_pipeline
        self._metrics = metrics
        self._external_mcp_hooks = external_mcp_hooks or ExternalMCPHookRegistry()
        self._logger = get_logger("multimodal_ai.service")

    def _record(
        self, action: str, start: float, trace_id: str, model_name: str | None = None
    ) -> float:
        latency_ms = (time.perf_counter() - start) * 1000.0
        self._metrics.record(action, latency_ms)
        self._sqlite.add_event(trace_id=trace_id, action=action, latency_ms=latency_ms)
        if model_name:
            self._sqlite.bump_model_usage(
                model_name=model_name, capability=action, latency_ms=latency_ms
            )
        return latency_ms

    def _response(
        self,
        trace_id: str,
        result: dict[str, Any],
        latency_ms: float,
        confidence: float | None = None,
    ) -> ResponseEnvelope:
        return ResponseEnvelope(
            status="ok",
            result=result,
            confidence=confidence,
            latency_ms=latency_ms,
            artifacts={},
            trace_id=trace_id,
        )

    def health(self) -> ResponseEnvelope:
        """Return platform health status."""

        trace_id = str(uuid4())
        start = time.perf_counter()
        status = {
            "adapters": self._registry.available(),
            "external_mcp_tools": self._external_mcp_hooks.available(),
            "system": collect_system_stats(),
        }
        latency = self._record("health", start, trace_id)
        return self._response(trace_id, status, latency, confidence=1.0)

    def caption(self, request: RequestEnvelope) -> ResponseEnvelope:
        """Generate image captions."""

        trace_id = request.trace.trace_id
        start = time.perf_counter()
        image_path = request.input.image_path or ""
        style = str(request.options.get("style", "detailed"))
        model_name = request.model_overrides.get("vision", self._config.default_vision_model)

        vision = self._registry.create_vision(model_name)
        payload = vision.caption(image_path=image_path, style=style)

        asset_id = str(uuid4())
        row_id = self._sqlite.add_asset(asset_id=asset_id, path=image_path, media_type="image")
        self._sqlite.add_caption(
            asset_row_id=row_id,
            style=style,
            content=payload.get("caption", ""),
            confidence=float(payload.get("confidence", 0.0)),
        )

        latency = self._record("caption", start, trace_id, model_name=model_name)
        return self._response(trace_id, payload, latency, confidence=payload.get("confidence"))

    def ocr(self, request: RequestEnvelope) -> ResponseEnvelope:
        """Extract OCR text and layout."""

        trace_id = request.trace.trace_id
        start = time.perf_counter()
        file_path = request.input.document_path or request.input.image_path or ""

        result = self._docs.run(file_path)

        asset_id = str(uuid4())
        media_type = infer_media_type(Path(file_path))
        row_id = self._sqlite.add_asset(asset_id=asset_id, path=file_path, media_type=media_type)
        self._sqlite.add_ocr(asset_row_id=row_id, engine=result.engine, text=result.text)

        latency = self._record("ocr", start, trace_id, model_name=result.engine)
        return self._response(trace_id, result.model_dump(), latency, confidence=0.8)

    def embeddings(self, request: RequestEnvelope) -> ResponseEnvelope:
        """Generate text or image embeddings."""

        trace_id = request.trace.trace_id
        start = time.perf_counter()

        model_name = request.model_overrides.get("embedding", self._config.default_embedding_model)
        embedding = self._registry.create_embedding(model_name)

        text = request.input.text
        image_path = request.input.image_path
        if text:
            vector = embedding.embed_text(text)
            result = {"type": "text", "dimension": len(vector), "vector_preview": vector[:8]}
        elif image_path:
            vector = embedding.embed_image(image_path)
            result = {"type": "image", "dimension": len(vector), "vector_preview": vector[:8]}
        else:
            vector = []
            result = {"type": "none", "dimension": 0, "vector_preview": []}

        latency = self._record("embeddings", start, trace_id, model_name=model_name)
        return self._response(trace_id, result, latency, confidence=0.9 if vector else 0.0)

    def search(self, request: RequestEnvelope) -> ResponseEnvelope:
        """Run semantic search across indexed assets."""

        trace_id = request.trace.trace_id
        start = time.perf_counter()

        query = request.input.query or request.input.text or ""
        modality = str(request.options.get("modality", "image"))
        top_k = int(request.options.get("top_k", self._config.retrieval_top_k))

        hits = self._retrieval.search(query=query, modality=modality, top_k=top_k)
        payload = {
            "query": query,
            "modality": modality,
            "top_k": top_k,
            "hits": [hit.model_dump() for hit in hits],
        }

        latency = self._record(
            "search", start, trace_id, model_name=self._config.default_embedding_model
        )
        confidence = hits[0].score if hits else 0.0
        return self._response(trace_id, payload, latency, confidence=confidence)

    def retrieve(self, request: RequestEnvelope) -> ResponseEnvelope:
        """Alias to semantic search endpoint."""

        return self.search(request)

    def vqa(self, request: RequestEnvelope) -> ResponseEnvelope:
        """Perform visual question answering."""

        trace_id = request.trace.trace_id
        start = time.perf_counter()

        question = request.input.question or ""
        image_path = request.input.image_path
        doc_path = request.input.document_path

        if image_path:
            model_name = request.model_overrides.get("vision", self._config.default_vision_model)
            vision = self._registry.create_vision(model_name)
            payload = vision.vqa(image_path=image_path, question=question)
            confidence = float(payload.get("confidence", 0.0))
            latency = self._record("vqa", start, trace_id, model_name=model_name)
            return self._response(trace_id, payload, latency, confidence=confidence)

        rag_payload = self._rag.answer(question=question, path=doc_path)
        latency = self._record("vqa", start, trace_id, model_name=self._config.default_llm_backend)
        return self._response(
            trace_id, rag_payload, latency, confidence=rag_payload.get("confidence", 0.0)
        )

    def compare(self, request: RequestEnvelope) -> ResponseEnvelope:
        """Compare two or more images."""

        trace_id = request.trace.trace_id
        start = time.perf_counter()

        paths = request.input.image_paths
        if len(paths) < 2:
            latency = self._record("compare", start, trace_id)
            return self._response(
                trace_id, {"error": "Need at least two images"}, latency, confidence=0.0
            )

        embedding_model = request.model_overrides.get(
            "embedding", self._config.default_embedding_model
        )
        embedding = self._registry.create_embedding(embedding_model)
        vectors = [embedding.embed_image(path) for path in paths]

        scores: list[dict[str, Any]] = []
        for idx in range(1, len(vectors)):
            score = _cosine_similarity(vectors[0], vectors[idx])
            scores.append({"pair": [paths[0], paths[idx]], "similarity": score})

        avg_similarity = sum(float(item["similarity"]) for item in scores) / max(len(scores), 1)
        payload = {
            "scores": scores,
            "summary": "Images are visually close"
            if avg_similarity > 0.7
            else "Images differ materially",
            "quality_assessment": "high" if avg_similarity > 0.8 else "moderate",
        }

        latency = self._record("compare", start, trace_id, model_name=embedding_model)
        return self._response(trace_id, payload, latency, confidence=avg_similarity)

    def analyze(self, request: RequestEnvelope) -> ResponseEnvelope:
        """Comprehensive image/screenshot/doc analysis."""

        trace_id = request.trace.trace_id
        start = time.perf_counter()

        image_path = request.input.image_path
        doc_path = request.input.document_path
        model_name = request.model_overrides.get("vision", self._config.default_vision_model)
        vision = self._registry.create_vision(model_name)

        payload: dict[str, Any] = {}

        if image_path:
            payload["caption"] = vision.caption(image_path=image_path, style="detailed")
            payload["technical_description"] = vision.caption(
                image_path=image_path, style="technical"
            )
            detector = self._registry.create_detector("yolo")
            detections = detector.detect(image_path)
            payload["detections"] = detections

            asset_id = str(uuid4())
            row_id = self._sqlite.add_asset(asset_id=asset_id, path=image_path, media_type="image")
            self._sqlite.add_detections(row_id, detections)

        if doc_path:
            payload["document"] = self._docs.run(doc_path).model_dump()
            rag_answer = self._rag.answer(
                question=request.input.question or "Summarize key insights", path=doc_path
            )
            payload["document_summary"] = rag_answer

        latency = self._record("analyze", start, trace_id, model_name=model_name)
        return self._response(trace_id, payload, latency, confidence=0.75)

    def documents(self, request: RequestEnvelope) -> ResponseEnvelope:
        """Ingest document into multimodal RAG index."""

        trace_id = request.trace.trace_id
        start = time.perf_counter()

        doc_path = request.input.document_path or ""
        payload = self._rag.ingest_document(doc_path)

        latency = self._record(
            "documents", start, trace_id, model_name=self._config.default_embedding_model
        )
        return self._response(trace_id, payload, latency, confidence=0.8)

    def analytics(self, request: RequestEnvelope | None = None) -> ResponseEnvelope:
        """Return usage and latency analytics."""

        trace_id = request.trace.trace_id if request else str(uuid4())
        start = time.perf_counter()

        payload = {
            "latency_summary": self._metrics.summary(),
            "model_usage": self._sqlite.list_model_usage(),
            "system": collect_system_stats(),
        }
        latency = self._record("analytics", start, trace_id)
        return self._response(trace_id, payload, latency, confidence=1.0)

    def index_asset(self, path: str, modality: str = "image") -> dict[str, Any]:
        """Index asset in vector store for search."""

        if modality == "image":
            record_id = self._retrieval.index_image(path, metadata={"path": path})
        else:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            record_id = self._retrieval.index_text(
                text=text, modality=modality, metadata={"path": path}
            )
        return {"record_id": record_id, "modality": modality}

    def call_external_mcp_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Invoke external MCP-compatible tool hook."""

        return self._external_mcp_hooks.call(tool_name, payload)
