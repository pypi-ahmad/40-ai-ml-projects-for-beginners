"""Multimodal RAG pipeline."""

from __future__ import annotations

from typing import Any

from multimodal_ai.adapters.registry import AdapterRegistry
from multimodal_ai.pipelines.document_pipeline import DocumentPipeline
from multimodal_ai.pipelines.retrieval_pipeline import RetrievalPipeline


class MultimodalRAGPipeline:
    """Upload -> OCR -> chunk -> embedding -> retrieve -> answer."""

    def __init__(
        self,
        registry: AdapterRegistry,
        document_pipeline: DocumentPipeline,
        retrieval_pipeline: RetrievalPipeline,
        llm_name: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._registry = registry
        self._document_pipeline = document_pipeline
        self._retrieval_pipeline = retrieval_pipeline
        self._llm_name = llm_name
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def _chunk(self, text: str) -> list[str]:
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self._chunk_size)
            chunks.append(text[start:end])
            if end == len(text):
                break
            start = max(0, end - self._chunk_overlap)
        return chunks

    def ingest_document(self, path: str) -> dict[str, Any]:
        """Ingest document into retrieval index."""

        ocr_result = self._document_pipeline.run(path)
        chunks = self._chunk(ocr_result.text)
        chunk_ids = [
            self._retrieval_pipeline.index_text(
                chunk, modality="document", metadata={"source": path}
            )
            for chunk in chunks
        ]
        return {
            "ocr": ocr_result.model_dump(),
            "chunks_indexed": len(chunk_ids),
            "chunk_ids": chunk_ids,
        }

    def answer(self, question: str, path: str | None = None, top_k: int = 5) -> dict[str, Any]:
        """Answer question using retrieved chunks + LLM."""

        if path:
            self.ingest_document(path)

        hits = self._retrieval_pipeline.search(question, modality="document", top_k=top_k)
        context = "\n\n".join(hit.text or "" for hit in hits)

        llm = self._registry.create_llm(self._llm_name)
        prompt = (
            "You are multimodal RAG assistant. Use context only. "
            "If answer absent, say not found.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        completion = llm.complete(prompt)
        return {
            "answer": completion.get("text", ""),
            "confidence": 0.7 if hits else 0.2,
            "retrieval_hits": [hit.model_dump() for hit in hits],
        }
