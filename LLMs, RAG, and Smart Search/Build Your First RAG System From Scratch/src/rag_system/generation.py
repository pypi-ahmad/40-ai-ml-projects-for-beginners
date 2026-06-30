"""Generation engine for local Ollama-backed RAG."""

from __future__ import annotations

import logging
import time
from typing import Any

import ollama

from rag_system.prompts import PromptLibrary
from rag_system.retrieval import RetrievalEngine
from rag_system.types import RAGResponse, RetrievedChunk

logger = logging.getLogger(__name__)


class GenerationEngine:
    """Text generation wrapper around Ollama chat API."""

    def __init__(
        self,
        model_name: str = "qwen3.5:4b",
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.2,
        max_tokens: int = 320,
        request_timeout_s: float = 180.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = ollama.Client(host=host, timeout=request_timeout_s)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | None = False,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate model response and normalized metadata.

        `think=False` is important for qwen3.5 models in Ollama because
        default thinking mode can consume token budget before final answer.
        """
        opts = {
            "temperature": self.temperature if temperature is None else temperature,
            "num_predict": self.max_tokens if max_tokens is None else max_tokens,
        }

        start = time.perf_counter()
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options=opts,
            think=think,
            format=response_format,
        )
        elapsed = time.perf_counter() - start

        text = (response.message.content or "").strip()
        if not text and response.message.thinking:
            logger.warning(
                "Model returned empty content but non-empty thinking trace; "
                "consider larger num_predict or stricter prompt."
            )

        return {
            "text": text,
            "thinking": response.message.thinking or "",
            "latency_s": elapsed,
            "done_reason": getattr(response, "done_reason", "unknown"),
            "prompt_tokens": int(getattr(response, "prompt_eval_count", 0) or 0),
            "completion_tokens": int(getattr(response, "eval_count", 0) or 0),
        }

    def generate_rag_answer(self, query: str, context: str) -> dict[str, Any]:
        """Generate grounded answer from retrieved context."""
        messages = PromptLibrary.rag_answer(query=query, context=context)
        return self.generate(messages=messages, think=False)

    def generate_baseline_answer(self, query: str) -> dict[str, Any]:
        """Generate no-context baseline answer for hallucination comparison."""
        messages = PromptLibrary.plain_answer(query=query)
        return self.generate(messages=messages, think=False)


class RAGPipeline:
    """End-to-end retrieve-augment-generate pipeline."""

    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        generation_engine: GenerationEngine,
        min_relevance_score: float = 0.40,
        abstain_threshold: float = 0.25,
    ) -> None:
        self.retrieval_engine = retrieval_engine
        self.generation_engine = generation_engine
        self.min_relevance_score = min_relevance_score
        self.abstain_threshold = abstain_threshold

    def answer(
        self,
        query: str,
        top_k: int = 6,
        metadata_filter: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """Run full RAG flow and return structured response payload."""
        t0 = time.perf_counter()
        retrieved: list[RetrievedChunk] = self.retrieval_engine.query(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        retrieval_latency = time.perf_counter() - t0

        context, citations = self.retrieval_engine.format_context(retrieved)
        top_score = retrieved[0].score if retrieved else 0.0
        should_abstain = (not retrieved) or (top_score < self.abstain_threshold)

        if should_abstain:
            abstain_reason = "no_retrieval" if not retrieved else f"low_relevance_top_score={top_score:.3f}"
            return RAGResponse(
                query=query,
                answer="I cannot find enough evidence in retrieved documents.",
                retrieved_chunks=retrieved,
                context=context,
                retrieval_latency_s=retrieval_latency,
                generation_latency_s=0.0,
                total_latency_s=retrieval_latency,
                citations=citations,
                abstained=True,
                abstain_reason=abstain_reason,
            )

        t1 = time.perf_counter()
        generation = self.generation_engine.generate_rag_answer(query=query, context=context)
        generation_latency = time.perf_counter() - t1

        return RAGResponse(
            query=query,
            answer=generation["text"],
            retrieved_chunks=retrieved,
            context=context,
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=generation_latency,
            total_latency_s=retrieval_latency + generation_latency,
            citations=citations,
            abstained=False,
            abstain_reason="",
        )

    def compare_with_no_rag(self, query: str, top_k: int = 6) -> dict[str, Any]:
        """Run side-by-side RAG vs no-RAG generation."""
        rag = self.answer(query=query, top_k=top_k)
        baseline = self.generation_engine.generate_baseline_answer(query)

        return {
            "query": query,
            "rag_answer": rag.answer,
            "rag_context": rag.context,
            "rag_citations": rag.citations,
            "rag_latency_s": rag.total_latency_s,
            "rag_abstained": rag.abstained,
            "rag_abstain_reason": rag.abstain_reason,
            "baseline_answer": baseline["text"],
            "baseline_latency_s": baseline["latency_s"],
        }
