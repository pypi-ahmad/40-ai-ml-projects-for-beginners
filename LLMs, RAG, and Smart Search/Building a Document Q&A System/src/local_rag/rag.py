"""End-to-end RAG orchestration."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import asdict

from local_rag.generator import OllamaGenerator
from local_rag.prompts import PromptTemplate, build_messages
from local_rag.retriever import RetrievalStrategy, Retriever
from local_rag.types import CitationRecord, RAGResponse, StreamSession, TimingBreakdown


class RAGPipeline:
    """Complete RAG query flow: embed -> retrieve -> prompt -> generate."""

    def __init__(
        self,
        retriever: Retriever,
        generator: OllamaGenerator,
        *,
        strict_grounding: bool = True,
        unavailable_response: str = "Information unavailable in provided context.",
        citation_instruction: str = (
            "Cite every factual claim using [source_path#chunk_id] markers."
        ),
        default_template: PromptTemplate = "enterprise_qa",
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.strict_grounding = strict_grounding
        self.unavailable_response = unavailable_response
        self.citation_instruction = citation_instruction
        self.default_template = default_template

    @staticmethod
    def _build_citations(retrieved) -> list[dict[str, str | int | float | None]]:
        citations: list[dict[str, str | int | float | None]] = []
        seen: set[str] = set()
        for hit in retrieved:
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            citation = CitationRecord(
                document_name=str(hit.metadata.get("document_name", "unknown")),
                source_path=str(hit.metadata.get("source_path", "unknown")),
                page_number=(
                    int(hit.metadata["page_number"])
                    if hit.metadata.get("page_number") is not None
                    else None
                ),
                chunk_id=hit.chunk_id,
                similarity_score=float(hit.score),
                evidence_text=hit.text[:400],
            )
            citations.append(asdict(citation))
        return citations

    def _prepare_session(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
        strategy: RetrievalStrategy,
        prompt_template: PromptTemplate,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> tuple[list[dict[str, str]], StreamSession]:
        started_total = time.perf_counter()

        started_embedding = time.perf_counter()
        retrieved, retrieval_ms = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters,
            strategy=strategy,
        )
        embedding_ms = (time.perf_counter() - started_embedding) * 1000 - retrieval_ms

        started_prompt = time.perf_counter()
        messages = build_messages(
            query=query,
            results=retrieved,
            unavailable_response=self.unavailable_response,
            strict_grounding=self.strict_grounding,
            citation_instruction=self.citation_instruction,
            template=prompt_template,
            conversation_history=conversation_history,
        )
        prompt_ms = (time.perf_counter() - started_prompt) * 1000

        return messages, StreamSession(
            query=query,
            model=self.generator.model,
            top_k=top_k,
            retrieval_strategy=strategy,
            retrieved=retrieved,
            citations=self._build_citations(retrieved),
            embedding_ms=max(embedding_ms, 0.0),
            retrieval_ms=retrieval_ms,
            prompt_ms=prompt_ms,
            started_total=started_total,
        )

    def ask(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
        strategy: RetrievalStrategy = "vector",
        prompt_template: PromptTemplate | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> RAGResponse:
        """Run full RAG pipeline and return rich response object."""

        messages, session = self._prepare_session(
            query,
            top_k=top_k,
            filters=filters,
            strategy=strategy,
            prompt_template=prompt_template or self.default_template,
            conversation_history=conversation_history,
        )
        answer, generation_ms = self.generator.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        total_ms = (time.perf_counter() - session.started_total) * 1000

        return RAGResponse(
            query=session.query,
            answer=answer,
            model=session.model,
            top_k=session.top_k,
            retrieval_strategy=session.retrieval_strategy,
            citations=session.citations,
            retrieved=session.retrieved,
            timings=TimingBreakdown(
                embedding_ms=session.embedding_ms,
                retrieval_ms=session.retrieval_ms,
                prompt_ms=session.prompt_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            ),
        )

    def ask_stream(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str] | None = None,
        strategy: RetrievalStrategy = "vector",
        prompt_template: PromptTemplate | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[Iterator[str], StreamSession]:
        """Prepare retrieval/prompt state and return generation token stream."""

        messages, session = self._prepare_session(
            query,
            top_k=top_k,
            filters=filters,
            strategy=strategy,
            prompt_template=prompt_template or self.default_template,
            conversation_history=conversation_history,
        )
        stream = self.generator.stream_generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return stream, session

    @staticmethod
    def finalize_stream(session: StreamSession, answer: str, generation_ms: float) -> RAGResponse:
        """Build final response object after stream consumption."""

        total_ms = (time.perf_counter() - session.started_total) * 1000
        return RAGResponse(
            query=session.query,
            answer=answer,
            model=session.model,
            top_k=session.top_k,
            retrieval_strategy=session.retrieval_strategy,
            citations=session.citations,
            retrieved=session.retrieved,
            timings=TimingBreakdown(
                embedding_ms=session.embedding_ms,
                retrieval_ms=session.retrieval_ms,
                prompt_ms=session.prompt_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            ),
        )
