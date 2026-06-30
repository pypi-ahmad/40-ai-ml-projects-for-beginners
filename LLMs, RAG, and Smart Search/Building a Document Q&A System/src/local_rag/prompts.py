"""Prompt templates and grounding policy helpers."""

from __future__ import annotations

from typing import Literal

from local_rag.types import RetrievalResult

PromptTemplate = Literal[
    "strict_grounding",
    "citation_focus",
    "enterprise_qa",
    "legal_qa",
    "technical_qa",
    "unknown_safe",
]

DEFAULT_UNAVAILABLE_RESPONSE = "Information unavailable in provided context."
DEFAULT_CITATION_INSTRUCTION = "Cite each factual claim using [source_path#chunk_id]."

TEMPLATE_OVERRIDES: dict[PromptTemplate, str] = {
    "strict_grounding": "Never use knowledge outside context. Decline when evidence missing.",
    "citation_focus": "Every sentence must include at least one citation marker.",
    "enterprise_qa": "Answer in enterprise analyst style with concise bullet points when useful.",
    "legal_qa": "Use conservative legal tone and mention uncertainty explicitly.",
    "technical_qa": "Prefer precise technical language and include implementation caveats.",
    "unknown_safe": "When evidence weak, state unavailable response without guessing.",
}


def build_system_prompt(
    *,
    unavailable_response: str = DEFAULT_UNAVAILABLE_RESPONSE,
    strict_grounding: bool = True,
    citation_instruction: str = DEFAULT_CITATION_INSTRUCTION,
    template: PromptTemplate = "enterprise_qa",
) -> str:
    """Create system prompt from configurable grounding policy."""

    grounding_rule = "Answer only using provided context chunks."
    if not strict_grounding:
        grounding_rule = "Prefer context first, clearly mark uncertain statements."

    return (
        "You are enterprise document assistant specialized in grounded QA.\n"
        "Rules:\n"
        f"1) {grounding_rule}\n"
        f"2) If context lacks answer, reply exactly: {unavailable_response}\n"
        "3) Keep answer concise, factual, and beginner-friendly when possible.\n"
        f"4) {citation_instruction}\n"
        f"5) Template policy: {TEMPLATE_OVERRIDES[template]}\n"
    )


SYSTEM_PROMPT = build_system_prompt()


def build_context_block(results: list[RetrievalResult]) -> str:
    """Serialize retrieval results into prompt context block."""

    blocks: list[str] = []
    for rank, item in enumerate(results, start=1):
        source_path = item.metadata.get("source_path", "unknown")
        chunk_id = item.chunk_id[:12]
        document_name = item.metadata.get("document_name", "unknown")
        page_number = item.metadata.get("page_number")
        score = f"{item.score:.4f}"
        blocks.append(
            "\n".join(
                [
                    f"[Chunk {rank}]",
                    f"document_name: {document_name}",
                    f"source_path: {source_path}",
                    f"page_number: {page_number}",
                    f"chunk_id: {chunk_id}",
                    f"retrieval_score: {score}",
                    f"text: {item.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_messages(
    query: str,
    results: list[RetrievalResult],
    *,
    unavailable_response: str = DEFAULT_UNAVAILABLE_RESPONSE,
    strict_grounding: bool = True,
    citation_instruction: str = DEFAULT_CITATION_INSTRUCTION,
    template: PromptTemplate = "enterprise_qa",
    conversation_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for generator model."""

    context = build_context_block(results)
    system_prompt = build_system_prompt(
        unavailable_response=unavailable_response,
        strict_grounding=strict_grounding,
        citation_instruction=citation_instruction,
        template=template,
    )

    user_content = (
        "Question:\n"
        f"{query}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Respond with grounded answer and citations. "
        f"If answer missing, output exact sentence: {unavailable_response}"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_content})
    return messages
