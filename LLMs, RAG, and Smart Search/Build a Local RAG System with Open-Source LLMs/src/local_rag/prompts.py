"""Prompt templates and grounding policy helpers."""

from __future__ import annotations

from local_rag.types import RetrievalResult

DEFAULT_UNAVAILABLE_RESPONSE = "Information unavailable in provided context."
DEFAULT_CITATION_INSTRUCTION = (
    "Include citation markers like [source_path#chunk_id] for each claim."
)


def build_system_prompt(
    *,
    unavailable_response: str = DEFAULT_UNAVAILABLE_RESPONSE,
    strict_grounding: bool = True,
    citation_instruction: str = DEFAULT_CITATION_INSTRUCTION,
) -> str:
    """Create system prompt from configurable grounding policy."""

    grounding_rule = "Answer only using provided context chunks."
    if not strict_grounding:
        grounding_rule = (
            "Prefer provided context first. If needed, mark uncertain parts clearly."
        )

    return (
        "You are a local enterprise RAG assistant.\n"
        "Rules:\n"
        f"1) {grounding_rule}\n"
        f"2) If context lacks answer, reply exactly: {unavailable_response}\n"
        "3) Keep answer concise and factual.\n"
        f"4) {citation_instruction}\n"
    )


SYSTEM_PROMPT = build_system_prompt()


def build_context_block(results: list[RetrievalResult]) -> str:
    """Serialize retrieval results into prompt context block."""

    blocks: list[str] = []
    for rank, item in enumerate(results, start=1):
        source_path = item.metadata.get("source_path", "unknown")
        chunk_id = item.chunk_id
        score = f"{item.score:.4f}"
        blocks.append(
            "\n".join(
                [
                    f"[Chunk {rank}]",
                    f"source_path: {source_path}",
                    f"chunk_id: {chunk_id}",
                    f"similarity_score: {score}",
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
) -> list[dict[str, str]]:
    """Build chat messages for generator model."""

    context = build_context_block(results)
    system_prompt = build_system_prompt(
        unavailable_response=unavailable_response,
        strict_grounding=strict_grounding,
        citation_instruction=citation_instruction,
    )
    user_content = (
        "Question:\n"
        f"{query}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Answer with citations. "
        f"If answer not in context, output exact sentence: {unavailable_response}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
