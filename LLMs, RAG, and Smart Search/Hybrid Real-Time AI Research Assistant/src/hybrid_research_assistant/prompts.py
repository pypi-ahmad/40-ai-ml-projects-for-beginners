"""Prompt library for generation styles and strict grounding."""

from __future__ import annotations

from hybrid_research_assistant.schemas import RetrievedContext

PROMPT_LIBRARY: dict[str, str] = {
    "strict_qa": (
        "You are a strict QA assistant. Answer ONLY using provided context. "
        "If context is insufficient, output exactly: {fallback}. "
        "Never invent facts or citations."
    ),
    "research_assistant": (
        "You are an AI research assistant. Synthesize evidence from context with concise analysis. "
        "If evidence is missing, output exactly: {fallback}."
    ),
    "teacher": (
        "You are a teacher. Explain clearly and step-by-step using only the provided context. "
        "If context is missing details, output exactly: {fallback}."
    ),
    "technical_mentor": (
        "You are a technical mentor. Give actionable implementation advice grounded strictly in context. "
        "If context is insufficient, output exactly: {fallback}."
    ),
    "summarizer": (
        "You are a summarizer. Produce a concise synthesis using only provided context. "
        "If context is insufficient, output exactly: {fallback}."
    ),
}


def build_context_block(rows: list[RetrievedContext]) -> str:
    """Serialize retrieval rows for LLM context."""

    blocks: list[str] = []
    for rank, row in enumerate(rows, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Context {rank}]",
                    f"source: {row.metadata.get('source', 'unknown')}",
                    f"page: {row.metadata.get('page_number')}",
                    f"url: {row.metadata.get('url', '')}",
                    f"chunk_id: {row.chunk_id}",
                    f"score: {row.score:.4f}",
                    f"text: {row.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_messages(
    *,
    query: str,
    rows: list[RetrievedContext],
    prompt_name: str,
    fallback: str,
) -> list[dict[str, str]]:
    """Build chat messages for the generation model."""

    prompt_template = PROMPT_LIBRARY.get(prompt_name, PROMPT_LIBRARY["research_assistant"])
    system_prompt = (
        prompt_template.format(fallback=fallback)
        + "\nAlways provide citations in [source|chunk_id] format after each claim."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Context:\n{build_context_block(rows)}\n\n"
        f"Instructions:\n"
        f"1) Use only context evidence.\n"
        f"2) If not enough evidence, reply exactly: {fallback}\n"
        f"3) Include citation markers [source|chunk_id]."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
