"""Backward-compatible prompt template module.

Kept for compatibility with earlier tutorial code while delegating to
PromptLibrary, which now contains the maintained prompt definitions.
"""

from __future__ import annotations

from rag_system.prompts import PromptLibrary


class PromptTemplates:
    """Compatibility wrapper exposing old method names."""

    @staticmethod
    def rag_generation(query: str, context: str) -> list[dict[str, str]]:
        return PromptLibrary.rag_answer(query=query, context=context)

    @staticmethod
    def no_context_generation(query: str) -> list[dict[str, str]]:
        return PromptLibrary.plain_answer(query=query)

    @staticmethod
    def query_expansion(original_query: str, num_queries: int = 3) -> list[dict[str, str]]:
        return PromptLibrary.query_expansion(query=original_query, n=num_queries)

    @staticmethod
    def llm_judge_evaluation(query: str, context: str, response: str) -> list[dict[str, str]]:
        return PromptLibrary.judge_prompt(query=query, context=context, answer=response)
