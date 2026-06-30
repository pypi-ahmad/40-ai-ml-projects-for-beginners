"""Prompt templates for retrieval-augmented generation and evaluation."""

from __future__ import annotations

import json


class PromptLibrary:
    """Central prompt library used across generation and judge steps."""

    @staticmethod
    def rag_answer(query: str, context: str) -> list[dict[str, str]]:
        """Grounded RAG prompt with explicit citation contract."""
        system = (
            "You are a retrieval-grounded assistant. "
            "Use only provided context. Do not use outside knowledge. "
            "If context is insufficient, say exactly: "
            "'I cannot find enough evidence in retrieved documents.' "
            "Cite supporting evidence as [1], [2], ... in final answer. "
            "If a claim is unsupported by context, omit it."
        )
        user = (
            "Context:\n"
            f"{context}\n\n"
            "Task:\n"
            f"Question: {query}\n"
            "Answer using concise factual language with citations."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def plain_answer(query: str) -> list[dict[str, str]]:
        """No-context baseline prompt for hallucination comparison."""
        return [
            {
                "role": "system",
                "content": "Answer user question clearly and concisely.",
            },
            {"role": "user", "content": query},
        ]

    @staticmethod
    def query_expansion(query: str, n: int = 4) -> list[dict[str, str]]:
        """Prompt to generate multiple paraphrased retrieval queries."""
        user = (
            f"Generate {n} diverse rewrite variants of this question for retrieval.\n"
            f"Question: {query}\n"
            "Rules: preserve intent, vary vocabulary, no explanations.\n"
            "Output JSON list of strings."
        )
        return [{"role": "user", "content": user}]

    @staticmethod
    def judge_prompt(query: str, context: str, answer: str) -> list[dict[str, str]]:
        """LLM-as-a-judge prompt returning strict JSON."""
        schema = {
            "relevance": "float 1-5",
            "correctness": "float 1-5",
            "groundedness": "float 1-5",
            "completeness": "float 1-5",
            "faithfulness": "float 1-5",
            "rationale": "short text",
        }

        user = (
            "Evaluate the answer quality for a RAG system.\n"
            f"Question: {query}\n"
            f"Retrieved context:\n{context}\n\n"
            f"Answer:\n{answer}\n\n"
            "Scoring rubric (1=poor, 5=excellent):\n"
            "- relevance: answers the question asked\n"
            "- correctness: factually correct given context\n"
            "- groundedness: claims supported by retrieved context\n"
            "- completeness: covers key asked points\n"
            "- faithfulness: no fabricated details beyond context\n"
            f"Return strict JSON matching schema: {json.dumps(schema)}"
        )
        system = "You are a strict evaluator. Only output valid JSON."
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def bad_prompt_example(query: str, context: str) -> str:
        """Intentionally weak prompt used for teaching bad practices."""
        return (
            "Answer anything user asks quickly. "
            "Maybe use context if useful. "
            f"Question: {query}\nContext: {context}"
        )

    @staticmethod
    def good_prompt_example(query: str, context: str) -> str:
        """Well-structured prompt used for teaching prompt best practices."""
        return (
            "Role: retrieval-grounded assistant.\n"
            "Constraints: only use context, cite evidence, admit uncertainty.\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Output: concise factual answer with citation markers [i]."
        )
