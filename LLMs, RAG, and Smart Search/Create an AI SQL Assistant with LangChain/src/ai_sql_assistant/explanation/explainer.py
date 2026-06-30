"""SQL explanation and business interpretation generation."""

from __future__ import annotations

import json
from typing import Any

from ai_sql_assistant.config import AppSettings
from ai_sql_assistant.llm.ollama_client import OllamaDeterministicClient
from ai_sql_assistant.prompts.templates import explanation_prompt


class SQLExplainer:
    """Generate step-by-step and business explanations for SQL."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.client = OllamaDeterministicClient(settings)

    def close(self) -> None:
        """Close client resources."""
        self.client.close()

    def explain(
        self,
        question: str,
        sql: str,
        explain_plan: list[dict[str, Any]],
        model: str,
    ) -> tuple[str, str]:
        """Return explanation and business interpretation text."""
        plan_text = json.dumps(explain_plan, indent=2)
        prompt = explanation_prompt(question=question, sql=sql, execution_plan=plan_text)
        try:
            response = self.client.generate(prompt=prompt, model=model)
            content = response.text
            business = self._extract_business_interpretation(content)
            return content, business
        except Exception:
            fallback = self._heuristic_explanation(question, sql, explain_plan)
            business = self._extract_business_interpretation(fallback)
            return fallback, business

    @staticmethod
    def _extract_business_interpretation(text: str) -> str:
        markers = ["Business interpretation", "Business Interpretation", "business interpretation"]
        for marker in markers:
            idx = text.find(marker)
            if idx >= 0:
                return text[idx:][:700]
        return text[:500]

    @staticmethod
    def _heuristic_explanation(question: str, sql: str, plan: list[dict[str, Any]]) -> str:
        return (
            "Step-by-step SQL logic:\n"
            f"1) Query answers: {question}.\n"
            "2) It reads tables and applies filters/aggregations in SELECT.\n"
            "3) It may use joins and grouping depending on clauses.\n\n"
            "Business interpretation:\n"
            "Results provide requested business KPI or trend over selected dimensions.\n\n"
            "Execution plan overview:\n"
            f"SQLite plan entries: {len(plan)}.\n"
            f"SQL used:\n{sql}"
        )
