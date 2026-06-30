"""LLM-as-judge evaluation engine."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import uuid4

from ai_spreadsheet_analytics.llm.base import LLMClient
from ai_spreadsheet_analytics.schemas import EvaluationReport, JudgeScores
from ai_spreadsheet_analytics.state_store import SQLiteStateStore

JUDGE_SYSTEM_PROMPT = """
You are evaluating business insight quality.
Return strict JSON with keys:
insight_quality, correctness, business_relevance, clarity, actionability, rationale.
Each score must be integer 1-5.
""".strip()


class LLMJudge:
    """Evaluate insights with judge model and structured rubric."""

    def __init__(self, llm_client: LLMClient, judge_model: str, state_store: SQLiteStateStore) -> None:
        self.llm_client = llm_client
        self.judge_model = judge_model
        self.state_store = state_store

    def evaluate(
        self,
        model_evaluated: str,
        question: str,
        deterministic_evidence: dict[str, Any],
        model_answer: str,
    ) -> EvaluationReport:
        """Run judge scoring for one answer."""
        prompt = (
            "Question:\n"
            f"{question}\n\n"
            "Deterministic evidence:\n"
            f"{deterministic_evidence}\n\n"
            "Model answer:\n"
            f"{model_answer}\n"
        )
        response = asyncio.run(
            self.llm_client.agenerate(
                model=self.judge_model,
                prompt=prompt,
                system=JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
            )
        )

        parsed = self._parse_response(response.text)
        run_id = f"judge_{uuid4().hex[:10]}"
        report = EvaluationReport(
            run_id=run_id,
            model_evaluated=model_evaluated,
            judge_model=self.judge_model,
            scores=JudgeScores(**parsed),
            metadata={"latency_ms": response.latency_ms, "token_estimate": response.token_estimate},
        )
        self.state_store.add_judge_report(
            run_id=report.run_id,
            model_evaluated=model_evaluated,
            judge_model=self.judge_model,
            report=report.model_dump(),
        )
        return report

    def _parse_response(self, text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback scoring when model returns prose.
            return {
                "insight_quality": 3,
                "correctness": 3,
                "business_relevance": 3,
                "clarity": 3,
                "actionability": 3,
                "rationale": "Judge model returned non-JSON output; fallback score applied.",
            }
