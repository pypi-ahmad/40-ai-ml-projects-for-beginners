"""LLM-as-a-judge evaluation using local Ollama model."""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ollama import Client

from local_rag.types import JudgeAggregate, JudgeScore
from local_rag.utils import write_jsonl


class LLMJudge:
    """Grade RAG answers with local judge model."""

    def __init__(self, host: str, model: str) -> None:
        self.client = Client(host=host)
        self.model = model

    def evaluate(
        self,
        *,
        query: str,
        answer: str,
        context: str,
        reference_answer: str | None = None,
    ) -> JudgeScore:
        """Return judge rubric scores for one answer."""

        rubric_prompt = (
            "Score answer from 1-5 for correctness, groundedness, completeness, "
            "faithfulness, conciseness, citation_quality. Return strict JSON object with keys: "
            "correctness, groundedness, completeness, faithfulness, conciseness, "
            "citation_quality, rationale. No markdown.\n\n"
            f"Query: {query}\n"
            f"Answer: {answer}\n"
            f"Context: {context}\n"
            f"Reference: {reference_answer or 'N/A'}\n"
        )

        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": rubric_prompt}],
            options={"temperature": 0.0, "num_predict": 256},
            think=False,
        )
        content = response.get("message", {}).get("content", "")
        payload = self._parse_json(content)

        return JudgeScore(
            correctness=int(payload.get("correctness", 1)),
            groundedness=int(payload.get("groundedness", 1)),
            completeness=int(payload.get("completeness", 1)),
            faithfulness=int(payload.get("faithfulness", 1)),
            conciseness=int(payload.get("conciseness", 1)),
            citation_quality=int(payload.get("citation_quality", 1)),
            rationale=str(payload.get("rationale", "")),
        )

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError as err:
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                return json.loads(match.group(0))
            raise ValueError("Judge response is not valid JSON") from err

    def evaluate_batch(
        self,
        rows: list[dict[str, str | None]],
    ) -> list[JudgeScore]:
        """Evaluate multiple rows with same rubric."""

        return [
            self.evaluate(
                query=str(row.get("query", "")),
                answer=str(row.get("answer", "")),
                context=str(row.get("context", "")),
                reference_answer=row.get("reference_answer"),
            )
            for row in rows
        ]

    @staticmethod
    def aggregate(scores: list[JudgeScore]) -> JudgeAggregate:
        """Aggregate judge scores by arithmetic mean."""

        if not scores:
            return JudgeAggregate(
                count=0,
                correctness=0.0,
                groundedness=0.0,
                completeness=0.0,
                faithfulness=0.0,
                conciseness=0.0,
                citation_quality=0.0,
            )

        return JudgeAggregate(
            count=len(scores),
            correctness=statistics.mean(float(row.correctness) for row in scores),
            groundedness=statistics.mean(float(row.groundedness) for row in scores),
            completeness=statistics.mean(float(row.completeness) for row in scores),
            faithfulness=statistics.mean(float(row.faithfulness) for row in scores),
            conciseness=statistics.mean(float(row.conciseness) for row in scores),
            citation_quality=statistics.mean(float(row.citation_quality) for row in scores),
        )


def dump_judge_scores(path: Path, rows: list[JudgeScore]) -> None:
    """Write judge outputs to JSONL."""

    write_jsonl(path, [asdict(row) for row in rows])
