"""LLM-as-a-judge scoring for SQL generations."""

from __future__ import annotations

import json

from ai_sql_assistant.config import AppSettings
from ai_sql_assistant.llm.ollama_client import OllamaDeterministicClient
from ai_sql_assistant.prompts.templates import judge_prompt
from ai_sql_assistant.types import JudgeScore


class SQLJudge:
    """Judge generated SQL quality with local Ollama model."""

    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.client = OllamaDeterministicClient(settings)

    def close(self) -> None:
        """Close resources."""
        self.client.close()

    def score(self, question: str, generated_sql: str, ground_truth_sql: str) -> JudgeScore:
        """Return structured judge score."""
        prompt = judge_prompt(question=question, generated_sql=generated_sql, ground_truth_sql=ground_truth_sql)
        response = self.client.generate_json(prompt=prompt, model=self.settings.models.judge_model)
        payload = json.loads(response.text)
        return JudgeScore(**payload)
