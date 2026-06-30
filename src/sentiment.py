import json
from typing import cast

from src.ollama_client import OllamaClient

SYSTEM_PROMPT = """You are a sentiment classifier. Return ONLY valid JSON with keys:
- "label": one of "positive", "negative", "neutral"
- "score": float 0.0-1.0
- "explanation": short string

No markdown, no extra text."""


class SentimentAnalyzer:
    def __init__(self, model: str = "qwen3.5:2b") -> None:
        self.model = model
        self._client = OllamaClient()

    def analyze(self, text: str) -> dict:
        result = self._client.generate(self.model, text, system=SYSTEM_PROMPT)
        raw = result.get("response", "")
        return self._parse(raw)

    def analyze_batch(self, texts: list[str]) -> list[dict]:
        return [self.analyze(t) for t in texts]

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _parse(raw: str) -> dict:
        try:
            cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
            return cast(dict, json.loads(cleaned))
        except json.JSONDecodeError:
            return {"label": "neutral", "score": 0.0, "explanation": "parse failed"}
