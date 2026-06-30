import json
from typing import cast

from src.ollama_client import OllamaClient

SYSTEM_PROMPT = """You are a text summarizer. Return ONLY valid JSON with keys:
- "summary": string (2-3 sentences)
- "key_points": list of strings (3-5 items)
- "tldr": one short sentence

No markdown, no extra text."""


class Summarizer:
    def __init__(self, model: str = "granite4.1:3b") -> None:
        self.model = model
        self._client = OllamaClient()

    def summarize(self, text: str) -> dict:
        if len(text.strip()) < 50:
            return {"summary": text, "key_points": [], "tldr": text}
        result = self._client.generate(self.model, text, system=SYSTEM_PROMPT)
        raw = result.get("response", "")
        return self._parse(raw)

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _parse(raw: str) -> dict:
        try:
            cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
            return cast(dict, json.loads(cleaned))
        except json.JSONDecodeError:
            return {"summary": raw[:200], "key_points": [], "tldr": "parse failed"}
