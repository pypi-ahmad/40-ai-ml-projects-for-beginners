import json
from typing import cast

from src.ollama_client import OllamaClient

LANGUAGES = {
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Arabic": "ar",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi",
    "Turkish": "tr",
    "Polish": "pl",
    "Swedish": "sv",
    "Greek": "el",
    "Vietnamese": "vi",
    "Thai": "th",
}

SYSTEM_PROMPT = """You are a translator. Translate the user text to the target language.
Return ONLY valid JSON: {"translated_text": "..."}
No markdown, no extra text."""


class Translator:
    def __init__(self, model: str = "translategemma:4b") -> None:
        self.model = model
        self._client = OllamaClient()

    @property
    def supported_languages(self) -> list[str]:
        return sorted(LANGUAGES.keys())

    def translate(self, text: str, target_language: str) -> dict:
        prompt = f"Translate this to {target_language}:\n{text}"
        result = self._client.generate(self.model, prompt, system=SYSTEM_PROMPT)
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
            return {"translated_text": raw.strip()[:200] if raw.strip() else "translation failed"}
