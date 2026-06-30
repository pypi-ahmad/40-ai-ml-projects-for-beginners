"""Translation module using local translation-capable Ollama models."""

from __future__ import annotations

import logging
from typing import Any

from src.config import get_config
from src.ollama_client import OllamaClient
from src.schemas import ErrorInfo, TranslationResult

logger = logging.getLogger(__name__)


SUPPORTED_LANGUAGES = [
    "English",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Russian",
    "Chinese",
    "Japanese",
    "Korean",
    "Arabic",
    "Hindi",
    "Dutch",
    "Polish",
    "Turkish",
    "Vietnamese",
    "Thai",
    "Indonesian",
]


class Translator:
    """Translate text between languages with explicit source/target controls."""

    def __init__(
        self,
        model: str | None = None,
        client: OllamaClient | None = None,
        max_input_chars: int = 6000,
    ) -> None:
        self.model = model or get_config().translation_model
        self.client = client or OllamaClient()
        self.max_input_chars = max_input_chars
        self._owns_client = client is None

    def close(self) -> None:
        """Close owned client resources."""

        if self._owns_client:
            self.client.close()

    @staticmethod
    def supported_languages() -> list[str]:
        """Return curated language list shown in UI."""

        return SUPPORTED_LANGUAGES

    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Run translation pipeline for one input text."""

        normalized = (text or "").strip()
        if not normalized:
            return TranslationResult(
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                model=self.model,
                error=ErrorInfo(message="Input text cannot be empty.", stage="validation"),
            )

        if source_lang not in SUPPORTED_LANGUAGES:
            return TranslationResult(
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                model=self.model,
                error=ErrorInfo(
                    message=f"Unsupported source language: {source_lang}.",
                    stage="validation",
                ),
            )

        if target_lang not in SUPPORTED_LANGUAGES:
            return TranslationResult(
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                model=self.model,
                error=ErrorInfo(
                    message=f"Unsupported target language: {target_lang}.",
                    stage="validation",
                ),
            )

        if source_lang == target_lang:
            return TranslationResult(
                translated_text=normalized,
                source_lang=source_lang,
                target_lang=target_lang,
                model=self.model,
                error=None,
            )

        if len(normalized) > self.max_input_chars:
            return TranslationResult(
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                model=self.model,
                error=ErrorInfo(
                    message=(
                        f"Input too long ({len(normalized)} chars). "
                        f"Limit is {self.max_input_chars} chars."
                    ),
                    stage="validation",
                ),
            )

        prompt = (
            f"Translate text from {source_lang} to {target_lang}.\n"
            "Return only translated text with no explanation.\n\n"
            f"Text:\n{normalized}"
        )

        result = self.client.generate(
            model=self.model,
            prompt=prompt,
            temperature=0.1,
            max_tokens=900,
        )

        if result["error"]:
            return TranslationResult(
                translated_text="",
                source_lang=source_lang,
                target_lang=target_lang,
                model=self.model,
                latency_ms=result["latency_ms"],
                error=ErrorInfo(message=result["error"], stage="inference"),
            )

        translated = result["response"].strip().strip('`"')
        return TranslationResult(
            translated_text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
            model=self.model,
            latency_ms=result["latency_ms"],
            error=None,
        )


def translate(
    text: str,
    source_lang: str = "English",
    target_lang: str = "Spanish",
    model: str | None = None,
    client: OllamaClient | None = None,
) -> dict[str, Any]:
    """Backwards-compatible wrapper returning dictionary payload."""

    translator = Translator(model=model, client=client)
    try:
        return translator.translate(text, source_lang, target_lang).model_dump()
    finally:
        translator.close()
