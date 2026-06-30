"""Unit tests for validation and normalization across task modules."""

from __future__ import annotations

from src.summarization import Summarizer
from src.translation import Translator


class StubClient:
    """Simple stub returning fixed content and no errors."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, **_: object) -> dict[str, object]:
        return {
            "response": self.response,
            "latency_ms": 22.5,
            "eval_count": 20,
            "eval_duration_ns": 100,
            "error": None,
        }

    def close(self) -> None:
        pass


def test_summarizer_empty_input_rejected() -> None:
    summarizer = Summarizer(client=StubClient(""))
    result = summarizer.summarize(" ")

    assert result.error is not None
    assert result.error.stage == "validation"


def test_summarizer_parses_json_payload() -> None:
    payload = (
        '{"summary":"Short summary.","key_points":["Point A","Point B"],"original_length":123}'
    )
    summarizer = Summarizer(client=StubClient(payload))
    result = summarizer.summarize("Long article body")

    assert result.error is None
    assert result.summary == "Short summary."
    assert result.key_points == ["Point A", "Point B"]


def test_translation_same_language_passthrough() -> None:
    translator = Translator(client=StubClient("unused"))
    result = translator.translate("Hello world", source_lang="English", target_lang="English")

    assert result.error is None
    assert result.translated_text == "Hello world"
