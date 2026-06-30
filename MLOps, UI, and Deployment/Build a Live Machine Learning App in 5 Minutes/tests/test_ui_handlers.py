"""Tests for app-level wrappers backed by shared AppHandlers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import app
from src.ui_handlers import AppHandlers


class DummySentiment:
    """Simple deterministic sentiment service."""

    def analyze(self, _: str):
        return SimpleNamespace(
            label="Positive",
            confidence=0.88,
            explanation="Looks positive",
            error=None,
            model_dump=lambda: {"label": "Positive", "confidence": 0.88},
        )


class DummyTranslator:
    """Simple deterministic translation service."""

    def translate(self, text: str, source_lang: str, target_lang: str):
        return SimpleNamespace(
            translated_text=f"{text} ({source_lang}->{target_lang})",
            source_lang=source_lang,
            target_lang=target_lang,
            error=None,
            model_dump=lambda: {"translated_text": text},
        )


class DummyChat:
    """In-memory chat service with deterministic response."""

    def __init__(self):
        self._history: list[dict[str, str]] = []

    def reset(self):
        self._history = []

    def load_history(self, history):
        self._history = list(history)

    def send(self, message: str, temperature: float, max_tokens: int):
        _ = temperature, max_tokens
        self._history.extend(
            [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Stub response"},
            ]
        )
        return SimpleNamespace(
            response="Stub response",
            error=None,
            model_dump=lambda: {"response": "Stub response"},
        )

    def history(self):
        return self._history


class DummyRegistry:
    """Minimal registry implementation for AppHandlers tests."""

    def __init__(self):
        self._chat_engines: dict[str, DummyChat] = {}

    def get_sentiment(self, model: str):
        _ = model
        return DummySentiment()

    def get_translator(self, model: str):
        _ = model
        return DummyTranslator()

    def get_chat(self, model: str):
        if model not in self._chat_engines:
            self._chat_engines[model] = DummyChat()
        return self._chat_engines[model]


@pytest.fixture
def patched_handlers(monkeypatch):
    """Patch app to use deterministic handlers without Ollama dependency."""

    handlers = AppHandlers(registry=DummyRegistry())
    monkeypatch.setattr(handlers, "_ensure_model_available", lambda _model: None)
    monkeypatch.setattr(app, "HANDLERS", handlers)
    return handlers


def test_handle_sentiment_success(patched_handlers):
    _ = patched_handlers
    label, conf, explanation, payload = app.handle_sentiment("Great app", "qwen3.5:2b")

    assert label == "Positive"
    assert conf == 0.88
    assert "positive" in explanation.lower()
    assert "confidence" in payload


def test_handle_translation_success(patched_handlers):
    _ = patched_handlers
    translated, payload = app.handle_translation("Hello", "English", "Spanish", "translategemma:4b")

    assert "English->Spanish" in translated
    assert "translated_text" in payload


def test_handle_chat_state_per_model_and_reset(patched_handlers):
    _ = patched_handlers
    _, display_1, state, payload = app.handle_chat("Hi", {}, "qwen3.5:4b", 0.3, 400)

    assert len(display_1) == 2
    assert state["qwen3.5:4b"][-1]["role"] == "assistant"
    assert "Stub response" in payload

    _, _, state, _ = app.handle_chat("Hola", state, "granite4.1:3b", 0.3, 400)

    assert len(state["qwen3.5:4b"]) == 2
    assert len(state["granite4.1:3b"]) == 2

    selected_display, _ = app.switch_chat_model(state, "qwen3.5:4b")
    assert selected_display == state["qwen3.5:4b"]

    reset_display, reset_state, reset_payload = app.reset_chat("qwen3.5:4b", state)
    assert reset_display == []
    assert reset_state["qwen3.5:4b"] == []
    assert len(reset_state["granite4.1:3b"]) == 2
    assert reset_payload == "{}"
