from unittest.mock import MagicMock, patch

import pytest

from src.chat import SYSTEM_PROMPT, ChatEngine


@pytest.fixture
def engine() -> ChatEngine:
    e = ChatEngine()
    yield e
    e.close()


def test_default_system_prompt(engine: ChatEngine) -> None:
    assert engine.history[0]["role"] == "system"
    assert engine.history[0]["content"] == SYSTEM_PROMPT


def test_reset_clears_history(engine: ChatEngine) -> None:
    engine.history.append({"role": "user", "content": "hi"})
    assert len(engine.history) == 2
    engine.reset()
    assert len(engine.history) == 1
    assert engine.history[0]["role"] == "system"


def test_reset_preserves_system_prompt(engine: ChatEngine) -> None:
    engine.reset()
    assert engine.history[0]["content"] == SYSTEM_PROMPT


@patch("src.chat.OllamaClient")
def test_send_appends_messages(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.chat.return_value = {"message": {"content": "Hello back!"}}
    e = ChatEngine()
    e.send("Hi!")
    assert len(e.history) == 3
    assert e.history[1]["role"] == "user"
    assert e.history[1]["content"] == "Hi!"
    assert e.history[2]["role"] == "assistant"
    assert e.history[2]["content"] == "Hello back!"
    e.close()


@patch("src.chat.OllamaClient")
def test_send_returns_reply(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.chat.return_value = {"message": {"content": "Sure, I can help!"}}
    e = ChatEngine()
    reply = e.send("Help me")
    assert reply == "Sure, I can help!"
    e.close()


@patch("src.chat.OllamaClient")
def test_send_handles_empty_reply(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.chat.return_value = {"message": {}}
    e = ChatEngine()
    reply = e.send("Hello")
    assert reply == ""
    e.close()


@patch("src.chat.OllamaClient")
def test_multiple_messages_accumulate(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.chat.return_value = {"message": {"content": "reply"}}
    e = ChatEngine()
    e.send("First")
    e.send("Second")
    assert len(e.history) == 5
    assert e.history[1]["content"] == "First"
    assert e.history[3]["content"] == "Second"
    e.close()


@patch("src.chat.OllamaClient")
def test_send_calls_chat_api(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.chat.return_value = {"message": {"content": "ok"}}
    e = ChatEngine()
    e.send("Ping")
    mock_ollama.return_value.chat.assert_called_once()
    args = mock_ollama.return_value.chat.call_args[0]
    assert args[0] == e.model
    history_at_call = args[1]
    assert history_at_call[0]["role"] == "system"
    assert history_at_call[1]["role"] == "user"
    assert history_at_call[1]["content"] == "Ping"
    e.close()
