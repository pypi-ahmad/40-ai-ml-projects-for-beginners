"""Unit tests for chat state management and history trimming."""

from __future__ import annotations

from src.chat import ChatEngine


class StubChatClient:
    """Stub chat client with deterministic assistant output."""

    def chat(self, **_: object) -> dict[str, object]:
        return {
            "response": "Acknowledged",
            "latency_ms": 35.0,
            "eval_count": 50,
            "eval_duration_ns": 200,
            "error": None,
        }

    def close(self) -> None:
        pass


def test_chat_rejects_empty_messages() -> None:
    engine = ChatEngine(client=StubChatClient())
    result = engine.send("   ")
    assert result.error is not None
    assert result.error.stage == "validation"


def test_chat_trim_keeps_recent_turns_only() -> None:
    engine = ChatEngine(client=StubChatClient(), max_turns=2)

    engine.load_history(
        [
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "content": "a2"},
            {"role": "user", "content": "u3"},
            {"role": "assistant", "content": "a3"},
        ]
    )

    assert len(engine.history()) == 4
    assert engine.history()[0]["content"] == "u2"


def test_chat_send_appends_user_and_assistant_turns() -> None:
    engine = ChatEngine(client=StubChatClient())
    result = engine.send("Hello")

    assert result.error is None
    assert result.response == "Acknowledged"
    assert len(engine.history()) == 2
    assert engine.history()[0]["role"] == "user"
    assert engine.history()[1]["role"] == "assistant"
