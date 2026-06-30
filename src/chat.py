from typing import cast

from src.ollama_client import OllamaClient

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer concisely and accurately."


class ChatEngine:
    def __init__(self, model: str = "qwen3.5:4b", max_turns: int = 20) -> None:
        self.model = model
        self.max_turns = max_turns
        self._client = OllamaClient()
        self.history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def send(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        result = self._client.chat(self.model, self.history)
        reply = result.get("message", {}).get("content", "")
        self.history.append({"role": "assistant", "content": reply})
        self._trim()
        return cast(str, reply)

    def _trim(self) -> None:
        max_msgs = self.max_turns * 2 + 1
        if len(self.history) > max_msgs:
            self.history = [self.history[0]] + self.history[-(max_msgs - 1) :]

    def reset(self) -> None:
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    def close(self) -> None:
        self._client.close()
