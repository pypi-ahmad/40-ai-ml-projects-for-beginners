"""Inference engine supporting local transformer generation and fallback."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass(slots=True)
class GenerationOutput:
    text: str
    latency_ms: float
    tokens_generated: int


class InferenceEngine:
    """Shared inference backend used by CLI, API, and Streamlit."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or "mock-model"
        self._pipeline = None

    def load(self) -> None:
        if self.model_id == "mock-model":
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline("text-generation", model=self.model_id)
        except Exception:
            self._pipeline = None

    def generate(self, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> GenerationOutput:
        start = perf_counter()
        if self._pipeline is not None:
            output = self._pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
            text = str(output[0].get("generated_text", ""))
        else:
            text = f"[mock:{self.model_id}] {prompt[:120]}"

        latency = (perf_counter() - start) * 1000
        token_count = max(len(text.split()), 1)
        return GenerationOutput(text=text, latency_ms=latency, tokens_generated=token_count)
