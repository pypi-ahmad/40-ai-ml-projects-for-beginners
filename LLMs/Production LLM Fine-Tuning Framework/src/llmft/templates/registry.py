"""Configurable prompt template registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

TemplateFn = Callable[[str, str, str], str]


@dataclass(slots=True)
class PromptTemplate:
    """Prompt template metadata."""

    name: str
    render: TemplateFn


class TemplateRegistry:
    """Registry for multiple instruction/chat prompt templates."""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register("alpaca", lambda i, inp, out: f"### Instruction\n{i}\n\n### Input\n{inp}\n\n### Response\n{out}")
        self.register("chatml", lambda i, inp, out: f"<|user|>\n{i}\n{inp}\n<|assistant|>\n{out}")
        self.register("llama3", lambda i, inp, out: f"<|start_header_id|>user<|end_header_id|>\n{i}\n{inp}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{out}<|eot_id|>")
        self.register("mistral", lambda i, inp, out: f"[INST] {i}\n{inp} [/INST] {out}")
        self.register("qwen", lambda i, inp, out: f"<|im_start|>user\n{i}\n{inp}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>")
        self.register("phi", lambda i, inp, out: f"<|user|>{i}\n{inp}<|assistant|>{out}")
        self.register("gemma", lambda i, inp, out: f"<start_of_turn>user\n{i}\n{inp}<end_of_turn>\n<start_of_turn>model\n{out}<end_of_turn>")
        self.register("custom", lambda i, inp, out: f"Instruction: {i}\nContext: {inp}\nAnswer: {out}")

    def register(self, name: str, renderer: TemplateFn) -> None:
        """Register or replace a prompt template."""
        self._templates[name.lower()] = PromptTemplate(name=name.lower(), render=renderer)

    def render(self, name: str, instruction: str, input_text: str, output_text: str) -> str:
        """Render template by name."""
        key = name.lower()
        if key not in self._templates:
            raise KeyError(f"Unknown template: {name}")
        return self._templates[key].render(instruction, input_text, output_text)

    def names(self) -> list[str]:
        """Return available template names."""
        return sorted(self._templates.keys())
