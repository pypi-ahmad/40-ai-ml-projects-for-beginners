"""Conversation and prompt templates."""

from __future__ import annotations

from peft_platform.data.schemas import Sample

_TEMPLATE_NAMES = {"chatml", "alpaca", "llama", "qwen", "gemma"}


def supported_templates() -> list[str]:
    return sorted(_TEMPLATE_NAMES)


def apply_template(sample: Sample, template_name: str, eos_token: str = "</s>") -> str:
    if template_name not in _TEMPLATE_NAMES:
        raise ValueError(f"Unsupported template: {template_name}")

    if sample.messages:
        chat = "\n".join([f"{m['role']}: {m['content']}" for m in sample.messages])
    else:
        chat = ""

    if template_name == "chatml":
        return f"<|system|>You are helpful assistant.\n<|user|>{sample.instruction}\n{sample.input}\n{chat}\n<|assistant|>{sample.output}{eos_token}"

    if template_name == "alpaca":
        return (
            "### Instruction:\n"
            f"{sample.instruction}\n\n"
            "### Input:\n"
            f"{sample.input}\n\n"
            "### Response:\n"
            f"{sample.output}{eos_token}"
        )

    if template_name == "llama":
        return f"<s>[INST] {sample.instruction}\n{sample.input} [/INST] {sample.output}{eos_token}"

    if template_name == "qwen":
        return f"<|im_start|>user\n{sample.instruction}\n{sample.input}<|im_end|>\n<|im_start|>assistant\n{sample.output}{eos_token}<|im_end|>"

    return f"<bos>{sample.instruction}\n{sample.input}\n<eos>\n{sample.output}{eos_token}"
