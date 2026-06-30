"""Model registry and support matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SupportTier(str, Enum):
    DEEP = "deep"
    SMOKE = "smoke"
    OPTIONAL = "optional"


@dataclass(slots=True)
class ModelSpec:
    id: str
    family: str
    instruction_tuned: bool
    tier: SupportTier
    default_template: str


MODEL_SPECS: dict[str, ModelSpec] = {
    "tinyllama_1_1b_chat": ModelSpec(
        id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        family="TinyLlama",
        instruction_tuned=True,
        tier=SupportTier.DEEP,
        default_template="llama",
    ),
    "smollm2_1_7b_instruct": ModelSpec(
        id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        family="SmolLM",
        instruction_tuned=True,
        tier=SupportTier.DEEP,
        default_template="chatml",
    ),
    "qwen3_1_7b_instruct": ModelSpec(
        id="Qwen/Qwen3-1.7B-Instruct",
        family="Qwen",
        instruction_tuned=True,
        tier=SupportTier.DEEP,
        default_template="qwen",
    ),
    "llama3_instruct": ModelSpec(
        id="meta-llama/Llama-3.2-3B-Instruct",
        family="Llama",
        instruction_tuned=True,
        tier=SupportTier.SMOKE,
        default_template="llama",
    ),
    "gemma3_instruct": ModelSpec(
        id="google/gemma-3-4b-it",
        family="Gemma",
        instruction_tuned=True,
        tier=SupportTier.SMOKE,
        default_template="gemma",
    ),
    "phi4_mini_instruct": ModelSpec(
        id="microsoft/Phi-4-mini-instruct",
        family="Phi",
        instruction_tuned=True,
        tier=SupportTier.SMOKE,
        default_template="chatml",
    ),
    "mistral_instruct": ModelSpec(
        id="mistralai/Mistral-7B-Instruct-v0.3",
        family="Mistral",
        instruction_tuned=True,
        tier=SupportTier.SMOKE,
        default_template="chatml",
    ),
    "modernbert_instruct": ModelSpec(
        id="answerdotai/ModernBERT-base",
        family="ModernBERT",
        instruction_tuned=False,
        tier=SupportTier.SMOKE,
        default_template="chatml",
    ),
    "deepseek": ModelSpec(
        id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        family="DeepSeek",
        instruction_tuned=True,
        tier=SupportTier.OPTIONAL,
        default_template="qwen",
    ),
}


def list_models() -> list[ModelSpec]:
    return list(MODEL_SPECS.values())


def get_model_spec(model_key: str) -> ModelSpec:
    if model_key not in MODEL_SPECS:
        raise KeyError(f"Unknown model key: {model_key}")
    return MODEL_SPECS[model_key]
