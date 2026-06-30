"""Transformer model registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ModelSpec:
    hf_id: str
    family: str
    supported: bool = True


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "distilbert": ModelSpec(hf_id="distilbert-base-uncased", family="distilbert"),
    "bert_base": ModelSpec(hf_id="bert-base-uncased", family="bert"),
    "roberta_base": ModelSpec(hf_id="roberta-base", family="roberta"),
    "deberta_v3_base": ModelSpec(hf_id="microsoft/deberta-v3-base", family="deberta"),
    "modernbert": ModelSpec(hf_id="answerdotai/ModernBERT-base", family="modernbert"),
    "minilm": ModelSpec(hf_id="microsoft/MiniLM-L12-H384-uncased", family="minilm"),
    "albert": ModelSpec(hf_id="albert-base-v2", family="albert"),
    "electra": ModelSpec(hf_id="google/electra-base-discriminator", family="electra"),
    "tinybert": ModelSpec(hf_id="huawei-noah/TinyBERT_General_4L_312D", family="tinybert"),
}


def resolve_model_id(alias_or_hf_id: str) -> str:
    """Resolve a model alias to Hugging Face model id."""
    if alias_or_hf_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[alias_or_hf_id].hf_id
    return alias_or_hf_id


def required_models() -> list[str]:
    return ["distilbert", "bert_base", "roberta_base", "deberta_v3_base", "modernbert"]
