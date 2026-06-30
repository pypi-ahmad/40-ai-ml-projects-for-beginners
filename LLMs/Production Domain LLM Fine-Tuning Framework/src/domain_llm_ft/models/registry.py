"""Model registry and loading utilities."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import AutoModelForSequenceClassification


@dataclass(frozen=True)
class ModelSpec:
    key: str
    hf_name: str


REQUIRED_MODELS: dict[str, ModelSpec] = {
    "distilbert": ModelSpec("distilbert", "distilbert-base-uncased"),
    "bert_base": ModelSpec("bert_base", "bert-base-uncased"),
    "roberta_base": ModelSpec("roberta_base", "roberta-base"),
    "deberta_v3": ModelSpec("deberta_v3", "microsoft/deberta-v3-base"),
    "modernbert": ModelSpec("modernbert", "answerdotai/ModernBERT-base"),
    "minilm": ModelSpec("minilm", "microsoft/MiniLM-L12-H384-uncased"),
    "e5": ModelSpec("e5", "intfloat/e5-base-v2"),
}

OPTIONAL_MODELS: dict[str, ModelSpec] = {
    "qwen": ModelSpec("qwen", "Qwen/Qwen2.5-0.5B-Instruct"),
    "gemma": ModelSpec("gemma", "google/gemma-2-2b-it"),
    "phi": ModelSpec("phi", "microsoft/Phi-3-mini-4k-instruct"),
    "tinyllama": ModelSpec("tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
}


def resolve_model_name(name_or_key: str) -> str:
    """Resolve model key to HF model name."""
    if name_or_key in REQUIRED_MODELS:
        return REQUIRED_MODELS[name_or_key].hf_name
    if name_or_key in OPTIONAL_MODELS:
        return OPTIONAL_MODELS[name_or_key].hf_name
    return name_or_key


def load_sequence_classifier(
    model_name: str,
    num_labels: int,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
):
    """Load sequence classification model with task metadata."""
    kwargs: dict[str, object] = {"num_labels": num_labels}
    if id2label is not None:
        kwargs["id2label"] = id2label
    if label2id is not None:
        kwargs["label2id"] = label2id
    return AutoModelForSequenceClassification.from_pretrained(
        resolve_model_name(model_name),
        **kwargs,
    )
