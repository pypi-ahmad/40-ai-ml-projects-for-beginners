"""Model loading and PEFT integration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .registry import resolve_model_id

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None


@dataclass(slots=True)
class ModelBundle:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    strategy: str
    trainable_params: int
    total_params: int


def _count_params(model: PreTrainedModel) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def _attach_lora(model: PreTrainedModel, rank: int = 8, alpha: int = 16, dropout: float = 0.1) -> PreTrainedModel:
    if get_peft_model is None or LoraConfig is None or TaskType is None:
        raise RuntimeError("PEFT is unavailable; install peft dependency.")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    return get_peft_model(model, lora_config)


def load_model_bundle(
    model_name: str,
    num_labels: int,
    strategy: str = "full",
    gradient_checkpointing: bool = False,
    use_4bit: bool = False,
) -> ModelBundle:
    """Load tokenizer + model bundle using full fine-tuning or PEFT strategies."""
    hf_model_id = resolve_model_id(model_name)
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)

    quantization_config = None
    if use_4bit and strategy in {"qlora", "lora"}:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        hf_model_id,
        num_labels=num_labels,
        quantization_config=quantization_config,
    )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    final_strategy = strategy
    if strategy in {"lora", "qlora"}:
        try:
            model = _attach_lora(model)
        except Exception as exc:
            logger.warning(f"LoRA setup failed for {model_name}; fallback to full fine-tuning. reason={exc}")
            final_strategy = "full"

    trainable, total = _count_params(model)
    logger.info(
        f"Loaded model={hf_model_id} strategy={final_strategy} trainable={trainable} total={total}"
    )

    return ModelBundle(
        model=model,
        tokenizer=tokenizer,
        strategy=final_strategy,
        trainable_params=trainable,
        total_params=total,
    )
