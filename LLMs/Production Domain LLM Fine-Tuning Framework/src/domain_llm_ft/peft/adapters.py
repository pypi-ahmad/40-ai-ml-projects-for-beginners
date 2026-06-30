"""PEFT adapter management."""

from __future__ import annotations

from pathlib import Path

from peft import LoraConfig, PeftModel, TaskType, get_peft_model

from domain_llm_ft.config.schemas import PeftConfigModel


def apply_peft(model, peft_cfg: PeftConfigModel):
    """Apply LoRA/QLoRA adapters to model."""
    if not peft_cfg.enabled or peft_cfg.strategy == "none":
        return model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=peft_cfg.r,
        lora_alpha=peft_cfg.alpha,
        lora_dropout=peft_cfg.dropout,
        target_modules=peft_cfg.target_modules,
        bias=peft_cfg.bias,
    )
    return get_peft_model(model, lora_config)


def save_adapter(model, output_dir: Path) -> None:
    """Persist PEFT adapter weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)


def load_adapter(base_model, adapter_dir: Path):
    """Load adapter into base model."""
    return PeftModel.from_pretrained(base_model, adapter_dir)


def merge_adapter(model):
    """Merge adapter into base model for deployment."""
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model
