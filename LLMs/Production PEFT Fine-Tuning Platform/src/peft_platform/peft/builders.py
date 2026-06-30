"""PEFT config builders."""

from __future__ import annotations

from typing import Any

from peft_platform.peft.registry import PeftMethod


def build_peft_config(
    method: PeftMethod,
    r: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
) -> Any:
    """Build PEFT config object with lazy imports.

    Returns native PEFT config object when available.
    """
    if method == PeftMethod.FULL_FINETUNE:
        return None

    try:
        from peft import (
            AdaLoraConfig,
            IA3Config,
            LoraConfig,
            LoHaConfig,
            LoKrConfig,
            PrefixTuningConfig,
            PromptEncoderConfig,
            PromptTuningConfig,
            TaskType,
        )
    except Exception as exc:
        raise RuntimeError("peft package not available") from exc

    common_kwargs: dict[str, Any] = {
        "task_type": TaskType.CAUSAL_LM,
        "target_modules": target_modules,
    }

    if method in {PeftMethod.LORA, PeftMethod.QLORA}:
        return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, **common_kwargs)
    if method == PeftMethod.ADALORA:
        return AdaLoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, **common_kwargs)
    if method == PeftMethod.LOHA:
        return LoHaConfig(r=r, alpha=alpha, module_dropout=dropout, **common_kwargs)
    if method == PeftMethod.LOKR:
        return LoKrConfig(r=r, alpha=alpha, module_dropout=dropout, **common_kwargs)
    if method == PeftMethod.IA3:
        return IA3Config(task_type=TaskType.CAUSAL_LM, target_modules=target_modules)
    if method == PeftMethod.PREFIX_TUNING:
        return PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)
    if method == PeftMethod.PROMPT_TUNING:
        return PromptTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)
    if method == PeftMethod.P_TUNING_V2:
        return PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=20)
    if method == PeftMethod.ADAPTER_FUSION:
        return LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, **common_kwargs)

    raise ValueError(f"Unsupported method: {method.value}")
