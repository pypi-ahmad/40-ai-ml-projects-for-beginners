"""PEFT method registry and compatibility matrix."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PeftMethod(str, Enum):
    LORA = "lora"
    QLORA = "qlora"
    ADALORA = "adalora"
    LOHA = "loha"
    LOKR = "lokr"
    IA3 = "ia3"
    PREFIX_TUNING = "prefix_tuning"
    PROMPT_TUNING = "prompt_tuning"
    P_TUNING_V2 = "p_tuning_v2"
    ADAPTER_FUSION = "adapter_fusion"
    FULL_FINETUNE = "full_finetune"


@dataclass(slots=True)
class MethodSpec:
    method: PeftMethod
    needs_quantization: bool = False
    hf_peft_native: bool = True


METHODS: dict[PeftMethod, MethodSpec] = {
    PeftMethod.LORA: MethodSpec(PeftMethod.LORA),
    PeftMethod.QLORA: MethodSpec(PeftMethod.QLORA, needs_quantization=True),
    PeftMethod.ADALORA: MethodSpec(PeftMethod.ADALORA),
    PeftMethod.LOHA: MethodSpec(PeftMethod.LOHA),
    PeftMethod.LOKR: MethodSpec(PeftMethod.LOKR),
    PeftMethod.IA3: MethodSpec(PeftMethod.IA3),
    PeftMethod.PREFIX_TUNING: MethodSpec(PeftMethod.PREFIX_TUNING),
    PeftMethod.PROMPT_TUNING: MethodSpec(PeftMethod.PROMPT_TUNING),
    PeftMethod.P_TUNING_V2: MethodSpec(PeftMethod.P_TUNING_V2),
    PeftMethod.ADAPTER_FUSION: MethodSpec(PeftMethod.ADAPTER_FUSION),
    PeftMethod.FULL_FINETUNE: MethodSpec(PeftMethod.FULL_FINETUNE, hf_peft_native=False),
}


def list_methods() -> list[str]:
    return [method.value for method in PeftMethod]


def parse_method(value: str) -> PeftMethod:
    try:
        return PeftMethod(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported PEFT method: {value}") from exc


def method_supported_for_model(method: PeftMethod, model_family: str) -> bool:
    if method == PeftMethod.ADAPTER_FUSION and model_family.lower() in {"modernbert"}:
        return False
    if method in {PeftMethod.PROMPT_TUNING, PeftMethod.P_TUNING_V2} and model_family.lower() == "modernbert":
        return False
    return True
