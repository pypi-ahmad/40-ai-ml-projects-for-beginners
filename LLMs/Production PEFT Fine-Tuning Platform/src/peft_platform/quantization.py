"""Quantization profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class QuantConfig:
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    quant_type: str = "nf4"
    double_quant: bool = True


def resolve_quant_config(profile: str) -> QuantConfig:
    profile = profile.lower()
    if profile == "4bit-nf4":
        return QuantConfig(load_in_4bit=True, quant_type="nf4")
    if profile == "4bit-fp4":
        return QuantConfig(load_in_4bit=True, quant_type="fp4")
    if profile == "8bit":
        return QuantConfig(load_in_8bit=True, quant_type="int8", double_quant=False)
    if profile == "none":
        return QuantConfig()
    raise ValueError(f"Unknown quantization profile: {profile}")
