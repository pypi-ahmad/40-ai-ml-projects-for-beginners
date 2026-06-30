from __future__ import annotations

import pytest

from peft_platform.peft.registry import PeftMethod, list_methods, method_supported_for_model, parse_method


def test_methods_list_contains_all() -> None:
    methods = list_methods()
    assert "lora" in methods
    assert "qlora" in methods
    assert "full_finetune" in methods


def test_parse_method_valid() -> None:
    assert parse_method("lora") == PeftMethod.LORA


def test_parse_method_invalid() -> None:
    with pytest.raises(ValueError):
        parse_method("nope")


def test_method_support_logic() -> None:
    assert method_supported_for_model(PeftMethod.LORA, "qwen")
    assert not method_supported_for_model(PeftMethod.ADAPTER_FUSION, "modernbert")
