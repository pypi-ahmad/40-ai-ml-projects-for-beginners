from __future__ import annotations

import pytest

from peft_platform.data.schemas import Sample
from peft_platform.data.templates import apply_template, supported_templates


def test_templates_supported() -> None:
    names = supported_templates()
    assert "alpaca" in names
    assert "qwen" in names


def test_apply_template() -> None:
    sample = Sample(task_type="instruction", instruction="Explain", input="LoRA", output="Low rank")
    formatted = apply_template(sample, "alpaca")
    assert "### Instruction:" in formatted
    assert "Low rank" in formatted


def test_apply_template_invalid() -> None:
    sample = Sample(task_type="instruction", instruction="Explain", input="LoRA", output="Low rank")
    with pytest.raises(ValueError):
        apply_template(sample, "unknown")
