from __future__ import annotations

from pathlib import Path

from peft_platform.peft.registry import PeftMethod
from peft_platform.training.runner import TrainingRunner


def test_training_smoke(tmp_path: Path) -> None:
    runner = TrainingRunner(tmp_path)
    result = runner.run_smoke("TinyLlama/TinyLlama-1.1B-Chat-v1.0", PeftMethod.LORA, steps=3)
    assert result.steps == 3
    assert Path(result.output_dir).exists()
