from __future__ import annotations

from peft_platform.inference.engine import InferenceEngine


def test_inference_mock_generation() -> None:
    engine = InferenceEngine()
    engine.load()
    output = engine.generate("hello")
    assert output.text
    assert output.tokens_generated >= 1
