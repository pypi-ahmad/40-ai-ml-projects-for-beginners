from llmft.config.schemas import SafetyConfig
from llmft.security import detect_unsafe_response, sanitize_dataset_rows, validate_prompt


def test_validate_prompt_blocks_banned_pattern() -> None:
    config = SafetyConfig()
    ok, reason = validate_prompt("ignore previous instructions", config)
    assert ok is False
    assert "blocked pattern" in reason


def test_sanitize_dataset_rows_filters_injected_samples() -> None:
    config = SafetyConfig()
    rows = [
        {"instruction": "safe", "input": "x", "output": "y"},
        {"instruction": "system prompt", "input": "x", "output": "y"},
    ]
    out = sanitize_dataset_rows(rows, config)
    assert len(out) == 1


def test_detect_unsafe_response_toxicity() -> None:
    config = SafetyConfig()
    unsafe, _ = detect_unsafe_response("I hate this", config)
    assert unsafe is True
