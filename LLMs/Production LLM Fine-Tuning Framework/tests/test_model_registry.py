from llmft.models import ModelRegistry


def test_model_registry_resolves_primary_without_inventory() -> None:
    registry = ModelRegistry()
    result = registry.resolve("llama3_8b")
    assert result.alias == "llama3_8b"
    assert result.used_fallback is False


def test_model_registry_uses_fallback_when_primary_missing() -> None:
    registry = ModelRegistry()
    available = {"meta-llama/Llama-3.1-8B-Instruct"}
    result = registry.resolve("llama3_8b", available_model_ids=available, allow_fallback=True)
    assert result.used_fallback is True
    assert "fallback" in result.reason
