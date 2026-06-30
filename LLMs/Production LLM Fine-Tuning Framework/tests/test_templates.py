from llmft.templates import TemplateRegistry


def test_template_registry_has_required_templates() -> None:
    registry = TemplateRegistry()
    expected = {"alpaca", "chatml", "llama3", "mistral", "qwen", "phi", "gemma", "custom"}
    assert expected.issubset(set(registry.names()))


def test_template_render_contains_sections() -> None:
    registry = TemplateRegistry()
    text = registry.render("alpaca", "do task", "context", "answer")
    assert "### Instruction" in text
    assert "### Response" in text
