from api_intel_agent.config import load_settings


def test_load_settings_has_required_sections():
    settings = load_settings()
    assert settings.llm.default_model
    assert settings.agent.max_iterations > 0
    assert "github" in settings.apis
