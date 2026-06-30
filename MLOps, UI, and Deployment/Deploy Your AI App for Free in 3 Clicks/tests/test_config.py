"""Configuration resolution and validation tests."""

from __future__ import annotations

from streamlit_app.utils import config as cfg


def _reset_cache() -> None:
    cfg.get_settings.cache_clear()


def test_env_used_when_secrets_absent(monkeypatch):
    monkeypatch.setattr(cfg, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setenv("HF_API_TOKEN", "env-token")
    monkeypatch.setenv("SENTIMENT_MODEL", "qwen3.5:4b")

    _reset_cache()
    settings = cfg.get_settings()

    assert settings.hf_api_token == "env-token"
    assert settings.models.sentiment == "qwen3.5:4b"


def test_secrets_override_env(monkeypatch):
    monkeypatch.setenv("HF_API_TOKEN", "env-token")
    monkeypatch.setenv("SENTIMENT_MODEL", "qwen3.5:4b")
    monkeypatch.setattr(
        cfg,
        "_load_streamlit_secrets",
        lambda: {
            "HF_API_TOKEN": "secret-token",
            "SENTIMENT_MODEL": "qwen3.5:2b",
            "MAX_CATEGORIES": 12,
        },
    )

    _reset_cache()
    settings = cfg.get_settings()

    assert settings.hf_api_token == "secret-token"
    assert settings.models.sentiment == "qwen3.5:2b"
    assert settings.limits.max_categories == 12


def test_invalid_numeric_env_uses_defaults(monkeypatch):
    monkeypatch.setattr(cfg, "_load_streamlit_secrets", lambda: {})
    monkeypatch.setenv("MAX_CATEGORIES", "not-a-number")

    _reset_cache()
    settings = cfg.get_settings()

    assert settings.limits.max_categories == cfg.ValidationLimits.max_categories


def test_is_hf_configured(monkeypatch):
    monkeypatch.setattr(cfg, "_load_streamlit_secrets", lambda: {"HF_API_TOKEN": "abc"})
    _reset_cache()
    assert cfg.is_hf_configured() is True

    monkeypatch.setattr(cfg, "_load_streamlit_secrets", lambda: {})
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    _reset_cache()
    assert cfg.is_hf_configured() is False
