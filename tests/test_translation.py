from unittest.mock import MagicMock, patch

import pytest

from src.translation import LANGUAGES, Translator


@pytest.fixture
def translator() -> Translator:
    t = Translator()
    yield t
    t.close()


def test_parse_valid_json() -> None:
    raw = '{"translated_text": "hola"}'
    result = Translator._parse(raw)
    assert result["translated_text"] == "hola"


def test_parse_json_with_markdown_fence() -> None:
    raw = '```json\n{"translated_text": "bonjour"}\n```'
    result = Translator._parse(raw)
    assert result["translated_text"] == "bonjour"


def test_parse_fallback_on_garbage() -> None:
    result = Translator._parse("not json at all")
    assert result["translated_text"] == "not json at all"


def test_parse_empty_string() -> None:
    result = Translator._parse("")
    assert result["translated_text"] == "translation failed"


def test_supported_languages(translator: Translator) -> None:
    langs = translator.supported_languages
    assert len(langs) == len(LANGUAGES)
    assert all(isinstance(lang, str) for lang in langs)
    assert langs == sorted(langs)
    assert "Spanish" in langs
    assert "French" in langs


def test_supported_languages_is_sorted(translator: Translator) -> None:
    langs = translator.supported_languages
    assert langs == sorted(langs)


@patch("src.translation.OllamaClient")
def test_translate_makes_api_call(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.generate.return_value = {
        "response": '{"translated_text": "Hola mundo"}'
    }
    t = Translator()
    result = t.translate("Hello world", "Spanish")
    assert result["translated_text"] == "Hola mundo"
    mock_ollama.return_value.generate.assert_called_once()
    t.close()


@patch("src.translation.OllamaClient")
def test_translate_api_fallback_on_bad_response(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.generate.return_value = {"response": "nope"}
    t = Translator()
    result = t.translate("Hello", "French")
    assert "translated_text" in result
    t.close()


@patch("src.translation.OllamaClient")
def test_translate_empty_text(mock_ollama: MagicMock) -> None:
    mock_ollama.return_value.generate.return_value = {"response": '{"translated_text": ""}'}
    t = Translator()
    result = t.translate("", "Spanish")
    assert result["translated_text"] == ""
    t.close()
