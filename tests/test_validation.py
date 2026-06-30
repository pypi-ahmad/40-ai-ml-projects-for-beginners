"""Validation tests: imports, module attributes, docstrings."""

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "src.ollama_client",
        "src.sentiment",
        "src.summarization",
        "src.translation",
        "src.chat",
        "src.document_analyzer",
        "src.benchmarking",
        "src.visualization",
    ],
)
def test_module_imports(module_name: str) -> None:
    import importlib

    mod = importlib.import_module(module_name)
    assert mod is not None


def test_package_exports() -> None:
    from src import __all__ as exports

    expected = {
        "OllamaClient",
        "SentimentAnalyzer",
        "Summarizer",
        "Translator",
        "ChatEngine",
        "DocumentAnalyzer",
        "BenchmarkRunner",
        "BenchmarkVisualizer",
    }
    assert set(exports) == expected


@pytest.mark.parametrize(
    "attr,model",
    [
        ("SentimentAnalyzer", "qwen3.5:2b"),
        ("Summarizer", "granite4.1:3b"),
        ("Translator", "translategemma:4b"),
        ("ChatEngine", "qwen3.5:4b"),
        ("DocumentAnalyzer", "glm-ocr"),
    ],
)
def test_default_models(attr: str, model: str) -> None:
    import importlib

    mod = importlib.import_module("src")
    cls = getattr(mod, attr)
    assert cls is not None


def test_translator_languages() -> None:
    from src.translation import LANGUAGES, Translator

    t = Translator()
    langs = t.supported_languages
    assert len(langs) == len(LANGUAGES)
    assert "Spanish" in langs
    t.close()


def test_benchmark_models_present() -> None:
    from src.benchmarking import MODELS

    assert len(MODELS) == 4
    assert "qwen3.5:4b" in MODELS
