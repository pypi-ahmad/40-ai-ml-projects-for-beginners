"""
Shared fixtures and configuration for the test suite.

This module provides:
  - Sample text fixtures for all test types
  - Clean environment helpers
  - Pytest configuration
"""

import os
import pytest


SAMPLE_TEXTS: dict[str, str] = {
    "positive": (
        "This product is absolutely amazing! I love the beautiful design "
        "and fantastic features. The customer service was excellent and "
        "helpful. I am very pleased with my purchase and would highly "
        "recommend it to everyone. Great experience overall!"
    ),
    "negative": (
        "This is the worst product I have ever purchased. The quality is "
        "terrible and the customer service was awful. I hate the poor design "
        "and horrible user experience. Very disappointing and frustrating. "
        "I would not recommend this to anyone."
    ),
    "neutral": (
        "The meeting is scheduled for tomorrow at 3 PM. Please bring the "
        "reports from last quarter. We will discuss the quarterly results "
        "and plan for the next phase of the project."
    ),
    "long_article": (
        "Artificial intelligence has transformed the way we interact with "
        "technology. From voice assistants to recommendation systems, AI "
        "is everywhere. Machine learning algorithms analyze vast amounts "
        "of data to make predictions and decisions. Deep neural networks "
        "have achieved remarkable results in image recognition, natural "
        "language processing, and game playing. The future of AI holds "
        "even more promise, with applications in healthcare, climate "
        "science, and education. However, challenges remain in ensuring "
        "AI systems are fair, transparent, and ethical. Researchers "
        "continue to work on making AI more interpretable and robust. "
        "The field evolves rapidly, with new breakthroughs announced "
        "regularly. Understanding AI basics is becoming essential for "
        "professionals across all industries."
    ),
    "tech": (
        "The new software update includes improved algorithms for data "
        "processing. The cloud-based platform uses machine learning to "
        "optimize network performance. Developers can integrate the API "
        "with existing systems using the digital transformation toolkit."
    ),
    "health": (
        "Regular exercise and proper nutrition are essential for maintaining "
        "good health. Patients should consult their doctor before starting "
        "any new treatment plan. Medical research shows that preventive care "
        "can reduce hospital visits and improve wellness outcomes."
    ),
}


@pytest.fixture
def sample_texts() -> dict[str, str]:
    """Return a dictionary of sample texts for testing."""
    return SAMPLE_TEXTS.copy()


@pytest.fixture
def clear_env():
    """Fixture that clears HF_API_TOKEN for testing fallback behavior."""
    old_token = os.environ.pop("HF_API_TOKEN", None)
    yield
    if old_token is not None:
        os.environ["HF_API_TOKEN"] = old_token


@pytest.fixture(autouse=True)
def no_external_apis(monkeypatch):
    """Prevent tests from hanging on HF/Ollama API timeouts.

    All test functions should exercise the rule-based fallback tier
    so they remain fast and hermetic regardless of network state.
    """
    monkeypatch.setattr(
        "streamlit_app.utils.models._call_hf_inference_api",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "streamlit_app.utils.models._call_ollama_api",
        lambda *args, **kwargs: None,
    )
