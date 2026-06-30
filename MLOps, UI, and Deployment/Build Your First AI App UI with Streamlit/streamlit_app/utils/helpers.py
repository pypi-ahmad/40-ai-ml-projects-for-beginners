"""Helper utilities shared across pages, notebooks, and tests."""

from __future__ import annotations

from datetime import datetime, UTC
import json
from pathlib import Path
from typing import Any

import streamlit as st

from streamlit_app.config import APP_CONFIG


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"


def ensure_output_dirs() -> None:
    """Create output directories for figures and metrics."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)


def truncate_text(text: str, max_chars: int = 200) -> str:
    """Truncate text while keeping word boundaries where possible."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def validate_text_input(text: str | None, min_length: int = 3) -> str | None:
    """Validate text input and return error message when invalid."""
    if not text or not text.strip():
        return "Please enter some text."
    if len(text.strip()) < min_length:
        return f"Text must be at least {min_length} characters."
    return None


def validate_categories(categories: list[str]) -> str | None:
    """Validate category list for classification tasks."""
    clean = [category.strip() for category in categories if category.strip()]
    if len(clean) < 2:
        return "Provide at least 2 categories for classification."
    if len(set(clean)) != len(clean):
        return "Categories must be unique."
    return None


def format_latency(seconds: float) -> str:
    """Format latency into a human-friendly representation."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.1f} min"


def now_utc_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(UTC).isoformat()


def load_sample_text(key: str) -> str:
    """Return reusable sample content for demos and notebooks."""
    samples = {
        "sentiment_positive": (
            "This product is absolutely amazing. The workflow is intuitive, the response "
            "time is fast, and support solved my issue in minutes."
        ),
        "sentiment_negative": (
            "Very frustrating experience. The app crashed twice, settings did not persist, "
            "and onboarding instructions were unclear."
        ),
        "sentiment_neutral": (
            "The package arrived on Tuesday. It contained all listed items and matched "
            "the order invoice."
        ),
        "summary": (
            "Artificial intelligence systems are moving from experimental prototypes "
            "to production workflows across healthcare, finance, operations, and retail. "
            "Teams now optimize for reliability, observability, and governance in addition "
            "to model quality. Local model inference can reduce cloud spend and improve data "
            "privacy, but it introduces constraints in memory, throughput, and deployment "
            "automation. Strong AI application design combines clear user interfaces, robust "
            "input validation, resilient model orchestration, caching layers, and meaningful "
            "performance telemetry so teams can balance quality, speed, and cost."
        ),
        "classify": (
            "The central bank announced an emergency rate cut after inflation cooled faster "
            "than expected, while equity markets reacted with broad gains."
        ),
        "translate": "Hello team, please send the final contract draft before tomorrow noon.",
        "chat": "I need help designing a production-ready AI app architecture for local inference.",
    }
    return samples.get(key, "Sample text not found.")


def get_save_path(name: str) -> Path:
    """Return output path and ensure parent directories exist."""
    ensure_output_dirs()
    path = OUTPUTS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json_artifact(data: dict[str, Any] | list[dict[str, Any]], name: str) -> Path:
    """Persist JSON artifact to outputs/metrics."""
    ensure_output_dirs()
    target = METRICS_DIR / name
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return target


def init_session_state() -> None:
    """Initialize all Streamlit session state keys used by app pages."""
    defaults: dict[str, Any] = {
        "page": "Home",
        "theme": "light",
        "chat_history": [],
        "saved_prompts": [],
        "last_benchmark_summary": [],
        "last_benchmark_runs": [],
        "model_settings": {
            "temperature": APP_CONFIG.default_temperature,
            "max_tokens": APP_CONFIG.default_max_tokens,
        },
        "user_preferences": {
            "show_teaching_notes": True,
            "show_prompt_examples": True,
        },
        "selected_models": {
            "chat": APP_CONFIG.models.chat,
            "sentiment": APP_CONFIG.models.sentiment,
            "summarization": APP_CONFIG.models.summarization,
            "classification": APP_CONFIG.models.classification,
            "translation": APP_CONFIG.models.translation,
            "ocr_primary": APP_CONFIG.models.ocr_primary,
            "ocr_fallback": APP_CONFIG.models.ocr_fallback,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def add_to_chat_history(role: str, content: str) -> None:
    """Append message to chat history while preserving insertion order."""
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    st.session_state["chat_history"].append({"role": role, "content": content})


def trim_chat_history(max_turns: int = 20) -> None:
    """Limit chat history to avoid oversized prompts and slow responses."""
    history = st.session_state.get("chat_history", [])
    if len(history) > max_turns * 2:
        st.session_state["chat_history"] = history[-(max_turns * 2) :]


def save_prompt(prompt: str) -> None:
    """Store unique prompts to session state for quick reuse."""
    cleaned = prompt.strip()
    if not cleaned:
        return
    prompts: list[str] = st.session_state.get("saved_prompts", [])
    if cleaned not in prompts:
        prompts.insert(0, cleaned)
    st.session_state["saved_prompts"] = prompts[:20]


def apply_theme_css(theme: str) -> None:
    """Apply lightweight visual customization using CSS variables.

    Notes:
        Streamlit theme files are static. This helper gives in-app toggling for
        tutorial purposes, while keeping defaults accessible.
    """
    if theme == "dark":
        bg = "#0f172a"
        panel = "#111827"
        text = "#e5e7eb"
        accent = "#22d3ee"
    else:
        bg = "#f8fafc"
        panel = "#ffffff"
        text = "#0f172a"
        accent = "#0284c7"

    st.markdown(
        f"""
        <style>
            :root {{
                --app-bg: {bg};
                --panel-bg: {panel};
                --text-color: {text};
                --accent-color: {accent};
            }}
            .stApp {{
                background: radial-gradient(circle at top right, rgba(2,132,199,0.12), transparent 40%), var(--app-bg);
                color: var(--text-color);
            }}
            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 2.0rem;
            }}
            div[data-testid="stMetric"] {{
                background: var(--panel-bg);
                border-radius: 0.8rem;
                padding: 0.6rem;
            }}
            div[data-testid="stChatMessage"] {{
                border-radius: 0.8rem;
            }}
            .app-note {{
                border-left: 4px solid var(--accent-color);
                padding-left: 0.8rem;
                margin: 0.6rem 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
