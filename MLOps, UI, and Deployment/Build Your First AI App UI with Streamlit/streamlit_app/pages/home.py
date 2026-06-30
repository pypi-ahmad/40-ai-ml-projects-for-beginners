"""Home page with architecture and learning path."""

from __future__ import annotations

import streamlit as st

from streamlit_app.config import APP_CONFIG
from streamlit_app.components.sidebar import PAGES
from streamlit_app.services.prompts import prompt_quality_examples
from streamlit_app.utils.models import is_ollama_available, list_available_models


def _render_architecture_diagram() -> None:
    st.markdown(
        """
```text
User Browser
   |
   v
Streamlit UI Layer (widgets, forms, layout)
   |
   v
Application Layer (validation, state, routing)
   |
   v
Model Service Layer (prompt templates, retries, parsing)
   |
   v
Inference Layer (Ollama local models)
   |
   v
Data + Artifact Layer (session state, outputs/figures, outputs/metrics)
```
        """
    )


def render() -> None:
    st.title("Build Your First AI App UI with Streamlit")
    st.caption("Project #6 - AI/ML Portfolio Build")

    available = is_ollama_available()
    left, right = st.columns([2, 1])

    with left:
        st.markdown(
            """
### What is an AI Application?
An AI application is software where core behavior is learned/inferred by models,
not fully hard-coded rules. Traditional software maps deterministic input to logic.
AI software adds probabilistic inference and confidence-aware output handling.

### Traditional Software vs AI-Powered Software
- Traditional: rules-first, deterministic outputs, lower runtime uncertainty.
- AI-powered: model-first, context-dependent outputs, requires prompt + eval + guardrails.

### Why Streamlit here?
- Fast path from Python to interactive UI.
- Great for ML/LLM demos, internal tools, and early-stage product validation.
- Easy state management and caching primitives for local model apps.
            """
        )

        if st.session_state["user_preferences"].get("show_teaching_notes", True):
            st.info(
                "Teaching note: Every page in this app maps one capability to a reusable "
                "AI app pattern (input validation -> inference -> post-processing -> UX feedback)."
            )

    with right:
        if available:
            st.success("Ollama daemon reachable.")
            model_count = len(list_available_models())
            st.metric("Local models detected", str(model_count))
        else:
            st.error("Ollama daemon not reachable.")
            st.caption("Start daemon with: `ollama serve` on your local machine.")

        st.metric("Configured benchmark runs", str(APP_CONFIG.benchmark_runs))
        st.metric("Cache TTL (seconds)", str(APP_CONFIG.cache_ttl_seconds))

    st.divider()
    st.subheader("AI Application Architecture")
    _render_architecture_diagram()

    st.subheader("Application Flow")
    st.markdown(
        """
1. User submits text/file input from UI widget.
2. Validation layer blocks empty or malformed input.
3. Prompt template builds structured instruction.
4. Model service calls Ollama and receives output.
5. Parser normalizes output into safe schema.
6. UI renders results, confidence, and diagnostics.
7. Optional artifacts (charts/CSV/JSON) are saved for analysis.
        """
    )

    st.subheader("Prompt Engineering Example")
    examples = prompt_quality_examples()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Bad prompt**")
        st.code(examples["bad"], language="text")
    with col2:
        st.markdown("**Good prompt**")
        st.code(examples["good"], language="text")

    st.subheader("Explore Mini-Apps")
    cols = st.columns(2)
    for idx, (name, short_label, description) in enumerate(PAGES[1:]):
        with cols[idx % 2]:
            with st.container(border=True):
                st.markdown(f"**{name}**")
                st.caption(short_label)
                st.write(description)
