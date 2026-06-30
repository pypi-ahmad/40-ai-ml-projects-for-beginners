"""
Home page — project overview, architecture, quick start.

Teaches what deployment means and why it matters.
"""

import streamlit as st


def render():
    st.markdown("# 🚀 Deploy Your AI App for Free in 3 Clicks")
    st.markdown(
        "**Build a production-ready AI application and deploy it to the public internet "
        "— zero server cost, zero DevOps, zero excuses.**"
    )

    st.markdown("---")

    st.markdown("## What You Will Learn")
    col1, col2, col3 = st.columns(3)

    lessons = [
        (
            "🛠️",
            "Build Multi-Page AI Apps",
            "Create a Streamlit application with sentiment analysis, "
            "summarization, classification, and translation features. "
            "Learn component architecture and state management.",
        ),
        (
            "🌐",
            "API Integration Patterns",
            "Use Hugging Face Inference API for AI inference without GPUs. "
            "Implement fallback strategies so your app never breaks. "
            "Handle rate limits and errors gracefully.",
        ),
        (
            "☁️",
            "Deploy to Streamlit Cloud",
            "Push code to GitHub, connect to Streamlit Cloud, deploy in 3 clicks. "
            "Configure secrets, monitor usage, update seamlessly.",
        ),
    ]

    for col, (emoji, title, desc) in zip([col1, col2, col3], lessons):
        with col:
            st.markdown(f"### {emoji} {title}")
            st.markdown(desc)

    st.markdown("---")

    st.markdown("## Architecture Overview")
    st.markdown(
        """
    ```
    User Browser
        │
        ▼
    Streamlit Cloud (free tier)
        │
        ├── streamlit_app/
        │   ├── app.py          ← Entry point
        │   ├── pages/          ← Multi-page routes
        │   ├── components/     ← Reusable UI parts
        │   └── utils/          ← Models, caching, helpers
        │
        ├── pyproject.toml      ← Dependencies
        ├── .streamlit/secrets.toml  ← API keys
        └── .github/            ← CI/CD (optional)
              │
              ▼
    Hugging Face Inference API  ← AI backend (free tier)
    ```
    """
    )

    st.markdown("---")

    st.markdown("## Quick Start Guide")
    steps = [
        (
            "1",
            "Set up API Token",
            "Get a free Hugging Face token → set as `HF_API_TOKEN` in `.streamlit/secrets.toml`. "
            "Without token, app runs in fallback mode.",
        ),
        (
            "2",
            "Run Locally",
            "```bash\nuv sync\nstreamlit run streamlit_app/app.py\n```",
        ),
        (
            "3",
            "Push to GitHub",
            "Create repo → push code → go to [Streamlit Cloud](https://streamlit.io/cloud) → "
            "Deploy in 3 clicks.",
        ),
        (
            "4",
            "Configure & Launch",
            "Add `HF_API_TOKEN` in Streamlit Cloud secrets → app goes live → share URL.",
        ),
    ]

    for num, title, content in steps:
        with st.expander(f"**Step {num}**: {title}", expanded=False):
            st.markdown(content)

    st.markdown("---")

    st.markdown("## Project Structure")
    st.code(
        """
        Deploy-Your-AI-App/
        ├── streamlit_app/
        │   ├── __init__.py
        │   ├── app.py                 # Main entry point
        │   ├── pages/
        │   │   ├── home.py            # ← You are here
        │   │   ├── sentiment.py
        │   │   ├── summarization.py
        │   │   ├── classification.py
        │   │   └── translation.py
        │   ├── components/
        │   │   ├── sidebar.py         # Navigation & settings
        │   │   └── ui_components.py   # Reusable UI elements
        │   └── utils/
        │       ├── models.py          # Inference engine
        │       ├── caching.py         # Caching layer
        │       └── helpers.py         # Utilities
        ├── notebooks/                 # Tutorial notebooks
        ├── tests/                     # Test suite
        ├── pyproject.toml
        ├── .gitignore
        └── README.md
        """,
        language="",
    )

    st.markdown("---")
    st.info(
        "👈 **Navigate using the sidebar** to explore each AI feature. "
        "Start with Sentiment Analysis to see the app in action!"
    )

    st.caption(
        "Built with Streamlit • Hugging Face Inference API • Python "
        "• Part of the 40 AI-ML Projects for Beginners series"
    )


def show():
    """Backward-compatible alias used by older tests/material."""
    render()
