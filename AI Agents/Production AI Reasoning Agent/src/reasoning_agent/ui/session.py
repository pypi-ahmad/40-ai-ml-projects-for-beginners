"""Session state helpers for Streamlit UI."""

from __future__ import annotations

from typing import Any

import streamlit as st


DEFAULT_KEYS: dict[str, Any] = {
    "chat_history": [],
    "run_history": [],
    "selected_run_index": None,
    "ui_dark_mode": True,
    "settings_override": {},
}


def ensure_session_state() -> None:
    for key, value in DEFAULT_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def append_chat(role: str, content: str) -> None:
    st.session_state.chat_history.append({"role": role, "content": content})


def append_run(run: dict[str, Any]) -> None:
    st.session_state.run_history.append(run)
    st.session_state.selected_run_index = len(st.session_state.run_history) - 1


def selected_run() -> dict[str, Any] | None:
    idx = st.session_state.selected_run_index
    if idx is None:
        return None
    if idx < 0 or idx >= len(st.session_state.run_history):
        return None
    return st.session_state.run_history[idx]
