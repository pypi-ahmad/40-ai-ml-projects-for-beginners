from __future__ import annotations

import importlib

import pytest


@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None, reason="streamlit missing")
def test_streamlit_module_importable() -> None:
    module = importlib.import_module("peft_platform.ui.streamlit_app")
    assert module is not None
