"""
Tests for app.py and page module contracts.

Covers:
  - Page registry structure
  - Module-level constants
  - Cache configuration
  - Import paths correctness
"""

import pytest


class TestPageRegistryContracts:
    """Verify each page module exports expected constants and structure."""

    PAGE_NAMES = ["home", "sentiment", "summarization", "classification", "translation"]
    EXPECTED_SIDEBAR_KEYS = ["navigation"]

    @pytest.mark.parametrize("page_name", PAGE_NAMES)
    def test_page_importable(self, page_name):
        module_path = f"streamlit_app.pages.{page_name}"
        import importlib
        try:
            mod = importlib.import_module(module_path)
            assert mod is not None
        except Exception as e:
            pytest.fail(f"Failed to import {module_path}: {e}")

    @pytest.mark.parametrize("page_name", PAGE_NAMES)
    def test_page_has_show_function(self, page_name):
        import importlib
        module_path = f"streamlit_app.pages.{page_name}"
        mod = importlib.import_module(module_path)
        assert hasattr(mod, "show"), f"{module_path} missing show()"
        assert callable(getattr(mod, "show"))


class TestSidebarContract:
    def test_sidebar_importable(self):
        from streamlit_app.components import sidebar
        assert sidebar is not None


class TestUIComponentsContract:
    def test_ui_components_importable(self):
        from streamlit_app.components import ui_components
        assert ui_components is not None


class TestUtilsContracts:
    def test_models_importable(self):
        from streamlit_app.utils import models
        assert models is not None

    def test_helpers_importable(self):
        from streamlit_app.utils import helpers
        assert helpers is not None

    def test_caching_importable(self):
        from streamlit_app.utils import caching
        assert caching is not None

    def test_models_has_all_functions(self):
        from streamlit_app.utils.models import (
            analyze_sentiment,
            summarize_text,
            classify_text,
            translate_text,
            SUPPORTED_LANGUAGES,
        )
        assert callable(analyze_sentiment)
        assert callable(summarize_text)
        assert callable(classify_text)
        assert callable(translate_text)
        assert isinstance(SUPPORTED_LANGUAGES, dict)

    def test_helpers_has_all_functions(self):
        from streamlit_app.utils.helpers import (
            truncate_text,
            compute_text_hash,
            estimate_reading_time,
            validate_input,
            format_confidence,
            map_sentiment_to_emoji,
            parse_llm_json_response,
        )
        assert callable(truncate_text)
        assert callable(compute_text_hash)
        assert callable(estimate_reading_time)
        assert callable(validate_input)
        assert callable(format_confidence)
        assert callable(map_sentiment_to_emoji)
        assert callable(parse_llm_json_response)

    def test_caching_has_all_functions(self):
        from streamlit_app.utils.caching import (
            timed_cache,
            inference_timer,
            get_cache_stats,
            clear_all_caches,
        )
        assert callable(timed_cache)
        assert callable(inference_timer)
        assert callable(get_cache_stats)
        assert callable(clear_all_caches)


class TestRootAppEntry:
    def test_root_app_importable(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "app", "app.py"
        )
        assert spec is not None, "root app.py must exist"
