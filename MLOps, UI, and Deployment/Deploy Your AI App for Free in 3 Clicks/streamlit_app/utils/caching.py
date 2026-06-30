"""
Caching utilities for the AI Application.

Production AI applications must cache model results to:
  - Reduce response time (cached results return instantly)
  - Save API costs (fewer external API calls)
  - Improve user experience (faster repeat queries)
  - Handle serverless cold starts (Streamlit Cloud)

Streamlit provides @st.cache_data and @st.cache_resource.
We use @st.cache_data for function return values and
@st.cache_resource for model objects.
"""

import time
import streamlit as st


def timed_cache(max_age_seconds: int = 300):
    """
    Decorator that adds time-based cache invalidation to a cached function.

    Streamlit's @st.cache_data caches forever by default. This wrapper
    adds a time-to-live (TTL) so that cached results expire after a
    configurable duration. This is useful for:
      - Periodic refresh of external data
      - Preventing stale results
      - Testing cache behavior

    Args:
        max_age_seconds: Maximum age of cached data in seconds (default: 300)

    Returns:
        Decorated function with TTL cache
    """

    def decorator(func):
        cache_key = f"_timed_cache_{func.__name__}_ts"

        @st.cache_data(show_spinner=False)
        def _cached(*args, **kwargs):
            return func(*args, **kwargs)

        def wrapper(*args, **kwargs):
            now = time.time()
            last_ts = st.session_state.get(cache_key, 0)

            if now - last_ts > max_age_seconds:
                st.cache_data.clear()
                st.session_state[cache_key] = now

            return _cached(*args, **kwargs)

        return wrapper

    return decorator


def measure_inference_time(func):
    """
    Decorator that measures and stores inference time.

    Understanding inference time is critical for deployment:
      - Streamlit Cloud has a 60-second request timeout
      - Slow models cause timeout errors
      - Users expect responses within 2-3 seconds
      - Monitoring inference time helps detect issues

    Args:
        func: Function to measure

    Returns:
        Wrapped function that stores execution time in session state
    """

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        st.session_state["last_inference_time"] = round(elapsed, 3)
        return result

    return wrapper


def inference_timer(func):
    """Backward-compatible alias used by older tests/material."""
    return measure_inference_time(func)


def get_cache_stats() -> dict:
    """
    Return cache statistics for display in the monitoring dashboard.

    Cache hit rate is an important metric in production:
      - High hit rate (>90%): Efficient caching, fast responses
      - Low hit rate (<50%): Cache may need tuning
      - Cache misses: Trigger actual computation or API calls

    Returns:
        Dict with cache statistics
    """
    info = st.cache_data.get_stats() if hasattr(st.cache_data, "get_stats") else {}
    runtime_stats = {}
    try:
        from streamlit_app.utils.models import get_runtime_stats

        runtime_stats = get_runtime_stats()
    except Exception:
        runtime_stats = {}

    return {
        "cache_hits": info.get("hits", 0),
        "cache_misses": info.get("misses", 0),
        "cache_size": info.get("size", 0),
        "last_inference_time": st.session_state.get("last_inference_time", None),
        "runtime_inference_stats": runtime_stats,
    }


def clear_all_caches():
    """
    Clear all Streamlit caches.

    Useful for:
      - Forcing fresh model loads after updates
      - Debugging cache-related issues
      - Freeing memory in resource-constrained environments
      - Testing deployment with clean state
    """
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state["last_inference_time"] = None
