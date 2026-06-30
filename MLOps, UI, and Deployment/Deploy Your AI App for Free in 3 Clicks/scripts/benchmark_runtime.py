"""Collect runtime metrics used in optimization and monitoring chapters."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import psutil

from streamlit_app.utils import models

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "metrics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def measure_startup_time() -> float:
    """Measure cold import time for Streamlit app module."""
    start = time.perf_counter()
    subprocess.run(
        ["python", "-c", "import streamlit_app.app"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return round(time.perf_counter() - start, 4)


def measure_inference_time(text: str, iterations: int = 200) -> dict[str, float]:
    """Measure cold/warm and cache-vs-no-cache latency for sentiment inference."""
    # Force deterministic local fallback path so benchmark is stable in CI/sandbox.
    models._call_hf_inference_api = lambda *args, **kwargs: None
    models._call_ollama_api = lambda *args, **kwargs: None
    models.clear_model_caches()
    models.reset_runtime_stats()

    t0 = time.perf_counter()
    models.analyze_sentiment(text)
    cold = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    models.analyze_sentiment(text)
    warm = (time.perf_counter() - t1) * 1000

    t2 = time.perf_counter()
    for _ in range(iterations):
        models.analyze_sentiment(text, use_cache=False)
    no_cache_avg = ((time.perf_counter() - t2) * 1000) / iterations

    t3 = time.perf_counter()
    for _ in range(iterations):
        models.analyze_sentiment(text, use_cache=True)
    cache_avg = ((time.perf_counter() - t3) * 1000) / iterations

    speedup = no_cache_avg / max(cache_avg, 1e-6)
    return {
        "cold_predict_ms": round(cold, 4),
        "warm_predict_ms": round(warm, 4),
        "avg_predict_no_cache_ms": round(no_cache_avg, 4),
        "avg_predict_with_cache_ms": round(cache_avg, 4),
        "cache_speedup_x": round(speedup, 3),
    }


def main() -> int:
    process = psutil.Process()
    baseline_memory_mb = round(process.memory_info().rss / (1024 ** 2), 2)

    startup_seconds = measure_startup_time()
    latency_metrics = measure_inference_time(
        "The product quality is great and the experience feels excellent."
    )

    payload = {
        "timestamp": time.time(),
        "startup_seconds": startup_seconds,
        "memory_rss_mb": baseline_memory_mb,
        **latency_metrics,
        "runtime_stats": models.get_runtime_stats(),
        "notes": "Cold vs warm and no-cache vs cache numbers are measured from real execution.",
    }

    out_path = OUT_DIR / "runtime_benchmark.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))
    print(f"Saved benchmark to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
