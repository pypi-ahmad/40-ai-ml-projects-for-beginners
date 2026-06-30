"""Compare direct Ollama REST vs LangChain orchestration."""

from __future__ import annotations

import asyncio
from statistics import mean
from typing import Any

from ai_spreadsheet_analytics.llm.langchain_adapter import LangChainOllamaAdapter
from ai_spreadsheet_analytics.llm.ollama_rest import OllamaRESTClient


async def compare_ollama_paths(
    prompt: str,
    model: str,
    runs: int = 3,
    base_url: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    """Benchmark direct REST vs LangChain adapter.

    Returns:
        Latency and sample outputs for both paths.
    """
    rest = OllamaRESTClient(base_url=base_url)
    langchain = LangChainOllamaAdapter(base_url=base_url)

    rest_latencies: list[float] = []
    rest_samples: list[str] = []
    for _ in range(runs):
        response = await rest.agenerate(model=model, prompt=prompt, temperature=0.0)
        rest_latencies.append(response.latency_ms)
        rest_samples.append(response.text)

    lc_latencies: list[float] = []
    lc_samples: list[str] = []
    for _ in range(runs):
        response = await langchain.agenerate(model=model, prompt=prompt, temperature=0.0)
        lc_latencies.append(response.latency_ms)
        lc_samples.append(response.text)

    return {
        "runs": runs,
        "rest": {
            "avg_latency_ms": mean(rest_latencies),
            "min_latency_ms": min(rest_latencies),
            "max_latency_ms": max(rest_latencies),
            "sample": rest_samples[-1] if rest_samples else "",
        },
        "langchain": {
            "avg_latency_ms": mean(lc_latencies),
            "min_latency_ms": min(lc_latencies),
            "max_latency_ms": max(lc_latencies),
            "sample": lc_samples[-1] if lc_samples else "",
        },
    }


def compare_ollama_paths_sync(prompt: str, model: str, runs: int = 3, base_url: str = "http://127.0.0.1:11434") -> dict[str, Any]:
    """Synchronous wrapper."""
    return asyncio.run(compare_ollama_paths(prompt=prompt, model=model, runs=runs, base_url=base_url))
