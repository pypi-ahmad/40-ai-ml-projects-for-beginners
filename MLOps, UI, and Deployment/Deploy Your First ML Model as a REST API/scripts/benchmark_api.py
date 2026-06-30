"""Benchmark API latency/throughput for cold vs warm and single vs batch prediction."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import PERFORMANCE_DIR

SAMPLE_RECORD = {
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.9841,
    "AveBedrms": 1.0238,
    "Population": 322.0,
    "AveOccup": 2.5556,
    "Latitude": 37.88,
    "Longitude": -122.23,
}


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(pd.Series(values).quantile(q))


async def _run_single_predict_benchmark(
    client: httpx.AsyncClient,
    n_requests: int,
    api_key: str,
) -> tuple[dict[str, float], list[float]]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    latencies: list[float] = []

    start = time.perf_counter()
    response = await client.post("/predict", json=SAMPLE_RECORD, headers=headers)
    response.raise_for_status()
    cold_ms = (time.perf_counter() - start) * 1000.0

    warm_start = time.perf_counter()
    for _ in range(n_requests):
        req_start = time.perf_counter()
        response = await client.post("/predict", json=SAMPLE_RECORD, headers=headers)
        response.raise_for_status()
        latencies.append((time.perf_counter() - req_start) * 1000.0)
    total_warm_s = time.perf_counter() - warm_start

    stats = {
        "cold_start_ms": cold_ms,
        "warm_avg_ms": float(statistics.mean(latencies)) if latencies else 0.0,
        "warm_p50_ms": _percentile(latencies, 0.50),
        "warm_p95_ms": _percentile(latencies, 0.95),
        "throughput_rps": float(n_requests / total_warm_s) if total_warm_s > 0 else 0.0,
    }
    return stats, latencies


async def _run_batch_benchmark(
    client: httpx.AsyncClient,
    batch_size: int,
    n_requests: int,
    api_key: str,
) -> dict[str, float]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    payload = {"records": [SAMPLE_RECORD for _ in range(batch_size)]}
    latencies: list[float] = []

    start = time.perf_counter()
    for _ in range(n_requests):
        req_start = time.perf_counter()
        response = await client.post("/predict-batch", json=payload, headers=headers)
        response.raise_for_status()
        latencies.append((time.perf_counter() - req_start) * 1000.0)
    total_s = time.perf_counter() - start

    total_records = batch_size * n_requests
    return {
        "batch_size": float(batch_size),
        "batch_avg_ms": float(statistics.mean(latencies)) if latencies else 0.0,
        "batch_p95_ms": _percentile(latencies, 0.95),
        "batch_throughput_records_per_s": float(total_records / total_s) if total_s > 0 else 0.0,
    }


async def run_in_process(
    n_single: int,
    n_batch: int,
    batch_size: int,
    api_key: str,
    request_log_sample_rate: float,
) -> tuple[dict[str, float], list[float]]:
    os.environ["REQUEST_LOG_SAMPLE_RATE"] = str(request_log_sample_rate)

    try:
        import psutil
    except Exception:  # pragma: no cover - optional at runtime
        psutil = None

    process = psutil.Process() if psutil else None
    rss_before_mb = (
        float(process.memory_info().rss / (1024 * 1024))
        if process is not None
        else 0.0
    )

    from app.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://in-process") as client:
        single_stats, single_latencies = await _run_single_predict_benchmark(
            client=client,
            n_requests=n_single,
            api_key=api_key,
        )
        batch_stats = await _run_batch_benchmark(
            client=client,
            batch_size=batch_size,
            n_requests=n_batch,
            api_key=api_key,
        )

    rss_after_mb = (
        float(process.memory_info().rss / (1024 * 1024))
        if process is not None
        else 0.0
    )
    summary = {
        **single_stats,
        **batch_stats,
        "memory_rss_before_mb": rss_before_mb,
        "memory_rss_after_mb": rss_after_mb,
        "memory_rss_delta_mb": rss_after_mb - rss_before_mb,
    }
    return summary, single_latencies


async def run_remote(
    base_url: str,
    n_single: int,
    n_batch: int,
    batch_size: int,
    api_key: str,
) -> tuple[dict[str, float], list[float]]:
    async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
        single_stats, single_latencies = await _run_single_predict_benchmark(
            client=client,
            n_requests=n_single,
            api_key=api_key,
        )
        batch_stats = await _run_batch_benchmark(
            client=client,
            batch_size=batch_size,
            n_requests=n_batch,
            api_key=api_key,
        )

    summary = {**single_stats, **batch_stats}
    return summary, single_latencies


def save_outputs(summary: dict[str, float], single_latencies: list[float]) -> None:
    PERFORMANCE_DIR.mkdir(parents=True, exist_ok=True)

    json_path = PERFORMANCE_DIR / "api_performance_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(single_latencies, columns=["single_predict_latency_ms"]).to_csv(
        PERFORMANCE_DIR / "single_predict_latencies.csv",
        index=False,
    )

    plt.figure(figsize=(10, 5))
    plt.hist(single_latencies, bins=20)
    plt.title("Single Prediction Latency Distribution")
    plt.xlabel("Latency (ms)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(PERFORMANCE_DIR / "single_latency_histogram.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="API benchmark utility.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--n-single", type=int, default=30)
    parser.add_argument("--n-batch", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--request-log-sample-rate",
        type=float,
        default=0.05,
        help="In-process mode only: override request log sampling to reduce benchmark noise.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Benchmark via ASGITransport without opening network sockets.",
    )
    args = parser.parse_args()

    if args.in_process:
        summary, single_latencies = asyncio.run(
            run_in_process(
                n_single=args.n_single,
                n_batch=args.n_batch,
                batch_size=args.batch_size,
                api_key=args.api_key,
                request_log_sample_rate=args.request_log_sample_rate,
            )
        )
    else:
        summary, single_latencies = asyncio.run(
            run_remote(
                base_url=args.base_url,
                n_single=args.n_single,
                n_batch=args.n_batch,
                batch_size=args.batch_size,
                api_key=args.api_key,
            )
        )

    save_outputs(summary, single_latencies)

    print("Benchmark finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
