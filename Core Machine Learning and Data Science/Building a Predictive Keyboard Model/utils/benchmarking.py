"""Benchmark helpers for latency, throughput, and memory usage."""

from __future__ import annotations

import time

import psutil
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def benchmark_model_inference(
    model: torch.nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: str = "cpu",
    max_batches: int = 50,
    warmup_batches: int = 3,
) -> dict[str, float]:
    """Benchmark average inference latency and throughput.

    Args:
        model: Model to benchmark.
        dataloader: Batches for timing.
        device: Execution device.
        max_batches: Upper bound on batches used for benchmark.

    Returns:
        Dict with latency/throughput and memory footprint.
    """

    process = psutil.Process()
    before_mem_mb = process.memory_info().rss / (1024**2)

    model.eval()
    total_examples = 0
    latencies: list[float] = []

    for batch_idx, (contexts, _) in enumerate(dataloader):
        if batch_idx >= max_batches + warmup_batches:
            break

        contexts = contexts.to(device)

        if batch_idx < warmup_batches:
            _ = model(contexts)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            continue

        start = time.perf_counter()
        _ = model(contexts)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append(end - start)
        total_examples += contexts.size(0)

    after_mem_mb = process.memory_info().rss / (1024**2)

    total_time = sum(latencies)
    avg_latency_ms = (total_time / max(len(latencies), 1)) * 1_000
    throughput = total_examples / max(total_time, 1e-9)

    return {
        "avg_latency_ms": float(avg_latency_ms),
        "throughput_examples_per_sec": float(throughput),
        "memory_rss_mb_before": float(before_mem_mb),
        "memory_rss_mb_after": float(after_mem_mb),
        "memory_delta_mb": float(after_mem_mb - before_mem_mb),
        "timed_batches": float(len(latencies)),
    }
