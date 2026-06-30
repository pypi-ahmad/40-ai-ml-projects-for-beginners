"""Benchmark API startup, latency, and throughput for host vs container runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import time
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "benchmarks"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

SAMPLE_INPUT = {
    "MedInc": 8.3,
    "HouseAge": 42.0,
    "AveRooms": 6.9,
    "AveBedrms": 1.0,
    "Population": 230.0,
    "AveOccup": 3.2,
    "Latitude": 37.9,
    "Longitude": -122.2,
}


def parse_args() -> argparse.Namespace:
    """Parse benchmark CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Run label: host or docker")
    parser.add_argument("--base-url", required=True, help="API base URL")
    parser.add_argument("--requests", type=int, default=200, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=20, help="Concurrent request workers")
    return parser.parse_args()


def wait_for_health(base_url: str, timeout_seconds: int = 60) -> float:
    """Wait for `/health` endpoint and return startup-ready latency."""
    start = time.perf_counter()
    deadline = start + timeout_seconds
    while time.perf_counter() < deadline:
        try:
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{base_url.rstrip('/')}/health")
                if response.status_code == 200:
                    return time.perf_counter() - start
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Health check did not succeed in {timeout_seconds}s")


def benchmark_predictions(base_url: str, total_requests: int, concurrency: int) -> dict[str, float]:
    """Issue concurrent `/predict` requests and return latency summary."""
    latencies_ms: list[float] = []
    failures = 0

    def one_request() -> tuple[float, int]:
        start = time.perf_counter()
        with httpx.Client(timeout=10.0) as client:
            response = client.post(f"{base_url.rstrip('/')}/predict", json=SAMPLE_INPUT)
        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, response.status_code

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(one_request) for _ in range(total_requests)]
        for future in as_completed(futures):
            elapsed, status_code = future.result()
            latencies_ms.append(elapsed)
            if status_code != 200:
                failures += 1
    wall_total = time.perf_counter() - wall_start

    return {
        "requests": total_requests,
        "concurrency": concurrency,
        "failures": failures,
        "latency_avg_ms": float(statistics.mean(latencies_ms)),
        "latency_p50_ms": float(statistics.median(latencies_ms)),
        "latency_p95_ms": float(pd.Series(latencies_ms).quantile(0.95)),
        "latency_p99_ms": float(pd.Series(latencies_ms).quantile(0.99)),
        "throughput_rps": float(total_requests / wall_total) if wall_total else 0.0,
    }


def save_outputs(result: dict[str, float], label: str) -> None:
    """Persist benchmark JSON and update comparison chart if both profiles exist."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / f"performance_{label}.json"
    output_file.write_text(json.dumps(result, indent=2))

    host_file = OUTPUT_DIR / "performance_host.json"
    docker_file = OUTPUT_DIR / "performance_docker.json"
    if not (host_file.exists() and docker_file.exists()):
        return

    host = json.loads(host_file.read_text())
    docker = json.loads(docker_file.read_text())
    chart_df = pd.DataFrame(
        [
            {"Mode": "Host", "Startup (s)": host["startup_seconds"], "P95 Latency (ms)": host["latency_p95_ms"]},
            {
                "Mode": "Docker",
                "Startup (s)": docker["startup_seconds"],
                "P95 Latency (ms)": docker["latency_p95_ms"],
            },
        ]
    )
    chart_df.to_csv(OUTPUT_DIR / "host_vs_docker_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(chart_df["Mode"], chart_df["Startup (s)"], color=["#4c72b0", "#dd8452"])
    axes[0].set_title("Startup Time Comparison")
    axes[0].set_ylabel("Seconds")

    axes[1].bar(chart_df["Mode"], chart_df["P95 Latency (ms)"], color=["#4c72b0", "#dd8452"])
    axes[1].set_title("P95 Latency Comparison")
    axes[1].set_ylabel("Milliseconds")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "host-vs-docker-performance.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Run benchmark workflow for one target endpoint."""
    args = parse_args()
    startup_seconds = wait_for_health(args.base_url)
    summary = benchmark_predictions(args.base_url, args.requests, args.concurrency)
    summary["label"] = args.label
    summary["startup_seconds"] = startup_seconds
    save_outputs(summary, args.label)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
