# Monitoring Guide

## What to Track (Streamlit Required)
- **Availability:** App is reachable and pages render.
- **Latency:** Cold-start and warm inference latency trends.
- **Errors:** Validation and runtime exceptions per feature flow.
- **Fallback rate:** Percent of requests served by HF, Ollama, and rule-based fallbacks.
- **Resource health:** Memory usage growth and CPU spikes.

## Optional API Track Signals
- `/health` readiness and `/metrics` request/latency counters.
- Batch-vs-single prediction throughput and error rates.

## Where to Observe
- Streamlit Community Cloud logs.
- Sidebar inference status (`method`, `latency`).
- Benchmark artifacts in `outputs/metrics/runtime_benchmark.json`.
- Optional API benchmark artifact in `outputs/metrics/fastapi_runtime_benchmark.json`.

## Suggested Operating Cadence
- **Daily:** check app health, errors, and deployment status.
- **Weekly:** review latency/fallback trends and user feedback.
- **Monthly:** dependency refresh + regression validation.

## Starter Alert Thresholds
- Error rate > 5% over 15 minutes.
- Median inference latency > 3 seconds.
- Fallback-only mode sustained > 1 hour.
- Memory usage consistently near platform limit.
