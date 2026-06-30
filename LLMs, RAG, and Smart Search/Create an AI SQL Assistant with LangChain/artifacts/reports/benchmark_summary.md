# Benchmark Summary

Source report: `benchmark_run_20260627_111555.json`
Run id: `21ed0e71-029d-4c85-adbe-0a55255b2faf`
Total cases: **1**
Total runs: **4**

## Runtime Metrics

| Model | Approach | Exec Accuracy | Result Correct | Gen Latency (ms) | Exec Latency (ms) | Rows | Tokens | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| granite4.1:3b | direct | 1.00 | 0.00 | 1572.44 | 58.48 | 12.0 | 45.0 | 0.613 |
| granite4.1:3b | langchain | 1.00 | 0.00 | 8935.64 | 20.86 | 0.0 | 28.0 | 0.112 |
| qwen3.5:4b | direct | 1.00 | 0.00 | 28149.92 | 104.54 | 12.0 | 45.0 | 0.035 |
| qwen3.5:4b | langchain | 1.00 | 0.00 | 13611.79 | 55.72 | 12.0 | 45.0 | 0.073 |

## LLM Judge Summary

| Model | Approach | SQL | Business | Completeness | Readability | Efficiency | Safety |
|---|---:|---:|---:|---:|---:|---:|---:|
| granite4.1:3b | direct | 1.00 | 1.00 | 1.00 | 0.90 | 0.95 | 1.00 |
| granite4.1:3b | langchain | 0.70 | 0.60 | 1.00 | 0.80 | 0.90 | 1.00 |
| qwen3.5:4b | direct | 1.00 | 1.00 | 1.00 | 0.90 | 0.95 | 1.00 |
| qwen3.5:4b | langchain | 1.00 | 1.00 | 1.00 | 0.90 | 0.95 | 1.00 |

## Notes
- Session run used 1-case matrix for runtime feasibility during interactive development.
- Full 100-case matrix supported via benchmark command with `benchmarks/benchmark_cases.json`.