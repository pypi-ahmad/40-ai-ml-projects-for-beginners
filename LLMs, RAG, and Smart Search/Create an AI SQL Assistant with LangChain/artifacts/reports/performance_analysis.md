# Performance Analysis

Source benchmark file: `benchmark_run_20260627_111555.json`

## Core Metrics

| Approach | Model | Gen Latency (ms) | Exec Latency (ms) | Rows | Tokens | Complexity | Memory (MB) | Throughput (QPS) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| direct | granite4.1:3b | 1572.44 | 58.48 | 12.0 | 45.0 | 0.29 | 235.40 | 0.613 |
| langchain | granite4.1:3b | 8935.64 | 20.86 | 0.0 | 28.0 | 0.31 | 235.28 | 0.112 |
| direct | qwen3.5:4b | 28149.92 | 104.54 | 12.0 | 45.0 | 0.29 | 235.40 | 0.035 |
| langchain | qwen3.5:4b | 13611.79 | 55.72 | 12.0 | 45.0 | 0.29 | 235.28 | 0.073 |

## Judge Metrics

| Approach | Model | SQL | Business | Completeness | Readability | Efficiency | Safety |
|---|---|---:|---:|---:|---:|---:|---:|
| direct | granite4.1:3b | 1.00 | 1.00 | 1.00 | 0.90 | 0.95 | 1.00 |
| langchain | granite4.1:3b | 0.70 | 0.60 | 1.00 | 0.80 | 0.90 | 1.00 |
| direct | qwen3.5:4b | 1.00 | 1.00 | 1.00 | 0.90 | 0.95 | 1.00 |
| langchain | qwen3.5:4b | 1.00 | 1.00 | 1.00 | 0.90 | 0.95 | 1.00 |

## Generated Plots
- `/home/ahmad/AI/Github/40 AI-ML Projects for Beginners/LLMs, RAG, and Smart Search/Create an AI SQL Assistant with LangChain/artifacts/plots/benchmark_generation_latency.png`
- `/home/ahmad/AI/Github/40 AI-ML Projects for Beginners/LLMs, RAG, and Smart Search/Create an AI SQL Assistant with LangChain/artifacts/plots/benchmark_throughput.png`