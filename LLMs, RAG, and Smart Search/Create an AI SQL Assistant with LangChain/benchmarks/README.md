# Benchmark Dataset

## Files
- `benchmark_cases.json`: 100 business QA pairs (question + ground-truth SQL).
- `benchmark_cases_sample6.json`: 6-case smoke subset for faster iteration.
- `benchmark_cases_sample1.json`: 1-case end-to-end CI/smoke subset.

## Categories Covered
- trend analysis
- top-N ranking
- segment/category breakdowns
- window functions and CTEs
- inventory analytics
- employee performance KPIs
- comparative month-over-month analysis

## Run
```bash
uv run ai-sql-assistant benchmark-run --cases-file benchmarks/benchmark_cases.json
```

For faster local iteration:
```bash
uv run ai-sql-assistant benchmark-run --cases-file benchmarks/benchmark_cases_sample6.json
```
