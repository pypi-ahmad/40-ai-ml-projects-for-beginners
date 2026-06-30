# Benchmark Evidence

- Profile: **full**
- Fairness policy: same train/val split, same target, fixed random seed (`42`).

## Tool Status
- LazyPredict: ok (42 rows)
- FLAML: skipped (0 rows) — Import failed: No module named 'flaml'
- PyCaret: skipped (0 rows) — Import failed: No module named 'pycaret'

## Interpretation
- Manual model ranking is always generated and drives model selection.
- AutoML tools complement analysis, not replace split discipline or metric interpretation.
