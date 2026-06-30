# Architecture

## Core Layers
- `data`: dataset loading, preprocessing, profiling, and version manifests.
- `tokenization`: tokenizer config, dynamic padding, sliding windows.
- `models`: registry + model/PEFT loading.
- `training`: Trainer orchestration, metrics, Optuna tuning.
- `evaluation`: error analysis, calibration, embedding analysis, robustness.
- `optimization`: ONNX export + runtime benchmarking.
- `serving`: async inference, FastAPI endpoints, monitoring metrics.
- `ui`: Streamlit dashboard for inference and benchmark inspection.
- `benchmarking` and `reporting`: matrix runs and report generation.

## Execution Modes
- `quick`: smoke/validation path.
- `full`: overnight benchmark path.

## Artifact Flow
`datasets -> preprocessing -> tokenization -> train/eval -> mlflow/artifacts -> onnx/api/ui -> reports`
