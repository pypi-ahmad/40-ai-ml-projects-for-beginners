# Project #21 - Production-Grade Text Classification Framework

Production text classification framework built with Hugging Face Transformers, PEFT/LoRA, Optuna, ONNX Runtime, MLflow, FastAPI, and Streamlit.

## Highlights
- Multi-dataset support: `SetFit/20_newsgroups`, `ag_news`, `imdb`
- Multi-model benchmark matrix: DistilBERT, BERT Base, RoBERTa Base, DeBERTa-v3 Base, ModernBERT (auto-skip fallback)
- Fine-tuning modes: full fine-tuning, LoRA, QLoRA fallback path
- Comprehensive evaluation: macro/weighted metrics, error analysis, calibration, robustness checks
- Explainability: LIME, SHAP, attention token importance
- Optimization: ONNX export + dynamic quantization utilities
- Production serving: FastAPI with async single/batch inference + monitoring endpoints
- Product UI: Streamlit dashboard for single and batch predictions, benchmark and evaluation views
- Experiment tracking: MLflow parameters/metrics/artifacts

## Architecture
See [Architecture](docs/architecture.md).

## Repository Structure
```text
src/textclf_framework/
  data/ tokenization/ models/ training/ evaluation/
  explainability/ optimization/ benchmarking/
  serving/ ui/ reporting/
configs/
notebooks/
tests/
apps/
```

## Setup
```bash
uv venv .venv
source .venv/bin/activate
uv sync --extra dev
```

## Quickstart
```bash
uv run textclf profile-data --dataset setfit_20_newsgroups --config-path configs/quick.yaml
uv run textclf train-run --dataset setfit_20_newsgroups --model distilbert --strategy full --config-path configs/quick.yaml
uv run textclf benchmark-run --config-path configs/quick.yaml
uv run textclf build-report --config-path configs/quick.yaml
```

## Full Pipeline
```bash
uv run textclf benchmark-run --config-path configs/default.yaml
uv run textclf build-report --config-path configs/default.yaml
```

## API
```bash
uv run uvicorn apps.fastapi_app:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `GET /metrics`
- `POST /predict`
- `POST /predict/batch`

Example request:
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"The product quality is excellent and shipping was fast.","top_k":3}'
```

## Streamlit UI
```bash
uv run streamlit run apps/streamlit_app.py
```

Pages:
- Single Prediction
- Batch Prediction
- Explainability
- Evaluation Dashboard
- Benchmark Results

## Testing
```bash
uv run pytest -v
```

## Notebook
- Zero-to-hero notebook: [notebooks/project21_zero_to_hero.ipynb](notebooks/project21_zero_to_hero.ipynb)
- Supports quick and full execution modes.

## MLflow Tracking
Default backend is local SQLite:
- URI: `sqlite:///mlflow.db`
- Experiment name: `project21_textclf`

Launch UI:
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## ONNX Export and Optimization
Use module utilities in `textclf_framework/optimization/onnx_utils.py` to:
- export Transformer classifier to ONNX
- run ONNX Runtime session
- apply dynamic quantization
- benchmark latency

## Monitoring Signals
`/metrics` provides:
- request count
- error rate
- mean/p95 latency
- throughput
- mean input length
- average prediction confidence

## Benchmarks and Reports
Generated under `reports/`:
- `benchmark_matrix.csv`
- `benchmark_summary.md`
- `benchmark_macro_f1.html`

## Known Runtime Constraints
- CUDA availability depends on local GPU driver/toolkit setup.
- ModernBERT may be skipped automatically when unsupported and logged as such.
- Full benchmark matrix is intended for overnight runs.

## Future Improvements
- Multi-label and zero-shot extensions
- Distillation and ensemble support
- Distributed training and energy profiling
- richer Streamlit explainability visualizations

## References
- Hugging Face Transformers and Datasets
- PEFT (LoRA/QLoRA)
- Optuna
- ONNX Runtime
- MLflow
- FastAPI
- Streamlit
