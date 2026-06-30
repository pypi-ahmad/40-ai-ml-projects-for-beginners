# Project #23: Production-Grade Domain LLM Fine-Tuning Framework

Production-ready, modular framework for domain adaptation of transformer models across classification tasks. Built with Hugging Face Transformers + Datasets + PEFT + Accelerate + MLflow + FastAPI + Streamlit.

## Architecture

```text
src/domain_llm_ft/
  config/          # YAML schema + loader
  data/            # loaders, cleaning, dedupe, splits, EDA
  tokenization/    # tokenizer pipeline + collators
  models/          # model registry + task adapters (zero/few/hierarchical)
  training/        # Trainer + Accelerate + instruction tuning
  peft/            # LoRA/QLoRA adapter workflows
  hpo/             # Optuna integration
  evaluation/      # metrics + plots + reports
  explainability/  # attention + token importance + method checks
  error_analysis/  # misclassification + confidence + confusion artifacts
  benchmark/       # latency/throughput/memory matrix + charts
  compression/     # ONNX/TorchScript/Safetensors/quantization
  inference/       # sync/async/batch torch + ONNX runtime
  serving/         # FastAPI app and contracts
  ui/              # Streamlit multipage dashboard
  monitoring/      # CPU/RAM/GPU snapshots
  utils/           # logging + I/O
```

## Supported Dataset Inputs

- HF Hub datasets (`ag_news`, `dbpedia_14`, `imdb`, `sst2`, `emotion`, `go_emotions`, `tweet_eval`, `financial_phrasebank`, `trec`)
- Local `CSV`, `JSONL`, `Parquet`
- Config-driven split strategy with deduplication, filtering, and sample caps
- Gated adapter paths for restricted datasets (`mimic_notes`, `legalbench`)

## Supported Tasks

- Binary classification
- Multi-class classification
- Multi-label classification (pipeline-ready)
- Hierarchical classification
- Zero-shot classification
- Few-shot classification
- Instruction tuning conversion and SFT workflow

## Model Matrix

Required:
- DistilBERT
- BERT Base
- RoBERTa Base
- DeBERTa-v3
- ModernBERT
- MiniLM
- E5

Practical optional:
- Qwen
- Gemma
- Phi
- TinyLlama

## Setup

```bash
uv venv .venv
source .venv/bin/activate
uv sync
# optional explainability extras (SHAP/LIME)
uv sync --extra xai
```

## Core Commands

```bash
# Data + EDA
uv run domain-llm prepare-data --config-path configs/baseline.yaml
# Offline local smoke dataset
uv run domain-llm prepare-data --config-path configs/local_csv.yaml

# Full training pipeline
uv run domain-llm train --config-path configs/baseline.yaml

# Hyperparameter tuning
uv run domain-llm tune --config-path configs/baseline.yaml

# Benchmark suite
uv run domain-llm benchmark --config-path configs/baseline.yaml

# Export artifacts
uv run domain-llm export --config-path configs/baseline.yaml

# API + UI
uv run domain-llm serve-api --host 0.0.0.0 --port 8000
uv run domain-llm serve-ui --port 8501
```

## FastAPI Endpoints

- `GET /health`
- `POST /predict`
- `POST /classify`
- `POST /batch`
- `GET /docs`

## Streamlit Pages

- Single Prediction
- Batch Prediction
- Dataset Explorer
- Training Dashboard
- Benchmark Dashboard
- Model Comparison
- Confusion Matrix
- Error Analysis
- Model Manager

## MLflow Tracking

Tracked artifacts include:
- Params and metrics
- Checkpoints and model exports
- Evaluation plots (confusion, ROC, PR, calibration)
- Benchmark CSV/charts
- Error analysis reports

## Docker

```bash
docker compose up --build
```

Services:
- API: `http://localhost:8000/docs`
- Streamlit: `http://localhost:8501`

## Educational Notebook

Notebook: `notebooks/01_zero_to_hero.ipynb`

Covers:
- Transformers and tokenization
- Fine-tuning workflow
- PEFT basics
- Metrics and model analysis
- Inference and deployment flow

## Testing

```bash
uv sync --extra dev
uv run pytest -q
```

If running in restricted sandboxes, set writable matplotlib cache:

```bash
export MPLCONFIGDIR=/tmp/mpl
```

## Outputs

Generated under `artifacts/`:
- `reports/`
- `figures/`
- `screenshots/`
- `exports/`
- `checkpoints/`

## Future Work

- Full multi-label loss strategies and threshold optimization
- FSDP/DeepSpeed distributed paths
- Distillation and active learning loops
- OpenVINO export path
