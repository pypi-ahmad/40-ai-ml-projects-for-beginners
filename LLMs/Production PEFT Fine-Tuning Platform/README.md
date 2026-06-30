# Production-Grade Parameter-Efficient Fine-Tuning (PEFT) Platform

Production-ready, modular framework for efficient LLM adaptation using LoRA family methods, quantization, experiment tracking, benchmarking, and deployment surfaces (CLI, FastAPI, Streamlit).

## 1) Architecture

High-level flow:

1. Dataset ingestion and normalization (HF/local CSV/JSONL)
2. Cleaning, validation, deduplication, split, prompt templating
3. Model + PEFT method + quantization profile selection
4. Training orchestration (TRL / Accelerate / Unsloth preferred path)
5. Evaluation and benchmark generation
6. MLflow artifact and metric tracking
7. Adapter/model export and serving via API/UI/CLI

See: `docs/architecture.md`

## 2) Supported Models

### Required (configured)
- Llama 3 (smoke-tier config)
- Qwen 3 (`Qwen/Qwen3-1.7B-Instruct` deep-tier)
- Gemma 3 (smoke-tier config)
- Phi-4 Mini (smoke-tier config)
- Mistral (smoke-tier config)
- TinyLlama (`TinyLlama/TinyLlama-1.1B-Chat-v1.0` deep-tier)
- SmolLM (`HuggingFaceTB/SmolLM2-1.7B-Instruct` deep-tier)
- ModernBERT (smoke-tier config)

### Optional (registry-ready extension path)
- DeepSeek and other optional families can be plugged via model registry.

## 3) Supported PEFT Methods

Implemented in strict registry with native-builder path:
- LoRA
- QLoRA
- AdaLoRA
- LoHa
- LoKr
- IA3
- Prefix Tuning
- Prompt Tuning
- P-Tuning v2
- Adapter Fusion
- Full Fine-Tuning baseline

## 4) Quantization Support

- 4-bit / 8-bit profiles
- NF4 / FP4 profiles
- GPTQ export path
- GGUF export path
- ONNX export placeholder path
- AWQ marked optional (extension point)

## 5) Dataset Strategy + Processing

Config-driven datasets include:
- Alpaca
- SAMSum
- SQuAD
- Financial PhraseBank

Pipeline features:
- Schema normalization to canonical `Sample`
- Cleaning and trimming
- Deduplication
- Split (train/validation/test)
- Prompt template application: ChatML, Alpaca, Llama, Qwen, Gemma
- Dataset stats generation

## 6) Training Framework

Modules under `src/peft_platform/training` and `src/peft_platform/pipeline.py` provide:
- Config-driven method selection
- Smoke training runner with artifact emission
- Run manifest capture (runtime, metrics, artifacts)
- Unsloth-preferred policy hook (with fallback path)
- Resume/checkpoint extension points

## 7) Hyperparameter Optimization

`src/peft_platform/training/hpo.py`
- Optuna integration wrapper
- Best trial capture and structured result object

## 8) Evaluation + Explainability

Evaluation modules provide:
- Classification metrics: accuracy, weighted F1, exact match
- Text metric wrapper with graceful fallback for ROUGE/BLEU/BERTScore
- Markdown report generation

Explainability module includes support summary hooks for:
- Attention availability
- Hidden state extraction readiness
- Token-importance workflow placeholders

## 9) Benchmark Suite

`src/peft_platform/benchmarking`:
- Latency avg/p95
- Throughput
- Peak memory (tracemalloc)
- Plot/text report export fallback

## 10) Experiment Tracking (MLflow)

`src/peft_platform/tracking/mlflow_client.py`
- Tracks params, metrics, artifacts
- Graceful no-op mode if MLflow unavailable

## 11) Adapter Management

`src/peft_platform/peft/adapters.py`
- Save/load/remove adapter records
- Merge-record creation
- JSON registry under `artifacts/adapter_registry.json`

## 12) Inference Surfaces

Shared inference engine: `src/peft_platform/inference/engine.py`

Supports:
- Prompt generation
- Model-specific load path
- Latency and token count
- Mock fallback mode for constrained environments

## 13) FastAPI Deployment

`src/peft_platform/api/app.py`

Endpoints:
- `GET /health`
- `GET /models`
- `GET /adapters`
- `POST /generate`
- `POST /generate/stream` (SSE)
- `POST /chat`
- `POST /batch`
- Swagger docs: `/docs`

## 14) Streamlit Dashboard

`src/peft_platform/ui/streamlit_app.py`

Pages:
- Chat
- Instruction Playground
- Adapter Manager
- Training Dashboard
- Benchmark Dashboard
- Dataset Explorer
- Evaluation Dashboard
- Model Comparison
- Generation Playground

## 15) CLI

`src/peft_platform/cli/main.py`

Commands:
- `runtime`
- `models`
- `dataset-smoke`
- `train`
- `infer`
- `benchmark`
- `run-pipeline`
- `adapter-add`
- `adapter-list`

## 16) Educational Notebook

Notebook: `notebooks/peft_zero_to_hero.ipynb`

Includes sections on:
- Why PEFT exists
- LoRA math and low-rank decomposition
- QLoRA and NF4
- Gradient checkpointing
- Instruction tuning and templates
- Adapter merging
- Inference and deployment

## 17) Installation

```bash
uv python install 3.12
uv venv .venv
source .venv/bin/activate
uv sync --extra dev
```

If cache path blocked:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev
```

## 18) Run Workflows

### Test
```bash
uv run pytest -q
```

### Pipeline
```bash
uv run peft-platform run-pipeline
```

### API
```bash
uv run uvicorn peft_platform.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

### Streamlit
```bash
uv run streamlit run src/peft_platform/ui/streamlit_app.py
```

### Smoke script
```bash
bash scripts/run_smoke.sh
```

## 19) Docker Profiles

`docker/docker-compose.yml` includes:
- API service
- Streamlit service
- MLflow server service

## 20) Testing Strategy

- Unit tests: data pipeline, templates, PEFT registry, adapters, inference, config loading
- Integration tests: training smoke, CLI smoke (when typer available)
- API tests: endpoint smoke (when fastapi available)
- UI smoke: streamlit import test

## 21) Current Execution Evidence in This Workspace

Generated artifacts:
- `artifacts/reports/lora_summary.md`
- `artifacts/reports/lora_manifest.json`
- `artifacts/plots/lora_latency.txt`
- `artifacts/checkpoints/.../result.json`
- `artifacts/reports/execution_report.md`

## 22) Limitations (Current Sandbox)

- DNS/network blocked for `uv` dependency download.
- GPU telemetry unavailable in current session.
- Full real-model deep matrix not executable without dependency install + GPU runtime.

## 23) Future Work

- Complete full deep-run matrix with real models and datasets
- Add AWQ, vLLM, DPO/ORPO/GRPO modules
- Add richer explainability visual dashboards
- Add full ONNX and GGUF verified conversion pipelines
- Add CI with matrix profiles (CPU smoke, GPU integration)
