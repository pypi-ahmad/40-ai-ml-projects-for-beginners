# Project #22: Production-Grade Open-Source LLM Fine-Tuning Framework

Production-ready, modular framework for fine-tuning open-source LLMs with PEFT and modern LLMOps.

Stack focus: **Unsloth, TRL, PEFT, Transformers, Datasets, BitsAndBytes, MLflow, FastAPI, Streamlit, Ollama, vLLM**.

## 1) Architecture

### System Design
- Config-first pipeline (`configs/project.yaml`) with reproducible runs.
- Module-per-capability architecture under `src/llmft/`.
- End-to-end orchestration through `ProjectRunner`.
- Optional heavy dependencies with safe fallback behavior for limited environments.

### Architecture Diagram
See [docs/architecture.md](docs/architecture.md).

## 2) Training Pipeline

`Dataset Build -> Model Resolve -> SFT/PEFT Train -> HPO -> Evaluate -> Benchmark -> Export -> Serve/UI`

### Supported Training Modes
- SFT via TRL contract.
- LoRA, QLoRA, RSLoRA (DoRA-ready extension point).
- Full fine-tuning path guard for smaller models.
- Checkpointing and resume-ready metadata.

### Advanced Alignment (v1 contract)
- DPO / ORPO / GRPO stubs implemented in `src/llmft/training/alignment.py`.
- Emits auditable artifacts and integration contract for full rollout.

## 3) Dataset Strategy and Pipeline

### Executed Benchmark Dataset Mix (default)
- `alpaca_cleaned` (general instruction)
- `codealpaca` (code generation)
- `medical_qa` (domain adaptation)

### Pipeline Capabilities
- Dataset ingestion hooks (download/stream-ready contract).
- Filtering and deduplication.
- Prompt templating and normalized formatting.
- Token and length statistics.
- Train/validation split.
- Cache/version manifest artifacts.

## 4) Prompt Templates

Built-in template registry (`src/llmft/templates/registry.py`):
- Alpaca
- ChatML
- Llama 3
- Mistral
- Qwen
- Phi
- Gemma
- Custom

## 5) Model Registry and Fallback Policy

Required family aliases included:
- `llama3_8b`
- `qwen3_8b`
- `mistral_7b`
- `gemma3`
- `phi4_mini`
- `granite41`

Policy:
- Resolve alias to primary model ID.
- If unavailable and fallback enabled, auto-select configured fallback.
- Persist fallback reason in artifacts for auditability.

## 6) Quantization and Export

### Quantization Modes (config-enabled)
- 4-bit
- 8-bit
- FP16
- BF16

### Export Targets
- Adapter safetensors
- Merged model manifest
- GGUF placeholder bundle
- Ollama `Modelfile`
- ONNX optional output flag

## 7) Evaluation and Benchmarking

### Metrics
- BLEU (fallback implementation)
- ROUGE-L (fallback implementation)
- Exact Match
- Loss / Perplexity proxy
- Latency benchmarking

### Judge Scoring
- LLM-as-judge contract fields:
  - correctness
  - helpfulness
  - faithfulness
  - coherence
  - instruction_following
  - grounding
  - safety

### Backend Benchmarks
Mandatory benchmark path executes all:
- Transformers backend
- vLLM backend
- Ollama backend

Outputs stored in `artifacts/benchmarks/`.

## 8) Inference and Deployment

### Inference Runtimes
- Transformers local backend.
- vLLM OpenAI-compatible HTTP backend.
- Ollama generate API backend.
- Async batch generation and latency benchmarking.

### FastAPI
`src/llmft/serving/api.py` provides:
- `/health`
- `/chat`
- `/batch`
- `/bench`
- `/stream` (placeholder stream response contract)

### Streamlit
`src/llmft/ui/app.py` pages:
- Chat
- Model Selector
- Dataset Explorer
- Training Dashboard
- Evaluation
- Benchmarks
- Adapter Manager
- Inference Settings

## 9) MLflow and Monitoring

### Tracking
- `MLflowTracker` logs params, metrics, and artifacts.
- If MLflow unavailable, local JSON fallback written under `artifacts/mlflow/`.

### Monitoring
- Runtime telemetry captures CPU load and GPU utilization (when `nvidia-smi` is available).
- Snapshot artifact path: `artifacts/reports/runtime_snapshot.json`.

## 10) Installation

### Base
```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv sync --extra dev
```

### Full stack
```bash
uv sync --extra train --extra serve --extra ui --extra http
```

## 11) Configuration

Primary config file: [configs/project.yaml](configs/project.yaml)

Key sections:
- `runtime`
- `data`
- `model`
- `train`
- `hpo`
- `evaluation`
- `inference`
- `export`
- `serve`
- `ui`
- `safety`

## 12) CLI Usage

```bash
PYTHONPATH=src python3 -m llmft.cli --config configs/project.yaml env validate
PYTHONPATH=src python3 -m llmft.cli --config configs/project.yaml data build
PYTHONPATH=src python3 -m llmft.cli --config configs/project.yaml train sft --dry-run
PYTHONPATH=src python3 -m llmft.cli --config configs/project.yaml eval run
PYTHONPATH=src python3 -m llmft.cli --config configs/project.yaml bench run
PYTHONPATH=src python3 -m llmft.cli --config configs/project.yaml infer run --backend transformers --prompt "Explain QLoRA"
PYTHONPATH=src python3 -m llmft.cli --config configs/project.yaml export run
```

Or scripted:
```bash
bash scripts/run_end_to_end.sh configs/project.yaml
```

## 13) Notebook

Zero-to-hero notebook:
- [notebooks/zero_to_hero_llm_finetuning.ipynb](notebooks/zero_to_hero_llm_finetuning.ipynb)

Covers:
- LLM/Transformer foundations
- Instruction tuning
- LoRA/QLoRA/PEFT concepts
- Dataset pipeline
- Training/Evaluation/Benchmarking/Export/Deployment walkthrough

## 14) Screenshots and Artifacts

Place generated evidence in:
- `artifacts/screenshots/` (MLflow, FastAPI, Streamlit, benchmark UI)
- `artifacts/reports/` (env, training, inference, telemetry)
- `artifacts/benchmarks/` (latency tables)
- `artifacts/exports/` (adapter/export outputs)

## 15) Tests

```bash
PYTHONPATH=src python3 -m pytest
```

Current test coverage includes:
- Dataset pipeline
- Prompt templates
- Model fallback resolution
- Inference backend fallback behavior
- Export manager
- Security checks
- Runner dry-run end-to-end flow
- CLI smoke path

## 16) Future Improvements
- Full TRL + Unsloth training integration for production GPU runs.
- Richer eval metrics integration (`evaluate`, BERTScore, pass@k harness).
- True streaming SSE/WebSocket in API.
- Real GGUF/ONNX export toolchain wiring.
- Distributed training (FSDP/DeepSpeed).
- Vision-language and tool-calling fine-tuning extensions.

## 17) References
- Unsloth docs and repository.
- TRL and PEFT documentation.
- Hugging Face Transformers/Datasets.
- Ollama API docs.
- vLLM docs.
- MLflow tracking documentation.
