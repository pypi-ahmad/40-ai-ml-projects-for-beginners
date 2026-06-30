# AGENTS.md — Production PEFT Fine-Tuning Platform

## Runtime Contract
- Python: 3.12 managed by `uv`.
- Package manager: `uv` only.
- Default execution: local-first, optional Docker profiles.

## Development Rules
- Preserve modular architecture under `src/peft_platform`.
- All behavior changes require tests.
- Keep model/dataset/PEFT choices config-driven via Hydra.
- Log all train/eval/benchmark runs to MLflow.

## Deep-Run Matrix (v1)
- TinyLlama-1.1B-Chat-v1.0: all PEFT methods + full fine-tuning baseline.
- Qwen3-1.7B-Instruct: LoRA, QLoRA, AdaLoRA, Full FT.
- SmolLM2-1.7B-Instruct: LoRA, QLoRA.
- Datasets: Alpaca, SAMSum, SQuAD, Financial PhraseBank.

## Acceptance Evidence
- Tests green (`pytest`).
- API smoke checks.
- Notebook smoke execution.
- MLflow runs with metrics + artifacts.
