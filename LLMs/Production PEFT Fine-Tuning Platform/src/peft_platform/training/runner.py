"""Training orchestration."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from peft_platform.peft.registry import PeftMethod
from peft_platform.utils.io import ensure_dir, write_json


@dataclass(slots=True)
class TrainingResult:
    method: str
    model_id: str
    loss: float
    steps: int
    duration_sec: float
    output_dir: str


@dataclass(slots=True)
class LiveTrainingResult:
    model_id: str
    adapter_dir: str
    train_runtime_sec: float
    train_steps: int
    train_loss: float
    generated_text: str


class TrainingRunner:
    """Run train workflows with smoke fallback for resource-constrained environments."""

    def __init__(self, artifacts_root: Path) -> None:
        self.artifacts_root = artifacts_root
        ensure_dir(artifacts_root)

    def run_smoke(self, model_id: str, method: PeftMethod, steps: int = 5) -> TrainingResult:
        start = perf_counter()
        random.seed(42)
        loss = 2.0
        for _ in range(steps):
            loss *= 0.95 + random.random() * 0.02

        run_dir = ensure_dir(self.artifacts_root / f"{model_id.replace('/', '_')}_{method.value}_smoke")
        result = TrainingResult(
            method=method.value,
            model_id=model_id,
            loss=round(loss, 4),
            steps=steps,
            duration_sec=round(perf_counter() - start, 3),
            output_dir=str(run_dir),
        )
        write_json(run_dir / "result.json", asdict(result))
        return result

    def run_live_lora(
        self,
        model_id: str,
        output_name: str,
        max_steps: int = 10,
        learning_rate: float = 2e-4,
        per_device_train_batch_size: int = 2,
    ) -> LiveTrainingResult:
        """Run real LoRA fine-tuning on a tiny instruction dataset.

        This path performs actual model loading, tokenization, gradient updates,
        adapter checkpoint save, and generation.
        """
        try:
            import torch
            from datasets import Dataset
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )
        except Exception as exc:  # pragma: no cover - import gate for runtime
            raise RuntimeError("Required training libraries are not available") from exc

        run_dir = ensure_dir(self.artifacts_root / output_name)
        adapter_dir = ensure_dir(run_dir / "adapter")

        examples: list[dict[str, str]] = [
            {"text": "### Instruction: Summarize PEFT.\n### Response: PEFT tunes few parameters and keeps base model frozen."},
            {"text": "### Instruction: Explain LoRA.\n### Response: LoRA learns low-rank update matrices for transformer layers."},
            {"text": "### Instruction: Define QLoRA.\n### Response: QLoRA combines low-rank adapters with 4-bit quantized base weights."},
            {"text": "### Instruction: What is instruction tuning?\n### Response: Supervised tuning on prompt-response pairs."},
            {"text": "### Instruction: Why use adapters?\n### Response: Adapters reduce memory and checkpoint size while preserving quality."},
            {"text": "### Instruction: Give one PEFT benefit.\n### Response: Faster experimentation with multiple domain-specific adapters."},
            {"text": "### Instruction: Describe IA3.\n### Response: IA3 scales key activations using learned vectors."},
            {"text": "### Instruction: One line about adapter fusion.\n### Response: Adapter fusion combines knowledge from multiple adapters."},
            {"text": "### Instruction: Summarize gradient checkpointing.\n### Response: It lowers memory use by recomputing activations."},
            {"text": "### Instruction: What is model merging?\n### Response: Merging applies adapter deltas into base weights for standalone deployment."},
        ]

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = Dataset.from_list(examples)

        def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
            encoded = tokenizer(
                batch["text"],
                truncation=True,
                max_length=128,
                padding="max_length",
            )
            encoded["labels"] = encoded["input_ids"].copy()
            return encoded

        tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])

        base_model = AutoModelForCausalLM.from_pretrained(model_id)
        base_model.config.use_cache = False

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules="all-linear",
        )
        model = get_peft_model(base_model, peft_config)

        training_args = TrainingArguments(
            output_dir=str(run_dir / "trainer"),
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            num_train_epochs=1,
            max_steps=max_steps,
            logging_steps=1,
            save_steps=max_steps,
            save_total_limit=1,
            report_to=[],
            fp16=False,
            bf16=False,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        )

        train_out = trainer.train()
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        prompt = "### Instruction: Explain LoRA briefly.\n### Response:"
        encoded_prompt = tokenizer(prompt, return_tensors="pt")
        model_device = next(model.parameters()).device
        encoded_prompt = {k: v.to(model_device) for k, v in encoded_prompt.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded_prompt,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        generated_text = tokenizer.decode(generated[0].detach().cpu(), skip_special_tokens=True)

        log_history = trainer.state.log_history
        train_loss = 0.0
        for row in reversed(log_history):
            if "loss" in row:
                train_loss = float(row["loss"])
                break

        result = LiveTrainingResult(
            model_id=model_id,
            adapter_dir=str(adapter_dir),
            train_runtime_sec=float(train_out.metrics.get("train_runtime", 0.0)),
            train_steps=int(trainer.state.global_step),
            train_loss=train_loss,
            generated_text=generated_text,
        )
        write_json(run_dir / "live_training_result.json", asdict(result))
        return result
