"""Instruction tuning utilities with TRL."""

from __future__ import annotations

from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from domain_llm_ft.models.registry import resolve_model_name


def to_instruction_dataset(dataset: Dataset, text_column: str, label_column: str) -> Dataset:
    """Convert classification records to instruction tuning format."""

    def _map(row: dict) -> dict[str, str]:
        prompt = (
            "Classify text into label category.\n"
            f"Text: {row[text_column]}\n"
            "Answer with label only."
        )
        return {
            "prompt": prompt,
            "completion": str(row[label_column]),
            "text": f"{prompt}\nLabel: {row[label_column]}",
        }

    return dataset.map(_map)


def run_instruction_tuning(
    model_name: str,
    train_dataset: Dataset,
    output_dir: Path,
    max_seq_length: int = 512,
) -> Path:
    """Run SFT training for instruction-format data."""
    resolved = resolve_model_name(model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = AutoModelForCausalLM.from_pretrained(resolved)

    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
        bf16=True,
        dataset_text_field="text",
        max_length=max_seq_length,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=args,
    )
    trainer.train()
    trainer.save_model(output_dir)
    return output_dir
