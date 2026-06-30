"""Accelerate-based training engine for custom loops."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler

from domain_llm_ft.config.schemas import ExperimentConfig
from domain_llm_ft.models.registry import load_sequence_classifier
from domain_llm_ft.peft.adapters import apply_peft


@dataclass
class AccelerateRunResult:
    train_loss: float


class AccelerateEngine:
    """Simple Accelerate training loop for sequence classification."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def run(self, tokenized, data_collator) -> AccelerateRunResult:
        """Run loop with mixed precision and gradient accumulation."""
        accelerator = Accelerator(
            mixed_precision="bf16" if self.config.training.bf16 else ("fp16" if self.config.training.fp16 else "no")
        )
        model = load_sequence_classifier(
            self.config.model.name,
            num_labels=self.config.model.num_labels,
            id2label=self.config.model.id2label,
            label2id=self.config.model.label2id,
        )
        model = apply_peft(model, self.config.peft)

        train_loader = DataLoader(
            tokenized["train"],
            shuffle=True,
            batch_size=self.config.training.train_batch_size,
            collate_fn=data_collator,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        num_steps = self.config.training.epochs * max(len(train_loader), 1)
        scheduler = get_scheduler(
            self.config.training.scheduler,
            optimizer=optimizer,
            num_warmup_steps=int(self.config.training.warmup_ratio * num_steps),
            num_training_steps=num_steps,
        )

        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model,
            optimizer,
            train_loader,
            scheduler,
        )

        model.train()
        total_loss = 0.0
        step_count = 0

        for _epoch in range(self.config.training.epochs):
            for step, batch in enumerate(train_loader):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                total_loss += float(loss.detach().item())
                step_count = step + 1

        return AccelerateRunResult(train_loss=total_loss / max(step_count, 1))
