"""Evaluation metrics for predictive keyboard language models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def top_k_accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Compute top-k accuracy from logits and targets."""

    k = max(1, min(k, logits.shape[-1]))
    _, topk = logits.topk(k, dim=-1)
    correct = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
    return float(correct)


@torch.no_grad()
def recall_at_k(logits: torch.Tensor, targets: torch.Tensor, k: int) -> float:
    """Recall@k for single-label next-token task (equivalent to top-k accuracy)."""

    return top_k_accuracy_from_logits(logits, targets, k)


@torch.no_grad()
def mean_reciprocal_rank(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute Mean Reciprocal Rank (MRR) for batched logits."""

    sorted_idx = logits.argsort(dim=-1, descending=True)
    reciprocal_ranks: list[float] = []
    for i in range(targets.shape[0]):
        target = targets[i].item()
        ranks = (sorted_idx[i] == target).nonzero(as_tuple=False)
        if ranks.numel() == 0:
            reciprocal_ranks.append(0.0)
            continue
        rank = int(ranks[0].item()) + 1
        reciprocal_ranks.append(1.0 / rank)
    return float(sum(reciprocal_ranks) / len(reciprocal_ranks))


@torch.no_grad()
def dataloader_metrics(
    model: nn.Module,
    dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: str = "cpu",
) -> dict[str, float]:
    """Aggregate weighted evaluation metrics across a dataloader."""

    model.eval()
    total_loss = 0.0
    total_examples = 0

    top1_correct = 0.0
    top3_correct = 0.0
    top5_correct = 0.0
    mrr_sum = 0.0

    for contexts, targets in dataloader:
        contexts = contexts.to(device)
        targets = targets.to(device)

        logits = model(contexts)
        loss = criterion(logits, targets)
        batch_size = int(targets.shape[0])

        # Criterion reduction is mean by default; convert to summed token loss.
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

        top1_correct += top_k_accuracy_from_logits(logits, targets, k=1) * batch_size
        top3_correct += top_k_accuracy_from_logits(logits, targets, k=3) * batch_size
        top5_correct += top_k_accuracy_from_logits(logits, targets, k=5) * batch_size
        mrr_sum += mean_reciprocal_rank(logits, targets) * batch_size

    avg_loss = total_loss / max(total_examples, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    top1 = top1_correct / max(total_examples, 1)
    top3 = top3_correct / max(total_examples, 1)
    top5 = top5_correct / max(total_examples, 1)
    mrr = mrr_sum / max(total_examples, 1)

    return {
        "cross_entropy": float(avg_loss),
        "perplexity": float(ppl),
        "top1_accuracy": float(top1),
        "top3_accuracy": float(top3),
        "top5_accuracy": float(top5),
        # Single-label next-token prediction: recall@k equals top-k accuracy.
        "recall_at_3": float(top3),
        "recall_at_5": float(top5),
        "mrr": float(mrr),
    }


@torch.no_grad()
def perplexity(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: str = "cpu",
) -> float:
    """Backward-compatible perplexity helper."""

    metrics = dataloader_metrics(model, dataloader, criterion, device)
    return float(metrics["perplexity"])


@torch.no_grad()
def top_k_accuracy(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    k: int = 5,
    device: str = "cpu",
) -> float:
    """Backward-compatible top-k accuracy helper."""

    model.eval()
    total_acc = 0.0
    total_batches = 0
    for contexts, targets in dataloader:
        contexts = contexts.to(device)
        targets = targets.to(device)
        logits = model(contexts)
        total_acc += top_k_accuracy_from_logits(logits, targets, k)
        total_batches += 1
    return float(total_acc / max(total_batches, 1))


@torch.no_grad()
def next_k_predictions(
    model: nn.Module,
    context: torch.Tensor,
    k: int = 5,
    device: str = "cpu",
) -> tuple[list[int], list[float]]:
    """Get top-k token IDs and probabilities for one context row."""

    model.eval()
    context = context.to(device)
    logits = model(context)
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = probs.topk(k)
    return top_indices[0].tolist(), top_probs[0].tolist()


def format_metric_row(model_name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    """Attach model name to metric dictionary for tabular reporting."""

    out = {"model": model_name}
    out.update(metrics)
    return out
