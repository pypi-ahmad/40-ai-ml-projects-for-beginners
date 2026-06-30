import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.evaluation import (
    dataloader_metrics,
    mean_reciprocal_rank,
    recall_at_k,
    top_k_accuracy_from_logits,
)


def test_top_k_accuracy_and_recall_metrics() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.9, 0.0],
            [0.8, 0.1, 0.1],
            [0.2, 0.2, 0.6],
        ]
    )
    targets = torch.tensor([1, 2, 0])

    acc1 = top_k_accuracy_from_logits(logits, targets, k=1)
    acc2 = top_k_accuracy_from_logits(logits, targets, k=2)
    rec2 = recall_at_k(logits, targets, k=2)

    assert 0 <= acc1 <= acc2 <= 1
    assert rec2 == acc2


def test_mean_reciprocal_rank() -> None:
    logits = torch.tensor(
        [
            [0.1, 0.5, 0.4],
            [0.9, 0.08, 0.02],
        ]
    )
    targets = torch.tensor([2, 1])
    mrr = mean_reciprocal_rank(logits, targets)
    assert 0 < mrr < 1


class ConstantLogitModel(nn.Module):
    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        index = x[:, 0].long()
        return self.fixed_logits[index]


def test_dataloader_metrics_uses_sample_weighted_loss() -> None:
    # 3 examples with deterministic logits.
    contexts = torch.tensor(
        [
            [0, 2],
            [1, 4],
            [2, 6],
        ],
        dtype=torch.long,
    )
    targets = torch.tensor([0, 1, 1], dtype=torch.long)
    dataset = list(zip(contexts, targets, strict=False))
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    logits = torch.tensor(
        [
            [3.0, 1.0],
            [0.1, 2.2],
            [0.2, 1.9],
        ],
        dtype=torch.float32,
    )
    model = ConstantLogitModel(logits)
    criterion = nn.CrossEntropyLoss()

    metrics = dataloader_metrics(model, loader, criterion, device="cpu")

    manual_loss = criterion(logits, targets).item()
    expected_ppl = float(torch.exp(torch.tensor(manual_loss)).item())
    assert abs(metrics["cross_entropy"] - manual_loss) < 1e-6
    assert abs(metrics["perplexity"] - expected_ppl) < 1e-6
