"""Training loops and checkpointing helpers for neural language models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(slots=True)
class EarlyStopping:
    """Simple patience-based early stopping."""

    patience: int = 3
    min_delta: float = 0.0
    best_loss: float = field(default=float("inf"), init=False)
    counter: int = field(default=0, init=False)
    early_stop: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.early_stop = False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


@dataclass(slots=True)
class EpochMetrics:
    """Per-epoch training metrics."""

    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    *,
    device: str = "cpu",
) -> float:
    """Evaluate one epoch and return average loss."""

    model.eval()
    total_loss = 0.0
    total_batches = 0

    for contexts, targets in dataloader:
        contexts = contexts.to(device)
        targets = targets.to(device)
        logits = model(contexts)
        loss = criterion(logits, targets)
        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    *,
    clip: float = 1.0,
    device: str = "cpu",
    use_amp: bool = False,
) -> float:
    """Train one epoch with optional mixed precision."""

    model.train()
    total_loss = 0.0
    total_batches = 0

    scaler = torch.amp.GradScaler("cuda") if use_amp and device.startswith("cuda") else None

    for contexts, targets in tqdm(dataloader, desc="Train", leave=False):
        contexts = contexts.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(contexts)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(contexts)
            loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str | Path,
) -> None:
    """Save model checkpoint to disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    path: str | Path,
    device: str = "cpu",
) -> int:
    """Load checkpoint. Returns epoch index stored in checkpoint."""

    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["epoch"])


def fit_model(
    *,
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    checkpoint_path: Path,
    history_path: Path,
    clip: float = 1.0,
    scheduler_patience: int = 2,
    early_stopping_patience: int = 3,
    device: str = "cpu",
    use_amp: bool = False,
) -> list[EpochMetrics]:
    """Run full training loop with scheduler + early stopping."""

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=scheduler_patience,
    )
    stopper = EarlyStopping(patience=early_stopping_patience, min_delta=1e-4)

    history: list[EpochMetrics] = []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            clip=clip,
            device=device,
            use_amp=use_amp,
        )
        val_loss = evaluate_epoch(model, val_loader, criterion, device=device)

        scheduler.step(val_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=float(train_loss),
            val_loss=float(val_loss),
            learning_rate=learning_rate,
        )
        history.append(metrics)

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        if stopper.step(val_loss):
            break

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps([asdict(item) for item in history], indent=2),
        encoding="utf-8",
    )

    return history
