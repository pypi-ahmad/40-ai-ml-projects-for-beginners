"""Calibration evaluation and temperature scaling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.calibration import calibration_curve


@dataclass(slots=True)
class CalibrationResult:
    temperature: float
    ece_before: float
    ece_after: float


class TemperatureScaler(nn.Module):
    """Single-parameter temperature scaling module."""

    def __init__(self) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=1e-4)

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 50) -> float:
        """Fit temperature on validation logits."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().cpu().item())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute expected calibration error (ECE)."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == labels

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        if not np.any(in_bin):
            continue
        bin_acc = np.mean(accuracies[in_bin])
        bin_conf = np.mean(confidences[in_bin])
        ece += np.abs(bin_acc - bin_conf) * np.mean(in_bin)

    return float(ece)


def calibration_curve_points(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fraction positive and mean predicted value arrays."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    binary_correct = (predictions == labels).astype(int)
    frac_pos, mean_pred = calibration_curve(binary_correct, confidences, n_bins=n_bins, strategy="uniform")
    return frac_pos, mean_pred
