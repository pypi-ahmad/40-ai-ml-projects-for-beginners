from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation import regression_metrics


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DeepLearningConfig:
    sequence_length: int = 30
    hidden_size: int = 64
    dropout: float = 0.2
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 25
    early_stopping_patience: int = 5
    device: str = "cpu"



def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    if len(X) <= sequence_length:
        raise ValueError("Not enough rows for sequence construction")

    X_seq: list[np.ndarray] = []
    y_seq: list[float] = []
    for idx in range(sequence_length, len(X)):
        X_seq.append(X[idx - sequence_length : idx])
        y_seq.append(y[idx])
    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)



def scale_for_sequences(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train_s = x_scaler.fit_transform(X_train)
    X_val_s = x_scaler.transform(X_val)
    X_test_s = x_scaler.transform(X_test)

    y_train_s = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_s = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
    y_test_s = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, x_scaler, y_scaler


class VanillaLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.head(out).squeeze(-1)


class StackedLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.head(out).squeeze(-1)


class GRUNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class CNNLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, seq, feat] -> conv expects [batch, feat, seq]
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.head(out).squeeze(-1)


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        # Align tail when padding adds extra positions.
        if out.size(-1) != residual.size(-1):
            out = out[..., : residual.size(-1)]
        return self.relu(out + residual)


class TCNRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.block1 = TemporalBlock(input_size, hidden_size, dilation=1, dropout=dropout)
        self.block2 = TemporalBlock(hidden_size, hidden_size, dilation=2, dropout=dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        out = self.block1(x)
        out = self.block2(out)
        out = out[:, :, -1]
        return self.head(out).squeeze(-1)



def _make_architecture(arch: str, input_size: int, hidden_size: int, dropout: float) -> nn.Module:
    arch = arch.lower()
    if arch == "vanilla_lstm":
        return VanillaLSTM(input_size, hidden_size, dropout)
    if arch == "stacked_lstm":
        return StackedLSTM(input_size, hidden_size, dropout)
    if arch == "bidirectional_lstm":
        return BidirectionalLSTM(input_size, hidden_size, dropout)
    if arch == "gru":
        return GRUNet(input_size, hidden_size, dropout)
    if arch == "cnn_lstm":
        return CNNLSTM(input_size, hidden_size, dropout)
    if arch == "tcn":
        return TCNRegressor(input_size, hidden_size, dropout)
    raise ValueError(f"Unknown architecture: {arch}")



def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train: bool,
    device: torch.device,
) -> float:
    model.train(train)
    losses: list[float] = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        if train:
            loss.backward()
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("inf")



def train_sequence_model(
    architecture: str,
    X_train_seq: np.ndarray,
    y_train_seq: np.ndarray,
    X_val_seq: np.ndarray,
    y_val_seq: np.ndarray,
    config: DeepLearningConfig,
) -> tuple[nn.Module, dict[str, list[float]]]:
    device = torch.device(config.device)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_seq).float(),
        torch.from_numpy(y_train_seq).float(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_seq).float(),
        torch.from_numpy(y_val_seq).float(),
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    model = _make_architecture(
        architecture,
        input_size=X_train_seq.shape[-1],
        hidden_size=config.hidden_size,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(config.epochs):
        train_loss = _run_epoch(model, train_loader, optimizer, loss_fn, train=True, device=device)
        val_loss = _run_epoch(model, val_loader, optimizer, loss_fn, train=False, device=device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= config.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info("%s best_val=%.6f epoch=%d", architecture, best_val, best_epoch)
    return model, history



def predict_sequence_model(model: nn.Module, X_seq: np.ndarray, device: str = "cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(X_seq).float().to(device)
        pred = model(x).detach().cpu().numpy().ravel()
    return pred



def run_deep_learning_benchmark(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    architectures: list[str],
    config: DeepLearningConfig,
) -> dict[str, dict[str, object]]:
    """Train requested sequence architectures and return predictions + metrics."""
    (
        X_train_s,
        X_val_s,
        X_test_s,
        y_train_s,
        y_val_s,
        y_test_s,
        _,
        y_scaler,
    ) = scale_for_sequences(X_train, X_val, X_test, y_train, y_val, y_test)

    X_train_seq, y_train_seq = make_sequences(X_train_s, y_train_s, config.sequence_length)
    X_val_seq, y_val_seq = make_sequences(X_val_s, y_val_s, config.sequence_length)
    X_test_seq, y_test_seq = make_sequences(X_test_s, y_test_s, config.sequence_length)

    # Keep aligned original targets for inverse transformed metrics.
    y_val_true = y_val[config.sequence_length :]
    y_test_true = y_test[config.sequence_length :]

    outputs: dict[str, dict[str, object]] = {}
    for arch in architectures:
        model, history = train_sequence_model(
            architecture=arch,
            X_train_seq=X_train_seq,
            y_train_seq=y_train_seq,
            X_val_seq=X_val_seq,
            y_val_seq=y_val_seq,
            config=config,
        )

        val_pred_scaled = predict_sequence_model(model, X_val_seq, device=config.device)
        test_pred_scaled = predict_sequence_model(model, X_test_seq, device=config.device)

        val_pred = y_scaler.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel()
        test_pred = y_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel()

        outputs[arch] = {
            "model": model,
            "history": history,
            "val_pred": val_pred,
            "test_pred": test_pred,
            "val_metrics": regression_metrics(y_val_true, val_pred),
            "test_metrics": regression_metrics(y_test_true, test_pred),
        }

    return outputs
