#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model factory for deep learning style models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor


def select_torch_device(torch: Any, requested: str | None = None) -> str:
    """Choose an accelerator, preferring CUDA, then Apple MPS, then CPU."""
    requested_device = str(requested or "auto").lower()
    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(
        hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    if requested_device in {"", "auto"}:
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"
    if requested_device.startswith("cuda"):
        if cuda_available:
            return str(requested or "cuda")
        return "mps" if mps_available else "cpu"
    if requested_device == "mps":
        if mps_available:
            return "mps"
        return "cuda" if cuda_available else "cpu"
    return requested_device


def make_deep_model(
    model_type: str = "mlp",
    random_state: int = 42,
    lookback: int = 30,
    epochs: int = 25,
    batch_size: int = 1024,
    hidden_size: int = 64,
):
    model_type = model_type.lower()
    if model_type in {"mlp", "sequence_mlp"}:
        return MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=512,
            learning_rate_init=0.001,
            max_iter=80,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=8,
            random_state=random_state,
            verbose=False,
    )
    if model_type in {"lstm", "seq_lstm", "sequence_lstm"}:
        import torch

        return TorchLSTMRegressor(
            lookback=lookback,
            hidden_size=hidden_size,
            epochs=epochs,
            batch_size=batch_size,
            random_state=random_state,
            device=select_torch_device(torch),
        )
    raise ValueError(f"Unsupported deep model type: {model_type}")


class _LSTMNetMixin:
    """Marker mixin for pickle-friendly local type checks."""


def _build_lstm_net(input_size: int, hidden_size: int, num_layers: int, dropout: float):
    import torch
    from torch import nn

    class LSTMNet(nn.Module, _LSTMNetMixin):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    return LSTMNet()


@dataclass
class TorchLSTMRegressor:
    """Small sklearn-like PyTorch LSTM regressor.

    Inputs are flattened lookback windows. The wrapper reshapes them to
    ``batch x lookback x feature_count`` internally so it can share the same
    dataset/scaler path as the MLP model.
    """

    lookback: int = 30
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.15
    lr: float = 0.001
    weight_decay: float = 0.0001
    batch_size: int = 1024
    epochs: int = 25
    patience: int = 5
    random_state: int = 42
    device: str = "cpu"
    input_size_: int | None = None
    best_val_loss_: float = 0.0
    loss_curve_: list[float] = field(default_factory=list)
    val_loss_curve_: list[float] = field(default_factory=list)
    state_dict_: dict[str, Any] | None = None

    def _runtime_device(self, torch: Any) -> str:
        return select_torch_device(torch, self.device)

    def fit(self, x: np.ndarray, y: np.ndarray):
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset

        device = self._runtime_device(torch)
        torch.manual_seed(self.random_state)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] % self.lookback != 0:
            raise ValueError("LSTM input must be flattened windows with columns divisible by lookback")
        self.input_size_ = x.shape[1] // self.lookback
        x_seq = x.reshape(len(x), self.lookback, self.input_size_)

        split = max(1, int(len(x_seq) * 0.88))
        if split >= len(x_seq):
            split = len(x_seq)
        x_train, y_train = x_seq[:split], y[:split]
        x_valid, y_valid = x_seq[split:], y[split:]

        train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        if len(x_valid):
            valid_x = torch.from_numpy(x_valid).to(device)
            valid_y = torch.from_numpy(y_valid).to(device)
        else:
            valid_x = valid_y = None

        model = _build_lstm_net(self.input_size_, self.hidden_size, self.num_layers, self.dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.SmoothL1Loss()
        best_loss = float("inf")
        best_state = None
        no_improve = 0

        for _epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            total_count = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(yb)
                total_count += len(yb)
            train_loss = total_loss / max(total_count, 1)
            self.loss_curve_.append(float(train_loss))

            if valid_x is not None and len(valid_x):
                model.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(model(valid_x), valid_y).detach().cpu())
            else:
                val_loss = train_loss
            self.val_loss_curve_.append(float(val_loss))

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_loss = self.loss_curve_[-1] if self.loss_curve_ else 0.0
        self.state_dict_ = best_state
        self.best_val_loss_ = float(best_loss)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        if self.input_size_ is None or self.state_dict_ is None:
            raise RuntimeError("TorchLSTMRegressor is not fitted")
        device = self._runtime_device(torch)
        x = np.asarray(x, dtype=np.float32)
        x_seq = x.reshape(len(x), self.lookback, self.input_size_)
        model = _build_lstm_net(self.input_size_, self.hidden_size, self.num_layers, self.dropout).to(device)
        state_dict = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in self.state_dict_.items()
        }
        model.load_state_dict(state_dict)
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(x_seq), self.batch_size):
                batch = torch.from_numpy(x_seq[start : start + self.batch_size]).to(device)
                preds.append(model(batch).detach().cpu().numpy())
        return np.concatenate(preds).astype(float) if preds else np.asarray([], dtype=float)
