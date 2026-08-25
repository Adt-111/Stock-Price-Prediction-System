"""
Training loop shared by both `SimpleRNNModel` and `LSTMModel`: mini-batch
gradient descent on MSE loss, per-epoch MAE logging, gradient clipping, and
early stopping on validation loss with best-weights restoration.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Per-epoch loss history for one model's training run."""
    train_mse: list[float] = field(default_factory=list)
    train_mae: list[float] = field(default_factory=list)
    val_mse: list[float] = field(default_factory=list)
    val_mae: list[float] = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False
    total_epochs_run: int = 0


@dataclass
class EarlyStoppingConfig:
    patience: int = 10
    min_delta: float = 1e-5


def _make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32)).unsqueeze(-1)  # (n, 1) to match model output
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip_norm: float = 1.0,
    early_stopping: EarlyStoppingConfig | None = None,
    device: torch.device | str = "cpu",
    model_name: str = "model",
    verbose: bool = True,
) -> TrainingHistory:
    """
    Trains `model` in place: Adam + MSE, gradient clipping, per-epoch
    MSE/MAE logging on train and val, early stopping on val MSE.
    X_val/y_val is usually the chronological test split held out entirely
    from training. Restores the best-val-MSE weights when done (whether or
    not early stopping actually triggered).
    """
    if len(X_train) == 0:
        raise ValueError("X_train is empty — cannot train on zero samples.")
    if len(X_val) == 0:
        raise ValueError("X_val is empty — cannot validate on zero samples.")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()

    train_loader = _make_dataloader(X_train, y_train, batch_size, shuffle=True)
    val_loader = _make_dataloader(X_val, y_val, batch_size, shuffle=False)

    history = TrainingHistory()
    best_val_mse = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(1, num_epochs + 1):
        # --- Training phase --- #
        model.train()
        running_mse, running_mae, n_train = 0.0, 0.0, 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = mse_criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            batch_n = X_batch.size(0)
            running_mse += loss.item() * batch_n
            running_mae += mae_criterion(predictions, y_batch).item() * batch_n
            n_train += batch_n

        train_mse = running_mse / n_train
        train_mae = running_mae / n_train

        # --- Validation phase --- #
        val_mse, val_mae = _evaluate_loss(model, val_loader, mse_criterion, mae_criterion, device)

        history.train_mse.append(train_mse)
        history.train_mae.append(train_mae)
        history.val_mse.append(val_mse)
        history.val_mae.append(val_mae)
        history.total_epochs_run = epoch

        if verbose:
            logger.info(
                "[%s] Epoch %3d/%d | train_mse=%.6f train_mae=%.6f | val_mse=%.6f val_mae=%.6f",
                model_name, epoch, num_epochs, train_mse, train_mae, val_mse, val_mae,
            )

        # --- Early stopping bookkeeping --- #
        improved = val_mse < (best_val_mse - (early_stopping.min_delta if early_stopping else 0.0))
        if improved:
            best_val_mse = val_mse
            best_state_dict = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if early_stopping is not None and epochs_without_improvement >= early_stopping.patience:
            history.stopped_early = True
            if verbose:
                logger.info(
                    "[%s] Early stopping triggered at epoch %d (best val_mse=%.6f at epoch %d).",
                    model_name, epoch, best_val_mse, history.best_epoch,
                )
            break

    # Restore the best-validation-MSE weights (matches early-stopping best practice
    # even when early stopping wasn't actually triggered before num_epochs ran out).
    model.load_state_dict(best_state_dict)
    return history


@torch.no_grad()
def _evaluate_loss(
    model: nn.Module, loader: DataLoader, mse_criterion: nn.Module, mae_criterion: nn.Module, device: torch.device | str
) -> tuple[float, float]:
    model.eval()
    running_mse, running_mae, n = 0.0, 0.0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        predictions = model(X_batch)
        batch_n = X_batch.size(0)
        running_mse += mse_criterion(predictions, y_batch).item() * batch_n
        running_mae += mae_criterion(predictions, y_batch).item() * batch_n
        n += batch_n
    return running_mse / n, running_mae / n


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.path.insert(0, ".")
    from src.models import LSTMModel, SimpleRNNModel

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)
    np.random.seed(0)

    # Synthetic sanity-check dataset: a noisy sine wave, easy for either model to learn.
    n_points, seq_len, n_features = 800, 30, 1
    t = np.linspace(0, 40 * np.pi, n_points)
    series = (np.sin(t) + np.random.normal(0, 0.05, size=n_points)).astype(np.float32).reshape(-1, 1)

    def window(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i : i + seq_length])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)

    X, y = window(series, seq_len)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    model = SimpleRNNModel(input_size=n_features, hidden_size=32, num_layers=2, dropout=0.2)
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        num_epochs=15, batch_size=16, early_stopping=EarlyStoppingConfig(patience=5),
        model_name="SimpleRNN-smoketest", verbose=True,
    )
    print(f"Final train MSE: {history.train_mse[-1]:.6f} | Final val MSE: {history.val_mse[-1]:.6f}")
    print(f"Best epoch: {history.best_epoch} | Stopped early: {history.stopped_early}")
    assert history.train_mse[-1] < history.train_mse[0], "Loss should decrease during training"
    print("train.py smoke test passed.")
