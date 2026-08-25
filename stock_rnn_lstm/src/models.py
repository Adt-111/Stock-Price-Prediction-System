"""
SimpleRNNModel: RNN -> BatchNorm -> Dropout -> Linear
LSTMModel:      LSTM -> Dropout -> Linear

Both take (batch, seq_length, n_features) and output one scalar (the next
scaled Close price). Dropout stays identifiable so uncertainty_engine.py
can flip it back on for MC Dropout at inference, while BatchNorm keeps
using its running stats instead of noisy per-sample batch stats.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleRNNModel(nn.Module):
    """
    RNN(num_layers, hidden_size) -> last timestep -> BatchNorm1d ->
    Dropout -> Linear(hidden_size, output_size).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            # nn.RNN only applies inter-layer dropout when num_layers > 1;
            # PyTorch warns (harmlessly) if dropout > 0 with a single layer.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, input_size)
        Returns:
            (batch, output_size) prediction.
        """
        if x.dim() != 3 or x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected input shape (batch, seq_length, {self.input_size}), got {tuple(x.shape)}"
            )

        rnn_out, _ = self.rnn(x)                 # (batch, seq_length, hidden_size)
        last_hidden = rnn_out[:, -1, :]            # (batch, hidden_size) — final timestep
        normalized = self.batch_norm(last_hidden)
        dropped = self.dropout(normalized)
        return self.fc(dropped)


class LSTMModel(nn.Module):
    """LSTM(num_layers, hidden_size) -> last timestep -> Dropout -> Linear."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_length, input_size)
        Returns:
            (batch, output_size) prediction.
        """
        if x.dim() != 3 or x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected input shape (batch, seq_length, {self.input_size}), got {tuple(x.shape)}"
            )

        lstm_out, _ = self.lstm(x)                # (batch, seq_length, hidden_size)
        last_hidden = lstm_out[:, -1, :]            # (batch, hidden_size) — final timestep
        dropped = self.dropout(last_hidden)
        return self.fc(dropped)


def count_trainable_parameters(model: nn.Module) -> int:
    """Utility: total number of trainable parameters in a model (handy for logging/comparison)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":  # pragma: no cover
    torch.manual_seed(0)
    batch, seq_len, n_features = 8, 60, 2

    dummy_input = torch.randn(batch, seq_len, n_features)

    rnn_model = SimpleRNNModel(input_size=n_features, hidden_size=64, num_layers=2, dropout=0.2)
    rnn_model.eval()
    rnn_out = rnn_model(dummy_input)
    print(f"SimpleRNNModel output shape: {tuple(rnn_out.shape)} "
          f"(params={count_trainable_parameters(rnn_model):,})")

    lstm_model = LSTMModel(input_size=n_features, hidden_size=64, num_layers=2, dropout=0.2)
    lstm_model.eval()
    lstm_out = lstm_model(dummy_input)
    print(f"LSTMModel output shape: {tuple(lstm_out.shape)} "
          f"(params={count_trainable_parameters(lstm_model):,})")

    assert rnn_out.shape == (batch, 1)
    assert lstm_out.shape == (batch, 1)
    print("models.py smoke test passed.")
