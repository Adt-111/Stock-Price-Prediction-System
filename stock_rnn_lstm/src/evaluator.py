"""
RMSE, MAE, MAPE, and directional accuracy for comparing SimpleRNN vs LSTM
on the (inverse-transformed, dollar-scale) test set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EvaluationMetrics:
    """Container for one model's evaluation results on a test set."""
    model_name: str
    rmse: float
    mae: float
    mape: float
    directional_accuracy: float
    n_samples: int

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "model_name": self.model_name,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "directional_accuracy": self.directional_accuracy,
            "n_samples": self.n_samples,
        }


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error, in the same units as y_true/y_pred (e.g. dollars)."""
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    _validate_shapes(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error, in the same units as y_true/y_pred."""
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    _validate_shapes(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error, as a percentage (e.g. 3.2 means 3.2%).
    `epsilon` guards against division by zero for any near-zero true values
    (not expected for stock prices, but kept for robustness).
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    _validate_shapes(y_true, y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), epsilon, None))) * 100.0)


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, previous_actual: np.ndarray) -> float:
    """
    % of test samples where the predicted up/down move matched the actual
    move, both measured against previous_actual (y_true shifted by one).
    Magnitude doesn't matter, just the sign.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    previous_actual = np.asarray(previous_actual, dtype=np.float64)
    _validate_shapes(y_true, y_pred)
    if previous_actual.shape != y_true.shape:
        raise ValueError(
            f"previous_actual shape {previous_actual.shape} must match y_true shape {y_true.shape}"
        )

    true_direction = np.sign(y_true - previous_actual)
    pred_direction = np.sign(y_pred - previous_actual)

    # Treat a predicted "no change" (sign == 0) as neither correct nor incorrect
    # in isolation; compare directly — sign(0) == sign(0) counts as correct only
    # when the actual move was also exactly zero, which is the intuitive behavior.
    correct = (true_direction == pred_direction)
    return float(np.mean(correct) * 100.0)


def _validate_shapes(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true shape {y_true.shape} must match y_pred shape {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("y_true/y_pred must not be empty.")


def evaluate_model(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray, previous_actual: np.ndarray
) -> EvaluationMetrics:
    """Computes the full metric suite for one model's test-set predictions (all on original $ scale)."""
    return EvaluationMetrics(
        model_name=model_name,
        rmse=compute_rmse(y_true, y_pred),
        mae=compute_mae(y_true, y_pred),
        mape=compute_mape(y_true, y_pred),
        directional_accuracy=compute_directional_accuracy(y_true, y_pred, previous_actual),
        n_samples=int(len(y_true)),
    )


def compare_models(results: list[EvaluationMetrics]) -> pd.DataFrame:
    """
    Builds a tidy comparison table across multiple models' `EvaluationMetrics`,
    e.g. for display in the Streamlit dashboard or a notebook.
    """
    if not results:
        raise ValueError("results must contain at least one EvaluationMetrics.")
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows).set_index("model_name")
    df = df.rename(
        columns={
            "rmse": "RMSE ($)",
            "mae": "MAE ($)",
            "mape": "MAPE (%)",
            "directional_accuracy": "Directional Accuracy (%)",
            "n_samples": "Test Samples",
        }
    )
    return df.round(4)


if __name__ == "__main__":  # pragma: no cover
    rng = np.random.default_rng(42)
    n = 100

    actual = 150 + np.cumsum(rng.normal(0, 1.5, size=n))
    previous = np.roll(actual, 1)
    previous[0] = actual[0] - rng.normal(0, 1.5)

    # Simulate two models: one fairly accurate, one noisier.
    rnn_pred = actual + rng.normal(0, 2.5, size=n)
    lstm_pred = actual + rng.normal(0, 1.2, size=n)

    rnn_metrics = evaluate_model("SimpleRNN", actual, rnn_pred, previous)
    lstm_metrics = evaluate_model("LSTM", actual, lstm_pred, previous)

    comparison = compare_models([rnn_metrics, lstm_metrics])
    print(comparison)
