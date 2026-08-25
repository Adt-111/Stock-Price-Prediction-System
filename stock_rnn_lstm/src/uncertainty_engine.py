"""
MC Dropout: run a bunch of stochastic forward passes with dropout left on
at inference time, use the spread of predictions as an uncertainty
estimate (Gal & Ghahramani, 2016).

The model as a whole stays in eval() so BatchNorm keeps using its learned
running stats -- only the Dropout sub-modules get individually flipped
back to train() so they keep zeroing activations across simulations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class UncertaintyForecast:
    """Result of `predict_with_uncertainty`: mean forecast + confidence bounds, in original $ scale."""
    mean: np.ndarray               # (n_samples,) mean predicted price across MC simulations
    lower_bound: np.ndarray        # (n_samples,) lower confidence bound (e.g. 2.5th percentile)
    upper_bound: np.ndarray        # (n_samples,) upper confidence bound (e.g. 97.5th percentile)
    std: np.ndarray                # (n_samples,) standard deviation across MC simulations
    confidence_level: float        # e.g. 0.95
    num_simulations: int
    raw_simulations: np.ndarray    # (num_simulations, n_samples) every individual MC forward pass


def _enable_mc_dropout(model: nn.Module) -> int:
    """
    eval() the whole model, then flip just the Dropout sub-modules back to
    train() so they keep firing. Returns how many were re-enabled (handy
    to sanity-check against a model with dropout=0).
    """
    model.eval()
    n_dropout_layers = 0
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            module.train()
            n_dropout_layers += 1
    return n_dropout_layers


@torch.no_grad()
def predict_with_uncertainty(
    model: nn.Module,
    sequence_input: torch.Tensor | np.ndarray,
    num_simulations: int = 100,
    confidence_level: float = 0.95,
    device: torch.device | str = "cpu",
) -> UncertaintyForecast:
    """
    Runs `num_simulations` stochastic forward passes with dropout active
    and computes the mean plus confidence bounds from the resulting
    distribution at each point. `sequence_input` should already be scaled
    the same way training data was -- batch can be 1 or the whole test set.

    Returns scaled values; call `.inverse_transform_target(...)` on
    `.mean`, `.lower_bound`, `.upper_bound` separately to get dollars back.
    Raises if the model has no Dropout layers, since MC Dropout would just
    be repeating the same prediction N times.
    """
    if num_simulations < 2:
        raise ValueError(f"num_simulations must be >= 2, got {num_simulations}")
    if not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    n_dropout_layers = _enable_mc_dropout(model)
    if n_dropout_layers == 0:
        raise ValueError(
            "Model has no nn.Dropout layers — Monte Carlo Dropout requires at least one "
            "active Dropout layer to produce a meaningful predictive distribution."
        )

    if isinstance(sequence_input, np.ndarray):
        sequence_input = torch.from_numpy(sequence_input.astype(np.float32))
    sequence_input = sequence_input.to(device)
    model = model.to(device)

    simulations = []
    for _ in range(num_simulations):
        preds = model(sequence_input)          # (batch, output_size)
        simulations.append(preds.squeeze(-1).cpu().numpy())  # (batch,)

    raw_simulations = np.stack(simulations, axis=0)   # (num_simulations, batch)

    mean = raw_simulations.mean(axis=0)
    std = raw_simulations.std(axis=0)

    alpha = 1.0 - confidence_level
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0
    lower_bound = np.percentile(raw_simulations, lower_pct, axis=0)
    upper_bound = np.percentile(raw_simulations, upper_pct, axis=0)

    model.eval()  # restore full eval mode (including Dropout) for any subsequent deterministic use

    return UncertaintyForecast(
        mean=mean,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        std=std,
        confidence_level=confidence_level,
        num_simulations=num_simulations,
        raw_simulations=raw_simulations,
    )


def forecast_to_dataframe(
    forecast: UncertaintyForecast,
    inverse_transform_fn,
    dates=None,
) -> "pd.DataFrame":  # noqa: F821 - pandas imported lazily below to keep this module import-light
    """
    Convenience helper: inverse-transforms an `UncertaintyForecast`'s
    scaled mean/bounds back to original dollar values via
    `inverse_transform_fn` (typically `PreparedDataset.inverse_transform_target`)
    and packages the result as a tidy DataFrame for plotting/display.
    """
    import pandas as pd

    mean_dollars = inverse_transform_fn(forecast.mean)
    lower_dollars = inverse_transform_fn(forecast.lower_bound)
    upper_dollars = inverse_transform_fn(forecast.upper_bound)

    df = pd.DataFrame(
        {
            "predicted_mean": mean_dollars,
            "lower_bound": lower_dollars,
            "upper_bound": upper_dollars,
        }
    )
    if dates is not None:
        df.insert(0, "date", np.asarray(dates))
    return df


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.path.insert(0, ".")
    from src.models import LSTMModel

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    batch, seq_len, n_features = 20, 60, 2
    dummy_input = torch.randn(batch, seq_len, n_features)

    model = LSTMModel(input_size=n_features, hidden_size=32, num_layers=2, dropout=0.3)
    model.eval()

    forecast = predict_with_uncertainty(model, dummy_input, num_simulations=100, confidence_level=0.95)

    print(f"Mean shape: {forecast.mean.shape}")
    print(f"Lower bound (first 5): {forecast.lower_bound[:5]}")
    print(f"Upper bound (first 5): {forecast.upper_bound[:5]}")
    print(f"Mean (first 5):        {forecast.mean[:5]}")
    assert np.all(forecast.lower_bound <= forecast.mean + 1e-6)
    assert np.all(forecast.mean <= forecast.upper_bound + 1e-6)
    print("uncertainty_engine.py smoke test passed: bounds correctly bracket the mean.")
