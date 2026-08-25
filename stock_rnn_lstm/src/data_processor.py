"""
Fetches OHLCV data, cleans it, scales it, and builds sliding windows for
the RNN/LSTM models.

On leakage: prices get split into train/test chronologically first, and
the scaler is fit only on the train side before transforming everything.
Windows are then built across the full series (early test windows can look
back into the tail of train for context -- that's fine, it's still only
past data) and assigned to train or test based on where their target
falls relative to the split point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore


# Fetching & cleaning

def _synthesize_mock_price_series(
    ticker: str, start_date: str, end_date: str, seed: int | None = None
) -> pd.DataFrame:
    """
    Generates a realistic-looking synthetic daily OHLCV series via geometric
    Brownian motion, used as an offline fallback when Yahoo Finance is
    unreachable (no network access) or returns no data for the requested
    ticker/date range. Never raises — this keeps the rest of the pipeline
    fully exercisable without live network access.
    """
    seed = seed if seed is not None else abs(hash(ticker)) % (2**31)
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start_date, end=end_date)
    if len(dates) == 0:
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=500)

    start_price = float(rng.uniform(50, 400))
    daily_drift = rng.uniform(0.0002, 0.0006)
    daily_vol = rng.uniform(0.015, 0.03)
    log_returns = rng.normal(daily_drift, daily_vol, size=len(dates))
    close = start_price * np.cumprod(1 + log_returns)

    intraday_range = np.abs(rng.normal(0, daily_vol * 0.6, size=len(dates)))
    high = close * (1 + intraday_range)
    low = close * (1 - intraday_range)
    open_ = np.roll(close, 1)
    open_[0] = start_price
    volume = rng.integers(int(1e6), int(5e7), size=len(dates)).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    df.index.name = "Date"
    logger.warning(
        "Using synthetic mock price data for '%s' (%s to %s) — Yahoo Finance was "
        "unreachable or returned no data. Results are for pipeline demonstration only.",
        ticker, start_date, end_date,
    )
    return df


def fetch_and_clean_data(
    ticker: str,
    start_date: str,
    end_date: str,
    allow_mock_fallback: bool = True,
) -> pd.DataFrame:
    """
    Pulls daily OHLCV for `ticker` between the two 'YYYY-MM-DD' dates and
    fills any gaps (forward-fill then back-fill). Falls back to a
    synthetic series with a logged warning if yfinance can't get real data
    -- set allow_mock_fallback=False to raise instead.
    """
    if not ticker or not isinstance(ticker, str):
        raise ValueError(f"Invalid ticker: {ticker!r}")

    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"start_date/end_date must be 'YYYY-MM-DD' strings: {exc}")

    df: pd.DataFrame | None = None

    if yf is not None:
        try:
            raw = yf.download(
                ticker, start=start_date, end=end_date, progress=False, auto_adjust=True
            )
            if isinstance(raw.columns, pd.MultiIndex):
                # yfinance returns MultiIndex columns when auto_adjust groups by ticker
                # in some versions; flatten to the simple OHLCV column names.
                raw.columns = raw.columns.get_level_values(0)
            if raw is not None and not raw.empty:
                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception as exc:  # pragma: no cover - depends on live network
            logger.warning("yfinance download failed for '%s': %s", ticker, exc)

    if df is None or df.empty:
        if not allow_mock_fallback:
            raise ValueError(
                f"No data retrieved for ticker '{ticker}' between {start_date} and "
                f"{end_date}, and allow_mock_fallback=False."
            )
        df = _synthesize_mock_price_series(ticker, start_date, end_date)

    # Handle missing values: forward-fill (carry last known price forward across
    # gaps/holidays), then back-fill any remaining leading NaNs.
    n_missing_before = int(df.isna().sum().sum())
    df = df.ffill().bfill()
    n_missing_after = int(df.isna().sum().sum())
    if n_missing_before > 0:
        logger.info(
            "Filled %d missing values for '%s' (forward-fill then back-fill); "
            "%d remain (should be 0).",
            n_missing_before, ticker, n_missing_after,
        )

    if df.empty:
        raise ValueError(f"Cleaned dataset for '{ticker}' is empty after processing.")

    return df


# Sequence windowing

def create_sequences(
    data: np.ndarray, seq_length: int = 60, forecast_horizon: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Builds (X, y) sliding windows from a scaled (timesteps, features)
    array -- column 0 is the target. Returns X, y, and target_indices (the
    row in `data` each y came from, so the caller can split train/test
    without leakage).
    """
    if seq_length < 1:
        raise ValueError(f"seq_length must be >= 1, got {seq_length}")
    if forecast_horizon < 1:
        raise ValueError(f"forecast_horizon must be >= 1, got {forecast_horizon}")
    if data.ndim != 2:
        raise ValueError(f"data must be a 2-D array (timesteps, features), got shape {data.shape}")

    n_timesteps = data.shape[0]
    n_windows = n_timesteps - seq_length - forecast_horizon + 1
    if n_windows <= 0:
        raise ValueError(
            f"Not enough data ({n_timesteps} rows) for seq_length={seq_length} + "
            f"forecast_horizon={forecast_horizon}; need at least "
            f"{seq_length + forecast_horizon} rows."
        )

    n_features = data.shape[1]
    X = np.zeros((n_windows, seq_length, n_features), dtype=np.float32)
    y = np.zeros(n_windows, dtype=np.float32)
    target_indices = np.zeros(n_windows, dtype=np.int64)

    for i in range(n_windows):
        window_end = i + seq_length
        target_idx = window_end + forecast_horizon - 1
        X[i] = data[i:window_end]
        y[i] = data[target_idx, 0]
        target_indices[i] = target_idx

    return X, y, target_indices


# Full pipeline: fetch -> scale (train-only fit) -> window -> chronological split

@dataclass
class PreparedDataset:
    """Bundled output of `DataProcessor.prepare()`, ready to feed into training/inference."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: MinMaxScaler
    feature_columns: list[str] = field(default_factory=list)
    target_column_index: int = 0
    dates: pd.DatetimeIndex | None = None
    train_end_date: pd.Timestamp | None = None
    raw_df: pd.DataFrame | None = None

    def inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
        """
        Inverse-transforms a 1-D array of SCALED target predictions back to
        original dollar values. Handles the fact that the scaler was fit on
        multiple features (e.g. Close + Volume) by padding the target column
        into a dummy array of the right width before calling
        `scaler.inverse_transform`, then extracting just the target column back out.
        """
        scaled_values = np.asarray(scaled_values).reshape(-1)
        n_features = len(self.feature_columns)
        dummy = np.zeros((len(scaled_values), n_features), dtype=np.float64)
        dummy[:, self.target_column_index] = scaled_values
        inversed = self.scaler.inverse_transform(dummy)
        return inversed[:, self.target_column_index]


class DataProcessor:
    """
    End-to-end data pipeline orchestrator: fetch -> clean -> chronologically
    split -> fit-scale (train only) -> window -> assign windows to
    train/test by target date.
    """

    def __init__(
        self,
        seq_length: int = 60,
        forecast_horizon: int = 1,
        feature_columns: list[str] | None = None,
        target_column: str = "Close",
        test_split: float = 0.2,
    ) -> None:
        if not (0.0 < test_split < 1.0):
            raise ValueError(f"test_split must be in (0, 1), got {test_split}")

        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.feature_columns = feature_columns or ["Close", "Volume"]
        if target_column not in self.feature_columns:
            raise ValueError(
                f"target_column '{target_column}' must be one of feature_columns {self.feature_columns}"
            )
        self.target_column = target_column
        self.target_column_index = self.feature_columns.index(target_column)
        self.test_split = test_split

    def prepare(self, df: pd.DataFrame) -> PreparedDataset:
        """
        Runs the full leakage-safe pipeline described in the module
        docstring on an already-fetched-and-cleaned OHLCV DataFrame.
        """
        missing_cols = [c for c in self.feature_columns if c not in df.columns]
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")

        raw = df[self.feature_columns].values.astype(np.float64)
        n_rows = raw.shape[0]

        split_idx = int(n_rows * (1 - self.test_split))
        if split_idx <= self.seq_length:
            raise ValueError(
                f"Training portion ({split_idx} rows) is too small for seq_length="
                f"{self.seq_length}. Use a longer date range, a smaller seq_length, "
                f"or a smaller test_split."
            )

        # Fit the scaler ONLY on the training portion to avoid leaking test-set
        # scale/distribution information into the normalization.
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(raw[:split_idx])
        scaled = scaler.transform(raw)

        X, y, target_indices = create_sequences(scaled, self.seq_length, self.forecast_horizon)
        # Re-point the target column to the configured target (create_sequences
        # always uses column 0 as target internally) — reorder features so the
        # target column is first if it isn't already, keeping this transparent.
        if self.target_column_index != 0:
            X, y = self._reorder_target_to_front(X, y, scaled, target_indices)

        train_mask = target_indices < split_idx
        test_mask = ~train_mask

        dataset = PreparedDataset(
            X_train=X[train_mask],
            y_train=y[train_mask],
            X_test=X[test_mask],
            y_test=y[test_mask],
            scaler=scaler,
            feature_columns=self.feature_columns,
            target_column_index=self.target_column_index,
            dates=df.index,
            train_end_date=df.index[split_idx - 1] if split_idx > 0 else None,
            raw_df=df,
        )
        logger.info(
            "Prepared dataset: %d train windows, %d test windows (seq_length=%d, "
            "forecast_horizon=%d, split=%.0f%%/%.0f%%)",
            len(dataset.X_train), len(dataset.X_test), self.seq_length,
            self.forecast_horizon, (1 - self.test_split) * 100, self.test_split * 100,
        )
        return dataset

    def _reorder_target_to_front(
        self, X: np.ndarray, y: np.ndarray, scaled: np.ndarray, target_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recomputes y from the correct target column when it isn't feature index 0."""
        y_correct = scaled[target_indices, self.target_column_index].astype(np.float32)
        return X, y_correct


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)

    df = fetch_and_clean_data("AAPL", "2020-01-01", "2024-01-01")
    print(f"Fetched {len(df)} rows for AAPL. Columns: {list(df.columns)}")
    print(df.head())

    processor = DataProcessor(seq_length=60, forecast_horizon=1, test_split=0.2)
    dataset = processor.prepare(df)
    print(f"X_train: {dataset.X_train.shape} | y_train: {dataset.y_train.shape}")
    print(f"X_test: {dataset.X_test.shape} | y_test: {dataset.y_test.shape}")

    sample_prices = dataset.inverse_transform_target(dataset.y_test[:5])
    print(f"First 5 test targets (inverse-transformed to $): {sample_prices}")
