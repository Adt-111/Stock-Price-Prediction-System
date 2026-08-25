# Stock Price Prediction System — SimpleRNN vs. LSTM with Uncertainty Quantification

An end-to-end time-series forecasting pipeline: fetches historical stock
data from Yahoo Finance, trains both a SimpleRNN and an LSTM to predict the
next day's closing price, compares their accuracy, and quantifies
prediction uncertainty via Monte Carlo Dropout — all explorable through an
interactive Streamlit dashboard.

## Project layout

```
stock_rnn_lstm/
├── requirements.txt
├── app.py                       # Streamlit dashboard
└── src/
    ├── data_processor.py        # fetch_and_clean_data, create_sequences, DataProcessor
    ├── models.py                 # SimpleRNNModel, LSTMModel
    ├── train.py                   # train_model (MSE/MAE logging, early stopping)
    ├── evaluator.py                # RMSE / MAE / MAPE / directional accuracy
    └── uncertainty_engine.py        # predict_with_uncertainty (MC Dropout)
```

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then in the sidebar: enter a ticker (e.g. `AAPL`, `NVDA`, `TSLA`), pick a
date range, tune the sequence window / train-test split / model
hyperparameters, and click **Fetch Data & Train Models**.

## What each stage does

1. **Data (`data_processor.py`)** — `fetch_and_clean_data` pulls daily
   OHLCV data via `yfinance` and forward/backward-fills any missing values
   (holidays, data gaps). `DataProcessor.prepare()` then:
   - Splits the raw price series chronologically (e.g. 80% train / 20%
     test) **before** fitting the scaler, so the `MinMaxScaler(0, 1)` never
     sees test-set values — this avoids look-ahead leakage.
   - Builds sliding-window `(X, y)` sequences via `create_sequences`, and
     assigns each window to train/test based on where its *target* falls
     relative to the split point.
2. **Models (`models.py`)** — `SimpleRNNModel` (RNN → BatchNorm → Dropout →
   Linear) and `LSTMModel` (LSTM → Dropout → Linear), both configurable in
   hidden size, layer count, and dropout rate from the sidebar.
3. **Training (`train.py`)** — Adam + MSE loss, gradient-norm clipping,
   per-epoch MSE/MAE logging on train and validation splits, and early
   stopping that restores the best-validation-MSE weights.
4. **Uncertainty (`uncertainty_engine.py`)** — `predict_with_uncertainty`
   runs N stochastic forward passes with dropout left active at inference
   time (Monte Carlo Dropout), then reports the mean forecast plus the
   2.5th/97.5th percentile bounds (for a 95% interval) across simulations.
5. **Evaluation (`evaluator.py`)** — RMSE, MAE, MAPE, and directional
   accuracy (% of days the predicted up/down move matched the actual
   move), computed on original dollar-scale prices after inverse-transforming
   both the point forecasts and the confidence bounds.

## Working offline / without a Yahoo Finance connection

If `yfinance` can't reach Yahoo Finance (no network access, rate limiting,
an invalid ticker, etc.), `fetch_and_clean_data` automatically falls back
to a synthetic geometric-Brownian-motion price series (with a logged
warning) so the rest of the pipeline — training, evaluation, uncertainty
bands, the dashboard — remains fully runnable for development/demo
purposes. Pass `allow_mock_fallback=False` to instead raise an error when
real data can't be fetched.

## Testing individual stages

Every module is independently runnable and self-demonstrating:

```bash
python -m src.data_processor
python -m src.models
python -m src.train
python -m src.evaluator
python -m src.uncertainty_engine
```
