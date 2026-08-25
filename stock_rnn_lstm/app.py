"""
Pick a ticker and date range, train SimpleRNN and LSTM live, compare their
loss curves, and see predictions with a Monte Carlo Dropout confidence
band overlaid on the actual price.

    streamlit run app.py
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from src.data_processor import DataProcessor, fetch_and_clean_data
from src.evaluator import compare_models, evaluate_model
from src.models import LSTMModel, SimpleRNNModel
from src.train import EarlyStoppingConfig, TrainingHistory, train_model
from src.uncertainty_engine import predict_with_uncertainty

st.set_page_config(page_title="Stock Price Prediction: RNN vs LSTM", layout="wide")


# Cached / session-state helpers

@st.cache_data(show_spinner=False)
def load_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Cached data fetch so re-running the app with the same inputs doesn't re-hit the network."""
    return fetch_and_clean_data(ticker, start_date, end_date)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# Plotting helpers

def plot_loss_curves(history_rnn: TrainingHistory, history_lstm: TrainingHistory):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history_rnn.train_mse, mode="lines", name="SimpleRNN — train MSE",
        line=dict(color="#4C78A8"),
    ))
    fig.add_trace(go.Scatter(
        y=history_rnn.val_mse, mode="lines", name="SimpleRNN — val MSE",
        line=dict(color="#4C78A8", dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        y=history_lstm.train_mse, mode="lines", name="LSTM — train MSE",
        line=dict(color="#F58518"),
    ))
    fig.add_trace(go.Scatter(
        y=history_lstm.val_mse, mode="lines", name="LSTM — val MSE",
        line=dict(color="#F58518", dash="dash"),
    ))
    fig.update_layout(
        title="Training Loss Curves (MSE, scaled units)",
        xaxis_title="Epoch", yaxis_title="MSE",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )
    return fig


def plot_forecast_with_confidence(
    dates: pd.DatetimeIndex,
    actual_prices: np.ndarray,
    predicted_mean: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    model_label: str,
    ticker: str,
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates, y=upper_bound, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=lower_bound, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(76, 120, 168, 0.25)",
        name="95% Confidence Interval", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=actual_prices, mode="lines", name="Actual Price",
        line=dict(color="#2E2E2E", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=predicted_mean, mode="lines", name=f"{model_label} Predicted (MC mean)",
        line=dict(color="#F58518", width=2, dash="dot"),
    ))

    fig.update_layout(
        title=f"{ticker} — Actual vs. Predicted Test Prices ({model_label}) with 95% CI",
        xaxis_title="Date", yaxis_title="Price ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520,
    )
    return fig


# Sidebar controls

def render_sidebar() -> dict:
    st.sidebar.header("Configuration")

    ticker = st.sidebar.text_input("Ticker symbol", value="AAPL", help="e.g. AAPL, NVDA, TSLA").strip().upper()

    default_end = dt.date.today()
    default_start = default_end - dt.timedelta(days=365 * 5)
    date_range = st.sidebar.date_input(
        "Date range", value=(default_start, default_end),
        max_value=default_end,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = default_start, default_end

    seq_length = st.sidebar.slider("Sequence window (days)", min_value=30, max_value=90, value=60, step=5)
    test_split = st.sidebar.slider("Test split ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

    st.sidebar.subheader("Model / Training")
    hidden_size = st.sidebar.slider("Hidden size", min_value=16, max_value=128, value=64, step=16)
    num_layers = st.sidebar.slider("Number of layers", min_value=1, max_value=3, value=2, step=1)
    dropout = st.sidebar.slider("Dropout rate", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    num_epochs = st.sidebar.slider("Max epochs", min_value=10, max_value=200, value=50, step=10)
    batch_size = st.sidebar.select_slider("Batch size", options=[16, 32, 64, 128], value=32)
    patience = st.sidebar.slider("Early stopping patience", min_value=3, max_value=30, value=10)

    st.sidebar.subheader("Uncertainty Quantification")
    num_simulations = st.sidebar.slider("MC Dropout simulations", min_value=20, max_value=300, value=100, step=20)
    confidence_level = st.sidebar.select_slider(
        "Confidence level", options=[0.80, 0.90, 0.95, 0.99], value=0.95
    )

    run_button = st.sidebar.button("Fetch Data & Train Models", type="primary", use_container_width=True)

    return {
        "ticker": ticker,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "seq_length": seq_length,
        "test_split": test_split,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "patience": patience,
        "num_simulations": num_simulations,
        "confidence_level": confidence_level,
        "run_button": run_button,
    }


# Main pipeline (data -> train -> evaluate -> uncertainty)

def run_pipeline(config: dict) -> None:
    set_seed(42)

    with st.spinner(f"Fetching data for {config['ticker']}..."):
        try:
            df = load_data(config["ticker"], config["start_date"], config["end_date"])
        except ValueError as exc:
            st.error(f"Data fetch failed: {exc}")
            return

    st.success(f"Loaded {len(df)} trading days for {config['ticker']} "
               f"({df.index.min().date()} to {df.index.max().date()}).")

    processor = DataProcessor(
        seq_length=config["seq_length"],
        forecast_horizon=1,
        feature_columns=["Close", "Volume"],
        target_column="Close",
        test_split=config["test_split"],
    )
    try:
        dataset = processor.prepare(df)
    except ValueError as exc:
        st.error(f"Data preparation failed: {exc}")
        return

    n_features = dataset.X_train.shape[-1]
    col1, col2 = st.columns(2)
    col1.metric("Training windows", len(dataset.X_train))
    col2.metric("Test windows", len(dataset.X_test))

    rnn_model = SimpleRNNModel(
        input_size=n_features, hidden_size=config["hidden_size"],
        num_layers=config["num_layers"], dropout=config["dropout"],
    )
    lstm_model = LSTMModel(
        input_size=n_features, hidden_size=config["hidden_size"],
        num_layers=config["num_layers"], dropout=config["dropout"],
    )

    early_stopping = EarlyStoppingConfig(patience=config["patience"])

    progress_text = st.empty()
    with st.spinner("Training SimpleRNN..."):
        progress_text.info("Training SimpleRNN...")
        history_rnn = train_model(
            rnn_model, dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test,
            num_epochs=config["num_epochs"], batch_size=config["batch_size"],
            early_stopping=early_stopping, model_name="SimpleRNN", verbose=False,
        )
    with st.spinner("Training LSTM..."):
        progress_text.info("Training LSTM...")
        history_lstm = train_model(
            lstm_model, dataset.X_train, dataset.y_train, dataset.X_test, dataset.y_test,
            num_epochs=config["num_epochs"], batch_size=config["batch_size"],
            early_stopping=early_stopping, model_name="LSTM", verbose=False,
        )
    progress_text.empty()

    st.subheader("Training Loss Curves")
    st.plotly_chart(plot_loss_curves(history_rnn, history_lstm), use_container_width=True)

    epoch_col1, epoch_col2 = st.columns(2)
    epoch_col1.caption(
        f"SimpleRNN: {history_rnn.total_epochs_run} epochs run "
        f"({'stopped early' if history_rnn.stopped_early else 'completed'}), "
        f"best epoch {history_rnn.best_epoch}."
    )
    epoch_col2.caption(
        f"LSTM: {history_lstm.total_epochs_run} epochs run "
        f"({'stopped early' if history_lstm.stopped_early else 'completed'}), "
        f"best epoch {history_lstm.best_epoch}."
    )

    # --- Uncertainty-quantified test-set forecasts for both models --- #
    with st.spinner("Running Monte Carlo Dropout uncertainty estimation..."):
        forecast_rnn = predict_with_uncertainty(
            rnn_model, dataset.X_test,
            num_simulations=config["num_simulations"], confidence_level=config["confidence_level"],
        )
        forecast_lstm = predict_with_uncertainty(
            lstm_model, dataset.X_test,
            num_simulations=config["num_simulations"], confidence_level=config["confidence_level"],
        )

    y_test_dollars = dataset.inverse_transform_target(dataset.y_test)
    rnn_mean_dollars = dataset.inverse_transform_target(forecast_rnn.mean)
    rnn_lower_dollars = dataset.inverse_transform_target(forecast_rnn.lower_bound)
    rnn_upper_dollars = dataset.inverse_transform_target(forecast_rnn.upper_bound)
    lstm_mean_dollars = dataset.inverse_transform_target(forecast_lstm.mean)
    lstm_lower_dollars = dataset.inverse_transform_target(forecast_lstm.lower_bound)
    lstm_upper_dollars = dataset.inverse_transform_target(forecast_lstm.upper_bound)

    previous_actual = np.roll(y_test_dollars, 1)
    previous_actual[0] = y_test_dollars[0]

    metrics_rnn = evaluate_model("SimpleRNN", y_test_dollars, rnn_mean_dollars, previous_actual)
    metrics_lstm = evaluate_model("LSTM", y_test_dollars, lstm_mean_dollars, previous_actual)

    st.subheader("Comparative Evaluation Metrics")
    st.dataframe(compare_models([metrics_rnn, metrics_lstm]), use_container_width=True)

    # Test-set dates: the last len(X_test) dates in the full date index correspond
    # to each test window's target date.
    test_dates = dataset.dates[-len(dataset.X_test):]

    st.subheader("Forecast vs. Actual with 95% Confidence Band")
    tab_lstm, tab_rnn = st.tabs(["LSTM", "SimpleRNN"])
    with tab_lstm:
        st.plotly_chart(
            plot_forecast_with_confidence(
                test_dates, y_test_dollars, lstm_mean_dollars,
                lstm_lower_dollars, lstm_upper_dollars, "LSTM", config["ticker"],
            ),
            use_container_width=True,
        )
    with tab_rnn:
        st.plotly_chart(
            plot_forecast_with_confidence(
                test_dates, y_test_dollars, rnn_mean_dollars,
                rnn_lower_dollars, rnn_upper_dollars, "SimpleRNN", config["ticker"],
            ),
            use_container_width=True,
        )

    st.caption(
        "Confidence bands are estimated via Monte Carlo Dropout: the trained model runs "
        f"{config['num_simulations']} stochastic forward passes per point with dropout left "
        f"active, and the band shows the {config['confidence_level']:.0%} interval across "
        "those simulations. Wider bands indicate the model is less certain about that "
        "prediction."
    )


# App entry point

def main() -> None:
    st.title("Stock Price Prediction: SimpleRNN vs. LSTM with Uncertainty Quantification")
    st.caption(
        "Trains both a SimpleRNN and an LSTM on historical price/volume data, compares their "
        "forecast accuracy, and quantifies prediction uncertainty via Monte Carlo Dropout."
    )

    config = render_sidebar()

    if config["run_button"]:
        run_pipeline(config)
    else:
        st.info("Configure a ticker and settings in the sidebar, then click **Fetch Data & Train Models**.")


if __name__ == "__main__":
    main()
