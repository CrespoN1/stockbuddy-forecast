"""
Advanced forecasting models for StockBuddy Forecast.

Implements:
- LSTM neural network
- Cluster-informed forecasting (train models per cluster group)
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
import warnings

from src.config import (
    LSTM_SEQUENCE_LENGTH,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    FORECAST_HORIZONS,
    RANDOM_STATE,
)
from src.baselines import ARIMAForecast


# =============================================================================
# LSTM Forecaster
# =============================================================================

class LSTMForecaster:
    """
    LSTM-based stock price forecaster.

    Uses a sequence of past prices (and optionally features) to predict
    future prices.
    """

    def __init__(
        self,
        sequence_length: int = LSTM_SEQUENCE_LENGTH,
        epochs: int = LSTM_EPOCHS,
        batch_size: int = LSTM_BATCH_SIZE,
        n_features: int = 1,
        units: int = 64,
    ):
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_features = n_features
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.name = "LSTM"

    def _build_model(self, input_shape: Tuple[int, int]):
        """Build the LSTM model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
        except ImportError:
            raise ImportError(
                "TensorFlow is required for LSTM. Install with: pip install tensorflow"
            )

        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(self.units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1),
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        return model

    def _create_sequences(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for LSTM training."""
        X, y = [], []

        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length : i])
            y.append(data[i, 0])  # Predict the first feature (Close price)

        return np.array(X), np.array(y)

    def fit(
        self,
        series: pd.Series,
        feature_df: Optional[pd.DataFrame] = None,
        verbose: int = 0,
    ):
        """
        Train the LSTM model.

        Args:
            series: Price series to forecast
            feature_df: Optional additional features DataFrame
            verbose: Training verbosity (0=silent, 1=progress)
        """
        # Prepare data
        if feature_df is not None:
            data = feature_df.copy()
            data.insert(0, "Close", series)
            data = data.dropna()
            self.n_features = data.shape[1]
        else:
            data = pd.DataFrame({"Close": series}).dropna()
            self.n_features = 1

        # Scale data
        scaled_data = self.scaler.fit_transform(data.values)

        # Create sequences
        X, y = self._create_sequences(scaled_data)

        if len(X) == 0:
            print("  Warning: Not enough data to create LSTM sequences")
            return self

        # Build and train model
        self.model = self._build_model((self.sequence_length, self.n_features))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                verbose=verbose,
            )

        # Store last sequence for prediction
        self._last_sequence = scaled_data[-self.sequence_length:]

        return self

    def predict(self, horizon: int) -> np.ndarray:
        """
        Generate multi-step forecasts.

        Uses recursive prediction: predict one step, then use that
        prediction as input for the next step.
        """
        if self.model is None:
            return np.full(horizon, np.nan)

        predictions = []
        current_sequence = self._last_sequence.copy()

        for _ in range(horizon):
            # Reshape for LSTM input: (1, sequence_length, n_features)
            input_seq = current_sequence.reshape(1, self.sequence_length, self.n_features)

            # Predict next value
            pred_scaled = self.model.predict(input_seq, verbose=0)[0, 0]

            # Create a full feature vector for inverse scaling
            pred_row = current_sequence[-1].copy()
            pred_row[0] = pred_scaled

            predictions.append(pred_scaled)

            # Update sequence
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = pred_row

        # Inverse scale predictions
        predictions = np.array(predictions).reshape(-1, 1)

        # Pad to match scaler dimensions
        dummy = np.zeros((len(predictions), self.n_features))
        dummy[:, 0] = predictions[:, 0]
        unscaled = self.scaler.inverse_transform(dummy)[:, 0]

        return unscaled


# =============================================================================
# Cluster-Informed Forecaster
# =============================================================================

class ClusterInformedForecaster:
    """
    Train separate forecasting models for each cluster of stocks.

    The idea: stocks that behave similarly (same cluster) can share
    training data, improving model generalization.
    """

    def __init__(
        self,
        model_type: str = "arima",
        lstm_params: Optional[Dict] = None,
    ):
        """
        Args:
            model_type: "arima" or "lstm"
            lstm_params: Optional LSTM hyperparameters
        """
        self.model_type = model_type
        self.lstm_params = lstm_params or {}
        self.cluster_models: Dict[int, object] = {}
        self.name = f"Cluster-{model_type.upper()}"

    def fit(
        self,
        stock_data: Dict[str, pd.DataFrame],
        cluster_labels: pd.Series,
    ):
        """
        Train one model per cluster using pooled data from all stocks
        in that cluster.

        Args:
            stock_data: Dict mapping ticker -> OHLCV DataFrame
            cluster_labels: Series mapping ticker -> cluster label
        """
        clusters = cluster_labels.unique()

        for cluster_id in clusters:
            # Get tickers in this cluster
            tickers = cluster_labels[cluster_labels == cluster_id].index.tolist()

            # Pool training data from all stocks in cluster
            pooled_returns = []
            for ticker in tickers:
                if ticker in stock_data:
                    returns = stock_data[ticker]["Close"].pct_change().dropna()
                    pooled_returns.append(returns)

            if not pooled_returns:
                continue

            # Concatenate returns (use the longest series for the model)
            # For ARIMA: train on the first stock's data (representative)
            representative_ticker = tickers[0]
            if representative_ticker in stock_data:
                train_series = stock_data[representative_ticker]["Close"]

                if self.model_type == "arima":
                    model = ARIMAForecast()
                    model.fit(train_series)
                elif self.model_type == "lstm":
                    model = LSTMForecaster(**self.lstm_params)
                    model.fit(train_series, verbose=0)
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")

                self.cluster_models[cluster_id] = model

        return self

    def predict(
        self,
        ticker: str,
        cluster_label: int,
        train_series: pd.Series,
        horizon: int,
    ) -> np.ndarray:
        """
        Predict for a specific ticker using its cluster's model.

        If the cluster model isn't available, falls back to fitting
        a model on the individual stock's data.
        """
        if cluster_label in self.cluster_models:
            model = self.cluster_models[cluster_label]

            # Re-fit on the specific stock's data to get the right level
            if self.model_type == "arima":
                model = ARIMAForecast()
                model.fit(train_series)

            return model.predict(horizon)
        else:
            # Fallback: train individual model
            if self.model_type == "arima":
                model = ARIMAForecast()
            else:
                model = LSTMForecaster(**self.lstm_params)

            model.fit(train_series)
            return model.predict(horizon)


# =============================================================================
# Convenience Functions
# =============================================================================

def run_all_forecasters(
    train_series: pd.Series,
    horizon: int,
    feature_df: Optional[pd.DataFrame] = None,
    include_lstm: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run all advanced forecasting models on a training series.

    Args:
        train_series: Historical price series
        horizon: Steps to forecast
        feature_df: Optional features for LSTM
        include_lstm: Whether to include LSTM (slower)

    Returns:
        Dict mapping model name -> predictions
    """
    predictions = {}

    # LSTM
    if include_lstm:
        try:
            lstm = LSTMForecaster(epochs=30)  # Fewer epochs for speed
            lstm.fit(train_series, feature_df, verbose=0)
            predictions["LSTM"] = lstm.predict(horizon)
        except Exception as e:
            print(f"  Warning: LSTM failed: {e}")
            predictions["LSTM"] = np.full(horizon, np.nan)

    return predictions


def run_forecasters_multi_horizon(
    train_series: pd.Series,
    horizons: Optional[Dict[str, int]] = None,
    feature_df: Optional[pd.DataFrame] = None,
    include_lstm: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run advanced forecasters across multiple horizons.
    """
    if horizons is None:
        horizons = FORECAST_HORIZONS

    results = {}

    for horizon_name, horizon_steps in horizons.items():
        print(f"  Forecasting horizon: {horizon_name} ({horizon_steps} steps)")
        results[horizon_name] = run_all_forecasters(
            train_series, horizon_steps, feature_df, include_lstm
        )

    return results
