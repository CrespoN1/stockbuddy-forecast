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
# Cluster-Informed Forecasters
# =============================================================================

class ClusterEnsembleARIMA:
    """
    Cluster Ensemble ARIMA: trains ARIMA on each cluster peer's price
    series, then averages predictions in return-space.

    Stocks in the same cluster share similar behavior (identified via
    PCA + K-Means). By ensembling forecasts from peer stocks, the model
    leverages cross-stock signal as regularization against overfitting.
    """

    def __init__(self, self_weight: float = 0.5):
        """
        Args:
            self_weight: Weight for the target stock's own ARIMA model.
                         Remaining (1 - self_weight) is split among peers.
        """
        self.self_weight = self_weight
        self.target_model = None
        self.peer_models: Dict[str, object] = {}
        self.last_price = None
        self.name = "Cluster-Ensemble-ARIMA"

    def fit(self, series: pd.Series, peer_series: Optional[Dict[str, pd.Series]] = None):
        """
        Fit ARIMA on target stock and all cluster peers.

        Args:
            series: Target stock's Close price series
            peer_series: Dict of ticker -> Close price series for cluster peers
        """
        self.last_price = series.iloc[-1]

        # Fit on target stock
        self.target_model = ARIMAForecast()
        self.target_model.fit(series)

        # Fit on each peer
        self.peer_models = {}
        if peer_series:
            for ticker, peer_s in peer_series.items():
                try:
                    aligned = peer_s.loc[peer_s.index.isin(series.index)]
                    if len(aligned) >= 252:
                        peer_model = ARIMAForecast()
                        peer_model.fit(aligned)
                        self.peer_models[ticker] = peer_model
                except Exception:
                    continue

        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Predict by weighted-averaging return-space forecasts from all models."""
        if self.target_model is None or self.target_model.model_fit is None:
            return np.full(horizon, np.nan)

        # Collect return-space predictions
        return_preds = []
        weights = []

        # Target stock's own forecast
        try:
            target_forecast = self.target_model.predict(horizon)
            if not np.any(np.isnan(target_forecast)):
                last_fitted = self.target_model.model_fit.data.endog[-1]
                target_returns = target_forecast / last_fitted - 1
                return_preds.append(target_returns)
                weights.append(self.self_weight)
        except (ValueError, AttributeError, RuntimeError):
            pass

        # Peer forecasts — collect successful predictions first, then assign equal weight
        peer_return_preds = []
        for ticker, model in self.peer_models.items():
            try:
                forecast = model.predict(horizon)
                if np.any(np.isnan(forecast)):
                    continue
                last_fitted = model.model_fit.data.endog[-1]
                peer_returns = forecast / last_fitted - 1
                peer_return_preds.append(peer_returns)
            except (ValueError, AttributeError, RuntimeError):
                continue

        # Distribute remaining weight equally among successful peers
        n_successful_peers = len(peer_return_preds)
        if n_successful_peers > 0:
            peer_weight = (1 - self.self_weight) / n_successful_peers
            for pr in peer_return_preds:
                return_preds.append(pr)
                weights.append(peer_weight)

        if not return_preds:
            return np.full(horizon, np.nan)

        # Normalize weights and compute weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        avg_returns = np.average(return_preds, axis=0, weights=weights)

        # Convert back to price level
        return self.last_price * (1 + avg_returns)


class ClusterConcatARIMA:
    """
    Concatenated Returns ARIMA: pools return series from all cluster
    members into a single longer series, fits ARIMA on the pooled
    returns, then maps predictions back to the target stock's price level.

    By concatenating returns, the ARIMA model has more data points to
    estimate autoregressive parameters, assuming cluster members share
    a common return-generating process.
    """

    def __init__(self):
        self.model = None
        self.last_price = None
        self.name = "Cluster-Concat-ARIMA"

    def fit(self, series: pd.Series, peer_series: Optional[Dict[str, pd.Series]] = None):
        """
        Fit ARIMA on concatenated return series from all cluster members.

        Args:
            series: Target stock's Close price series
            peer_series: Dict of ticker -> Close price series for cluster peers
        """
        self.last_price = series.iloc[-1]

        target_returns = series.pct_change().dropna()

        if peer_series and len(peer_series) > 0:
            all_returns = [target_returns]
            for ticker, peer_s in peer_series.items():
                try:
                    peer_returns = peer_s.pct_change().dropna()
                    if len(peer_returns) >= 100:
                        all_returns.append(peer_returns)
                except Exception:
                    continue

            concat_returns = pd.concat(all_returns, ignore_index=True)
        else:
            concat_returns = target_returns.reset_index(drop=True)

        self.model = ARIMAForecast()
        self.model.fit(concat_returns)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Predict returns, then convert to price level."""
        if self.model is None or self.model.model_fit is None:
            return np.full(horizon, np.nan)

        try:
            return_forecast = self.model.predict(horizon)
            prices = [self.last_price]
            for r in return_forecast:
                prices.append(prices[-1] * (1 + r))
            return np.array(prices[1:])
        except Exception:
            return np.full(horizon, np.nan)


class ClusterARIMAWrapper:
    """
    Adapter that conforms to the standard fit(series)/predict(horizon)
    interface for use with backtest_model(). Pre-configured with cluster
    peer data at construction time.
    """

    def __init__(self, stock_data: Dict[str, pd.DataFrame],
                 cluster_labels: pd.Series, target_ticker: str,
                 method: str = "ensemble", self_weight: float = 0.5):
        self.stock_data = stock_data
        self.cluster_labels = cluster_labels
        self.target_ticker = target_ticker
        self.method = method
        self.self_weight = self_weight
        self.inner_model = None
        self.name = f"Cluster-{method.capitalize()}-ARIMA"

    def _get_peer_series(self, train_end_date) -> Dict[str, pd.Series]:
        """Get peer price series truncated to the training period."""
        cluster_id = self.cluster_labels.loc[self.target_ticker]
        peers = self.cluster_labels[self.cluster_labels == cluster_id].index.tolist()
        peers = [t for t in peers if t != self.target_ticker]

        peer_series = {}
        for ticker in peers:
            if ticker in self.stock_data:
                s = self.stock_data[ticker]["Close"]
                peer_series[ticker] = s.loc[s.index <= train_end_date]
        return peer_series

    def fit(self, series: pd.Series):
        """Standard fit interface — uses cluster peers internally."""
        peer_series = self._get_peer_series(series.index[-1])

        if self.method == "ensemble":
            self.inner_model = ClusterEnsembleARIMA(self_weight=self.self_weight)
        else:
            self.inner_model = ClusterConcatARIMA()

        self.inner_model.fit(series, peer_series)
        self.name = self.inner_model.name
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return self.inner_model.predict(horizon)


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
