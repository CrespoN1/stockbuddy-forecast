"""
Baseline forecasting models for StockBuddy Forecast.

Implements simple baselines to benchmark against more sophisticated models:
- Naive (last value) forecast
- Random walk
- Simple Moving Average forecast
- ARIMA
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings

from src.config import FORECAST_HORIZONS


class NaiveForecast:
    """
    Naive forecast: predict the last known value for all future periods.
    """

    def __init__(self):
        self.last_value = None
        self.name = "Naive"

    def fit(self, series: pd.Series):
        self.last_value = series.iloc[-1]
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.last_value)


class RandomWalkForecast:
    """
    Random walk: each step is the previous value plus random noise
    based on historical volatility.
    """

    def __init__(self, random_state: int = 42):
        self.last_value = None
        self.daily_std = None
        self.rng = np.random.RandomState(random_state)
        self.name = "Random Walk"

    def fit(self, series: pd.Series):
        self.last_value = series.iloc[-1]
        returns = series.pct_change().dropna()
        self.daily_std = returns.std()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        predictions = [self.last_value]
        for _ in range(horizon):
            noise = self.rng.normal(0, self.daily_std)
            next_val = predictions[-1] * (1 + noise)
            predictions.append(next_val)
        return np.array(predictions[1:])


class SMAForecast:
    """
    Simple Moving Average forecast: predict using the average
    of the last `window` values.
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.sma_value = None
        self.name = f"SMA({window})"

    def fit(self, series: pd.Series):
        self.sma_value = series.iloc[-self.window:].mean()
        return self

    def predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.sma_value)


class ARIMAForecast:
    """
    ARIMA time series model.
    Auto-selects differencing order based on stationarity test.
    """

    def __init__(self, order: Optional[Tuple[int, int, int]] = None, max_d: int = 2):
        self.order = order
        self.max_d = max_d
        self.model_fit = None
        self.name = "ARIMA"

    def _determine_d(self, series: pd.Series) -> int:
        """Determine differencing order using ADF test."""
        for d in range(self.max_d + 1):
            if d == 0:
                test_series = series
            else:
                test_series = series.diff(d).dropna()

            try:
                result = adfuller(test_series, autolag="AIC")
                if result[1] < 0.05:  # Stationary at 5% significance
                    return d
            except Exception:
                continue

        return 1  # Default to d=1

    def fit(self, series: pd.Series):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.order is None:
                d = self._determine_d(series)
                self.order = (5, d, 1)  # Reasonable defaults

            try:
                model = ARIMA(series, order=self.order)
                self.model_fit = model.fit()
                self.name = f"ARIMA{self.order}"
            except Exception as e:
                # Fallback to simpler model
                try:
                    model = ARIMA(series, order=(1, 1, 0))
                    self.model_fit = model.fit()
                    self.name = "ARIMA(1,1,0)"
                except Exception:
                    self.model_fit = None

        return self

    def predict(self, horizon: int) -> np.ndarray:
        if self.model_fit is None:
            return np.full(horizon, np.nan)

        try:
            forecast = self.model_fit.forecast(steps=horizon)
            return forecast.values
        except Exception:
            return np.full(horizon, np.nan)


def run_all_baselines(
    train_series: pd.Series,
    horizon: int,
) -> Dict[str, np.ndarray]:
    """
    Run all baseline models on a training series and return predictions.

    Args:
        train_series: Historical price series for training
        horizon: Number of steps to forecast

    Returns:
        Dict mapping model name -> prediction array
    """
    models = [
        NaiveForecast(),
        RandomWalkForecast(),
        SMAForecast(window=20),
        SMAForecast(window=50),
        ARIMAForecast(),
    ]

    predictions = {}

    for model in models:
        try:
            model.fit(train_series)
            preds = model.predict(horizon)
            predictions[model.name] = preds
        except Exception as e:
            print(f"  Warning: {model.name} failed: {e}")
            predictions[model.name] = np.full(horizon, np.nan)

    return predictions


def run_baselines_multi_horizon(
    train_series: pd.Series,
    horizons: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Run baselines across multiple forecast horizons.

    Returns:
        Nested dict: horizon_name -> model_name -> predictions
    """
    if horizons is None:
        horizons = FORECAST_HORIZONS

    results = {}

    for horizon_name, horizon_steps in horizons.items():
        results[horizon_name] = run_all_baselines(train_series, horizon_steps)

    return results
