"""
Feature engineering module for StockBuddy Forecast.

Computes technical indicators, return-based features, and fundamental features
for use in PCA, clustering, and forecasting models.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from src.config import TECHNICAL_WINDOWS


# =============================================================================
# Technical Indicators
# =============================================================================

def compute_sma(series: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=window).mean()


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (RSI).
    Returns values between 0 and 100.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Moving Average Convergence Divergence (MACD).
    Returns DataFrame with MACD line, signal line, and histogram.
    """
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return pd.DataFrame({
        "MACD": macd_line,
        "MACD_Signal": signal_line,
        "MACD_Hist": histogram,
    })


def compute_bollinger_bands(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands.
    Returns upper band, middle band, lower band, and bandwidth.
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()

    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    bandwidth = (upper - lower) / middle
    pct_b = (series - lower) / (upper - lower)

    return pd.DataFrame({
        "BB_Upper": upper,
        "BB_Middle": middle,
        "BB_Lower": lower,
        "BB_Bandwidth": bandwidth,
        "BB_PctB": pct_b,
    })


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (ATR) - measures volatility.
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV).
    """
    close = df["Close"]
    volume = df["Volume"]

    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    obv = (volume * direction).cumsum()
    return pd.Series(obv, index=df.index, name="OBV")


# =============================================================================
# Return-Based Features
# =============================================================================

def compute_returns(series: pd.Series, periods: List[int] = None) -> pd.DataFrame:
    """
    Compute returns over multiple periods.

    Args:
        series: Price series
        periods: List of lookback periods (default: [1, 5, 21])

    Returns:
        DataFrame with return columns
    """
    if periods is None:
        periods = [1, 5, 21]  # Daily, weekly, monthly

    returns = {}
    for p in periods:
        returns[f"Return_{p}d"] = series.pct_change(periods=p)

    return pd.DataFrame(returns)


def compute_volatility(series: pd.Series, windows: List[int] = None) -> pd.DataFrame:
    """
    Compute rolling volatility (standard deviation of returns).
    """
    if windows is None:
        windows = [5, 21, 63]  # Weekly, monthly, quarterly

    daily_returns = series.pct_change()

    vol = {}
    for w in windows:
        vol[f"Volatility_{w}d"] = daily_returns.rolling(window=w).std() * np.sqrt(252)

    return pd.DataFrame(vol)


def compute_momentum(series: pd.Series, periods: List[int] = None) -> pd.DataFrame:
    """
    Compute momentum indicators (price relative to past price).
    """
    if periods is None:
        periods = [5, 10, 21, 63]

    momentum = {}
    for p in periods:
        momentum[f"Momentum_{p}d"] = series / series.shift(p) - 1

    return pd.DataFrame(momentum)


# =============================================================================
# Main Feature Engineering Pipeline
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full feature engineering pipeline to a single stock's OHLCV data.

    Args:
        df: DataFrame with Open, High, Low, Close, Volume columns

    Returns:
        DataFrame with all engineered features (original columns + new features)
    """
    features = df.copy()
    close = df["Close"]

    # --- Technical Indicators ---

    # Moving averages
    for name, window in [("SMA_20", 20), ("SMA_50", 50), ("SMA_200", 200)]:
        features[name] = compute_sma(close, window)
        # Price relative to MA
        features[f"Price_vs_{name}"] = close / features[name] - 1

    # RSI
    features["RSI"] = compute_rsi(close, TECHNICAL_WINDOWS["rsi_period"])

    # MACD
    macd = compute_macd(
        close,
        TECHNICAL_WINDOWS["macd_fast"],
        TECHNICAL_WINDOWS["macd_slow"],
        TECHNICAL_WINDOWS["macd_signal"],
    )
    features = pd.concat([features, macd], axis=1)

    # Bollinger Bands
    bb = compute_bollinger_bands(
        close,
        TECHNICAL_WINDOWS["bollinger_period"],
        TECHNICAL_WINDOWS["bollinger_std"],
    )
    features = pd.concat([features, bb], axis=1)

    # ATR
    features["ATR"] = compute_atr(df, TECHNICAL_WINDOWS["atr_period"])
    features["ATR_Pct"] = features["ATR"] / close  # ATR as percentage of price

    # OBV
    features["OBV"] = compute_obv(df)
    features["OBV_SMA"] = compute_sma(features["OBV"], 20)

    # --- Return-Based Features ---
    returns = compute_returns(close)
    features = pd.concat([features, returns], axis=1)

    # Volatility
    vol = compute_volatility(close)
    features = pd.concat([features, vol], axis=1)

    # Momentum
    mom = compute_momentum(close)
    features = pd.concat([features, mom], axis=1)

    # --- Volume Features ---
    features["Volume_SMA_20"] = compute_sma(df["Volume"], 20)
    features["Volume_Ratio"] = df["Volume"] / features["Volume_SMA_20"]

    # --- Price Pattern Features ---
    features["Daily_Range"] = (df["High"] - df["Low"]) / close
    features["Gap"] = df["Open"] / close.shift(1) - 1

    return features


def create_feature_matrix(
    stock_data: Dict[str, pd.DataFrame],
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Create a cross-sectional feature matrix for all stocks (one row per stock).
    Uses the latest available data point (or a specified date) to create a snapshot.

    This is useful for PCA and clustering (comparing stocks against each other).

    Args:
        stock_data: Dict mapping ticker -> OHLCV DataFrame
        as_of_date: Optional date string to use as snapshot date

    Returns:
        DataFrame with tickers as index and features as columns
    """
    snapshots = []

    for ticker, df in stock_data.items():
        try:
            features = engineer_features(df)

            if as_of_date:
                # Get the closest date
                idx = features.index.get_indexer([pd.Timestamp(as_of_date)], method="ffill")
                if idx[0] >= 0:
                    snapshot = features.iloc[idx[0]]
                else:
                    snapshot = features.iloc[-1]
            else:
                snapshot = features.iloc[-1]

            snapshot.name = ticker

            # Select only numeric features for the matrix
            numeric_snapshot = snapshot[snapshot.apply(lambda x: isinstance(x, (int, float, np.number)))]
            snapshots.append(numeric_snapshot)

        except Exception as e:
            print(f"  Error engineering features for {ticker}: {e}")
            continue

    if not snapshots:
        return pd.DataFrame()

    matrix = pd.DataFrame(snapshots)

    # Drop columns with too many NaN values
    nan_threshold = len(matrix) * 0.5
    matrix = matrix.dropna(axis=1, thresh=int(nan_threshold))

    # Fill remaining NaN with column median
    matrix = matrix.fillna(matrix.median())

    return matrix


def create_lagged_features(
    df: pd.DataFrame,
    target_col: str = "Close",
    n_lags: int = 5,
) -> pd.DataFrame:
    """
    Create lagged features for time series forecasting.

    Args:
        df: Feature DataFrame
        target_col: Target column name
        n_lags: Number of lag periods

    Returns:
        DataFrame with lagged features added
    """
    result = df.copy()

    for lag in range(1, n_lags + 1):
        result[f"{target_col}_Lag_{lag}"] = df[target_col].shift(lag)
        result[f"Return_Lag_{lag}"] = df[target_col].pct_change().shift(lag)

    return result
