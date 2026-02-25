"""
Configuration and constants for StockBuddy Forecast.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys (optional - set in .env file)
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

# Data Settings
DEFAULT_PERIOD = "2y"  # 2 years of historical data
DEFAULT_INTERVAL = "1d"  # Daily data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")

# Feature Engineering
TECHNICAL_WINDOWS = {
    "sma_short": 20,
    "sma_medium": 50,
    "sma_long": 200,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bollinger_period": 20,
    "bollinger_std": 2,
    "atr_period": 14,
}

# PCA Settings
PCA_VARIANCE_THRESHOLD = 0.90  # Keep components explaining 90% variance

# Clustering Settings
MAX_CLUSTERS = 15
RANDOM_STATE = 42

# Forecasting Settings
FORECAST_HORIZONS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
}

# LSTM Settings
LSTM_SEQUENCE_LENGTH = 60  # 60 trading days lookback
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# Evaluation Settings
TEST_SIZE = 0.2  # 20% of data for testing
WALK_FORWARD_WINDOWS = 5  # Number of walk-forward validation windows
