"""
Data loading and caching module for StockBuddy Forecast.

Handles:
- Fetching S&P 500 constituent list
- Downloading historical OHLCV data via yfinance
- Fetching fundamental data
- Local caching to avoid repeated API calls
"""
import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from src.config import DEFAULT_PERIOD, DEFAULT_INTERVAL, DATA_DIR


def get_sp500_tickers() -> pd.DataFrame:
    """
    Fetch the current S&P 500 constituent list from Wikipedia.

    Returns:
        DataFrame with columns: Symbol, Security, GICS Sector, GICS Sub-Industry
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    sp500_table = tables[0]

    # Clean up ticker symbols (replace dots with dashes for yfinance)
    sp500_table["Symbol"] = sp500_table["Symbol"].str.replace(".", "-", regex=False)

    return sp500_table[["Symbol", "Security", "GICS Sector", "GICS Sub-Industry"]]


def get_cached_path(ticker: str, data_type: str = "ohlcv") -> str:
    """Get the file path for cached data."""
    os.makedirs(DATA_DIR, exist_ok=True)
    return os.path.join(DATA_DIR, f"{ticker}_{data_type}.parquet")


def is_cache_valid(filepath: str, max_age_hours: int = 24) -> bool:
    """Check if cached file exists and is recent enough."""
    if not os.path.exists(filepath):
        return False

    mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    return (datetime.now() - mod_time) < timedelta(hours=max_age_hours)


def fetch_stock_data(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    use_cache: bool = True,
    cache_hours: int = 24,
) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV data for a single ticker.

    Args:
        ticker: Stock ticker symbol
        period: Data period (e.g., '2y', '5y')
        interval: Data interval (e.g., '1d', '1wk')
        use_cache: Whether to use cached data
        cache_hours: Hours before cache expires

    Returns:
        DataFrame with OHLCV data, or None if fetch fails
    """
    cache_path = get_cached_path(ticker)

    # Check cache
    if use_cache and is_cache_valid(cache_path, cache_hours):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass  # Cache corrupted, re-fetch

    # Fetch from yfinance
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)

        if df.empty:
            print(f"  Warning: No data returned for {ticker}")
            return None

        # Clean up columns
        df.index.name = "Date"
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # Cache the data
        if use_cache:
            os.makedirs(DATA_DIR, exist_ok=True)
            df.to_parquet(cache_path)

        return df

    except Exception as e:
        print(f"  Error fetching {ticker}: {e}")
        return None


def fetch_multiple_stocks(
    tickers: List[str],
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple tickers.

    Args:
        tickers: List of ticker symbols
        period: Data period
        interval: Data interval
        use_cache: Whether to use cached data

    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    data = {}
    failed = []

    for ticker in tqdm(tickers, desc="Fetching stock data"):
        df = fetch_stock_data(ticker, period, interval, use_cache)
        if df is not None and not df.empty:
            data[ticker] = df
        else:
            failed.append(ticker)

    if failed:
        print(f"\nFailed to fetch data for {len(failed)} tickers: {failed[:10]}...")

    print(f"Successfully loaded data for {len(data)}/{len(tickers)} tickers")
    return data


def fetch_fundamentals(ticker: str, use_cache: bool = True) -> Optional[Dict]:
    """
    Fetch fundamental data for a ticker via yfinance.

    Returns dict with: sector, industry, market_cap, pe_ratio, forward_pe,
    beta, dividend_yield, 52_week_high, 52_week_low, avg_volume
    """
    cache_path = os.path.join(DATA_DIR, f"{ticker}_fundamentals.json")

    # Check cache
    if use_cache and is_cache_valid(cache_path, max_age_hours=168):  # 1 week cache
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except Exception:
            pass

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        fundamentals = {
            "ticker": ticker,
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "market_cap": info.get("marketCap", None),
            "pe_ratio": info.get("trailingPE", None),
            "forward_pe": info.get("forwardPE", None),
            "beta": info.get("beta", None),
            "dividend_yield": info.get("dividendYield", None),
            "52_week_high": info.get("fiftyTwoWeekHigh", None),
            "52_week_low": info.get("fiftyTwoWeekLow", None),
            "avg_volume": info.get("averageVolume", None),
            "profit_margin": info.get("profitMargins", None),
            "revenue_growth": info.get("revenueGrowth", None),
            "earnings_growth": info.get("earningsGrowth", None),
        }

        # Cache
        if use_cache:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(fundamentals, f, indent=2, default=str)

        return fundamentals

    except Exception as e:
        print(f"  Error fetching fundamentals for {ticker}: {e}")
        return None


def fetch_multiple_fundamentals(
    tickers: List[str], use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch fundamentals for multiple tickers and return as DataFrame.
    """
    records = []

    for ticker in tqdm(tickers, desc="Fetching fundamentals"):
        fund = fetch_fundamentals(ticker, use_cache)
        if fund:
            records.append(fund)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.set_index("ticker")
    return df


def build_price_matrix(
    stock_data: Dict[str, pd.DataFrame],
    column: str = "Close",
) -> pd.DataFrame:
    """
    Build a matrix of prices (or returns) from individual stock DataFrames.

    Args:
        stock_data: Dict mapping ticker -> OHLCV DataFrame
        column: Which price column to use

    Returns:
        DataFrame with dates as index and tickers as columns
    """
    price_series = {}

    for ticker, df in stock_data.items():
        if column in df.columns:
            price_series[ticker] = df[column]

    price_matrix = pd.DataFrame(price_series)
    price_matrix = price_matrix.dropna(how="all")

    return price_matrix


def build_returns_matrix(
    stock_data: Dict[str, pd.DataFrame],
    column: str = "Close",
) -> pd.DataFrame:
    """
    Build a matrix of daily returns from stock data.
    """
    price_matrix = build_price_matrix(stock_data, column)
    returns_matrix = price_matrix.pct_change().dropna()
    return returns_matrix
