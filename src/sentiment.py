"""
Sentiment analysis module for StockBuddy Forecast.

Fetches news headlines and computes sentiment scores using VADER.
Supports fallback to finviz scraping if NewsAPI key is not available.
"""
import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from tqdm import tqdm

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: vaderSentiment not installed. Run: pip install vaderSentiment")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

from src.config import NEWS_API_KEY, DATA_DIR


def get_vader_analyzer():
    """Get VADER sentiment analyzer instance."""
    if not VADER_AVAILABLE:
        raise ImportError("vaderSentiment is required. Install with: pip install vaderSentiment")
    return SentimentIntensityAnalyzer()


def analyze_text_sentiment(text: str, analyzer=None) -> Dict[str, float]:
    """
    Analyze sentiment of a text string using VADER.

    Returns:
        Dict with 'neg', 'neu', 'pos', 'compound' scores.
        compound ranges from -1 (most negative) to +1 (most positive).
    """
    if analyzer is None:
        analyzer = get_vader_analyzer()

    return analyzer.polarity_scores(text)


def fetch_news_newsapi(
    ticker: str,
    days_back: int = 30,
    max_articles: int = 50,
) -> List[Dict]:
    """
    Fetch news headlines from NewsAPI.
    Requires a NEWS_API_KEY in .env file.
    """
    if not NEWS_API_KEY:
        return []

    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": ticker,
        "from": from_date,
        "sortBy": "relevancy",
        "pageSize": max_articles,
        "language": "en",
        "apiKey": NEWS_API_KEY,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            return [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "published": a.get("publishedAt", ""),
                    "source": a.get("source", {}).get("name", ""),
                }
                for a in articles
                if a.get("title")
            ]
    except Exception as e:
        print(f"  NewsAPI error for {ticker}: {e}")

    return []


def fetch_news_finviz(ticker: str) -> List[Dict]:
    """
    Scrape news headlines from finviz as a free fallback.
    """
    if not BS4_AVAILABLE:
        return []

    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            news_table = soup.find(id="news-table")

            if news_table is None:
                return []

            articles = []
            rows = news_table.find_all("tr")

            for row in rows[:30]:  # Limit to 30 headlines
                a_tag = row.find("a")
                if a_tag:
                    title = a_tag.get_text(strip=True)
                    td_tag = row.find("td")
                    date_str = td_tag.get_text(strip=True) if td_tag else ""

                    articles.append({
                        "title": title,
                        "description": "",
                        "published": date_str,
                        "source": "finviz",
                    })

            return articles

    except Exception as e:
        print(f"  Finviz scraping error for {ticker}: {e}")

    return []


def get_news_headlines(ticker: str, days_back: int = 30) -> List[Dict]:
    """
    Get news headlines for a ticker.
    Tries NewsAPI first, falls back to finviz scraping.
    """
    # Try NewsAPI first
    articles = fetch_news_newsapi(ticker, days_back)

    # Fallback to finviz
    if not articles:
        articles = fetch_news_finviz(ticker)

    return articles


def compute_ticker_sentiment(
    ticker: str,
    days_back: int = 30,
    analyzer=None,
) -> Dict[str, float]:
    """
    Compute aggregate sentiment score for a ticker based on recent news.

    Returns:
        Dict with:
        - 'compound_mean': Average compound sentiment
        - 'compound_std': Std dev of compound sentiment
        - 'positive_ratio': Fraction of positive headlines
        - 'negative_ratio': Fraction of negative headlines
        - 'num_articles': Number of articles analyzed
    """
    if analyzer is None:
        analyzer = get_vader_analyzer()

    articles = get_news_headlines(ticker, days_back)

    if not articles:
        return {
            "compound_mean": 0.0,
            "compound_std": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "num_articles": 0,
        }

    compounds = []
    for article in articles:
        text = article.get("title", "")
        if article.get("description"):
            text += " " + article["description"]

        if text.strip():
            scores = analyzer.polarity_scores(text)
            compounds.append(scores["compound"])

    if not compounds:
        return {
            "compound_mean": 0.0,
            "compound_std": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "num_articles": 0,
        }

    compounds = np.array(compounds)

    return {
        "compound_mean": float(compounds.mean()),
        "compound_std": float(compounds.std()),
        "positive_ratio": float((compounds > 0.05).mean()),
        "negative_ratio": float((compounds < -0.05).mean()),
        "num_articles": len(compounds),
    }


def compute_batch_sentiment(
    tickers: List[str],
    days_back: int = 30,
) -> pd.DataFrame:
    """
    Compute sentiment scores for multiple tickers.

    Returns:
        DataFrame with tickers as index and sentiment features as columns.
    """
    analyzer = get_vader_analyzer()
    records = []

    for ticker in tqdm(tickers, desc="Computing sentiment"):
        sentiment = compute_ticker_sentiment(ticker, days_back, analyzer)
        sentiment["ticker"] = ticker
        records.append(sentiment)

    df = pd.DataFrame(records)
    df = df.set_index("ticker")
    return df
