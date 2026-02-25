# StockBuddy Forecast

A machine learning pipeline for stock price forecasting using PCA, clustering, and time series models. Built as an ISyE 6740 (Computational Data Analysis) final project.

## Overview

StockBuddy Forecast applies unsupervised learning techniques (PCA for dimensionality reduction, K-Means and Gaussian Mixture Models for clustering) combined with time series forecasting models (ARIMA, LSTM) to predict stock prices across multiple horizons.

The pipeline:
1. Collects multi-source data (price, fundamentals, news sentiment) for S&P 500 stocks
2. Engineers technical, fundamental, and sentiment features
3. Reduces dimensionality with PCA and clusters stocks by behavior
4. Trains forecasting models (with cluster-informed variants)
5. Evaluates against baselines across 1-day, 1-week, and 1-month horizons

## Project Structure

```
stockbuddy-forecast/
├── data/raw/              # Downloaded stock data (gitignored)
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_eda_and_pca.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_forecasting_models.ipynb
│   ├── 06_evaluation.ipynb
│   └── 07_results_visualization.ipynb
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── sentiment.py
│   ├── pca_analysis.py
│   ├── clustering.py
│   ├── forecasters.py
│   ├── baselines.py
│   ├── evaluation.py
│   └── config.py
├── figures/               # Report-ready visualizations
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/stockbuddy-forecast.git
cd stockbuddy-forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (optional - for news sentiment)
```

## Usage

Run the notebooks in order (01 through 07). Each notebook imports from `src/` modules and builds on the previous step.

```bash
jupyter notebook
```

## Methods

- **PCA**: Dimensionality reduction on multi-feature stock data
- **K-Means & GMM**: Clustering stocks by behavioral similarity
- **ARIMA/SARIMA**: Classical time series forecasting
- **LSTM**: Deep learning sequential model
- **Cluster-Informed Forecasting**: Training models per cluster group

## Team

- Berto Crespo
- [Team Member 2]

## Course

ISyE 6740 — Computational Data Analysis (Georgia Tech)
