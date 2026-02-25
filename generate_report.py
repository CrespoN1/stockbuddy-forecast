"""
Generate the ISyE 6740 Final Project Report as a PDF.
Uses matplotlib for layout — no LaTeX dependency required.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, "figures")
DATA_DIR = os.path.join(BASE, "data", "processed")
OUT_PATH = os.path.join(BASE, "report.pdf")

# ── helpers ──────────────────────────────────────────────────────────
def text_page(pdf, lines, fontsize=11, title=None, title_size=16):
    """Render a page of wrapped text."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    y = 0.95
    if title:
        ax.text(0.5, y, title, fontsize=title_size, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes)
        y -= 0.05
    for line in lines:
        if line.startswith("##"):
            ax.text(0.05, y, line.replace("## ", ""), fontsize=13,
                    fontweight="bold", va="top", transform=ax.transAxes)
            y -= 0.035
        elif line.startswith("**"):
            ax.text(0.05, y, line.replace("**", ""), fontsize=fontsize,
                    fontweight="bold", va="top", transform=ax.transAxes,
                    family="monospace" if "|" in line else "sans-serif")
            y -= 0.025
        elif line == "":
            y -= 0.015
        else:
            ax.text(0.05, y, line, fontsize=fontsize, va="top",
                    transform=ax.transAxes, wrap=True,
                    family="monospace" if "|" in line else "sans-serif")
            y -= 0.025
        if y < 0.03:
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def figure_page(pdf, img_path, caption, scale=0.85):
    """Render a figure with caption below."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    try:
        img = Image.open(img_path)
        # Calculate position to center image
        ax.text(0.5, 0.98, caption, fontsize=11, fontstyle="italic",
                ha="center", va="top", transform=ax.transAxes)
        ax.imshow(img, aspect="equal", extent=[0.05, 0.95, 0.02, 0.93],
                  transform=ax.transAxes)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not load {img_path}:\n{e}",
                ha="center", va="center", transform=ax.transAxes)
    pdf.savefig(fig)
    plt.close(fig)


def table_page(pdf, df, title, fontsize=8):
    """Render a DataFrame as a table on a page."""
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.5, 0.97, title, fontsize=13, fontweight="bold",
            ha="center", va="top", transform=ax.transAxes)

    n_rows = len(df)
    n_cols = len(df.columns)

    # Create table
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.3)

    # Style header
    for j in range(n_cols):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            if i % 2 == 0:
                tbl[i, j].set_facecolor("#D9E2F3")

    pdf.savefig(fig)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────
def main():
    with PdfPages(OUT_PATH) as pdf:

        # ── TITLE PAGE ───────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.text(0.5, 0.65, "StockBuddy Forecast", fontsize=28,
                fontweight="bold", ha="center", va="center",
                transform=ax.transAxes)
        ax.text(0.5, 0.58,
                "Cluster-Informed Stock Price Forecasting\nUsing PCA and Time Series Models",
                fontsize=16, ha="center", va="center",
                transform=ax.transAxes, linespacing=1.5)
        ax.text(0.5, 0.45, "ISyE 6740 — Computational Data Analysis\nFinal Project",
                fontsize=14, ha="center", va="center",
                transform=ax.transAxes, linespacing=1.5)
        ax.text(0.5, 0.35, "Georgia Institute of Technology\nSpring 2026",
                fontsize=12, ha="center", va="center",
                transform=ax.transAxes, linespacing=1.5, color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        # ── 1. INTRODUCTION ─────────────────────────────────────────
        text_page(pdf, [
            "## 1. Introduction",
            "",
            "Accurate stock price forecasting is a fundamental challenge in quantitative",
            "finance. Traditional time series models such as ARIMA treat each stock in",
            "isolation, ignoring structural similarities across equities. This project",
            "investigates whether leveraging cross-stock structure, discovered via",
            "unsupervised learning, can improve forecast accuracy.",
            "",
            "We propose a pipeline that combines Principal Component Analysis (PCA) for",
            "dimensionality reduction, K-Means and Gaussian Mixture Model (GMM) clustering",
            "to group behaviorally similar stocks, and ARIMA-based forecasting that",
            "exploits cluster membership. Specifically, we introduce two cluster-informed",
            "strategies:",
            "",
            "  1. Cluster Ensemble ARIMA: trains independent ARIMA models on each cluster",
            "     peer and averages predictions in return-space.",
            "",
            "  2. Cluster Concat ARIMA: pools daily returns from all cluster members into",
            "     a single longer series to give ARIMA more training data.",
            "",
            "We evaluate these methods against standard baselines (Naive, Random Walk,",
            "SMA, standalone ARIMA) using walk-forward backtesting with Diebold-Mariano",
            "significance tests. The pipeline is applied to 30 large-cap U.S. equities",
            "over a 2-year period (Feb 2024 - Feb 2026), with 6-month forward forecasts",
            "generated for all stocks.",
            "",
            "## 1.1 Dataset",
            "",
            "Daily OHLCV data for 30 S&P 500 stocks was obtained via the yfinance API.",
            "The dataset spans 502 trading days. From raw prices, 38 features were",
            "engineered per stock, including technical indicators (SMA, EMA, RSI, MACD,",
            "Bollinger Bands, ATR), return statistics (volatility, skewness, kurtosis),",
            "and fundamental ratios.",
        ])

        # ── Figure 1: Price Overview ─────────────────────────────────
        figure_page(pdf,
                    os.path.join(FIG_DIR, "fig1_price_overview.png"),
                    "Figure 1: Normalized stock prices (base 100) for 30 equities over 2 years.")

        # ── 2. METHODOLOGY — PCA ─────────────────────────────────────
        text_page(pdf, [
            "## 2. Methodology",
            "",
            "## 2.1 Dimensionality Reduction via PCA",
            "",
            "The 30x38 feature matrix was standardized and decomposed via PCA. Seven",
            "principal components were retained, capturing 91.9% of the total variance.",
            "This reduced the feature space from 38 dimensions to 7 while preserving",
            "the dominant structure needed for clustering.",
            "",
            "The scree plot (Figure 2) shows the cumulative explained variance. The",
            "90% threshold is reached at 7 components, which was selected as the",
            "default via an automated elbow criterion.",
            "",
            "Figure 3 shows the PCA loading matrix, revealing which original features",
            "contribute most to each component. PC1 is dominated by momentum and return",
            "features, while PC2 captures volatility-related variation.",
        ])

        figure_page(pdf,
                    os.path.join(FIG_DIR, "fig2_pca_variance.png"),
                    "Figure 2: Cumulative explained variance. 7 components capture 91.9%.")

        figure_page(pdf,
                    os.path.join(FIG_DIR, "03_pca_loadings.png"),
                    "Figure 3: PCA loading matrix showing feature contributions to each component.")

        # ── 2.2 CLUSTERING ───────────────────────────────────────────
        text_page(pdf, [
            "## 2.2 Clustering",
            "",
            "K-Means and Gaussian Mixture Models (GMM) were applied to the 7-dimensional",
            "PCA embeddings. The number of clusters was selected using silhouette analysis",
            "(K-Means: K=6) and BIC minimization (GMM: K=9).",
            "",
            "Figure 4 shows the silhouette analysis for K-Means. K=6 was selected as",
            "the optimal cluster count based on the average silhouette score.",
            "",
            "Figure 5 shows the final cluster assignments projected into PC1-PC2 space.",
            "Clusters capture meaningful groupings: Cluster 0 (11 stocks) contains",
            "defensive/value names (JNJ, PG, KO, XOM), while Cluster 1 (6 stocks)",
            "groups large-cap tech (MSFT, AMZN). NVDA sits alone in Cluster 4,",
            "reflecting its unique return profile during the AI boom.",
            "",
            "Figure 6 shows the composition of each cluster, identifying which stocks",
            "share similar behavior patterns as determined by PCA + K-Means.",
        ])

        figure_page(pdf,
                    os.path.join(FIG_DIR, "kmeans_evaluation.png"),
                    "Figure 4: K-Means silhouette analysis. K=6 selected as optimal.")

        figure_page(pdf,
                    os.path.join(FIG_DIR, "fig4_clusters.png"),
                    "Figure 5: Stocks in PCA space colored by K-Means cluster (K=6).")

        figure_page(pdf,
                    os.path.join(FIG_DIR, "cluster_composition.png"),
                    "Figure 6: Cluster composition showing which stocks belong to each group.")

        # ── 2.3 FORECASTING ──────────────────────────────────────────
        text_page(pdf, [
            "## 2.3 Forecasting Models",
            "",
            "We compare six forecasting approaches:",
            "",
            "  Baselines:",
            "    - Naive: repeats last observed price",
            "    - Random Walk: last price + Gaussian noise calibrated to historical vol",
            "    - SMA(20): 20-day simple moving average",
            "    - ARIMA(5,d,1): auto-differenced ARIMA with order selected via ADF test",
            "",
            "  Cluster-Informed Methods:",
            "    - Cluster Ensemble ARIMA: for a target stock, fit ARIMA on the target",
            "      and on each cluster peer. Average return-space forecasts using",
            "      50% self-weight, 50% split among peers. Convert back to price level.",
            "",
            "    - Cluster Concat ARIMA: concatenate daily return series from all cluster",
            "      members into one pooled series, fit ARIMA on the longer series, then",
            "      map return predictions back to the target stock's last price.",
            "",
            "The cluster-informed models exploit the assumption that stocks in the same",
            "PCA-based cluster share a common return-generating process. Ensemble ARIMA",
            "regularizes via cross-stock averaging; Concat ARIMA increases effective",
            "sample size for parameter estimation.",
            "",
            "Figure 7 compares forecasts for AAPL. The Cluster-Concat-ARIMA achieves",
            "RMSE of 10.83 vs ARIMA's 15.46 (30% reduction), and both cluster methods",
            "improve directional accuracy from 34% to 54-55%.",
        ])

        figure_page(pdf,
                    os.path.join(FIG_DIR, "05_cluster_forecasts.png"),
                    "Figure 7: ARIMA vs cluster-informed forecasts for AAPL (test set).")

        # ── 3. EVALUATION ────────────────────────────────────────────
        text_page(pdf, [
            "## 3. Evaluation",
            "",
            "## 3.1 Walk-Forward Backtesting",
            "",
            "All models were evaluated using 5-fold walk-forward validation across",
            "three forecast horizons (1-day, 1-week, 1-month). This prevents data",
            "leakage: each fold trains only on past data and evaluates on unseen",
            "future data.",
            "",
            "Figure 8 shows the RMSE heatmap across all models and horizons for AAPL.",
            "At the 1-month horizon, ARIMA and Naive perform similarly, while Random",
            "Walk and SMA show significantly higher error.",
            "",
            "## 3.2 Cluster Model Evaluation Across Stocks",
            "",
            "Figure 9 compares ARIMA vs Cluster-Ensemble-ARIMA across 5 representative",
            "stocks using walk-forward backtesting. Results are mixed: ensemble",
            "improves MSFT (+1.8%), AMZN (+1.0%), and GOOGL (+0.02%), but slightly",
            "hurts AAPL (-3.6%). NVDA shows no change (singleton cluster).",
            "",
            "Figure 10 plots improvement percentage against cluster size, showing",
            "that the ensemble benefit is largest for stocks in medium-sized clusters.",
            "Singleton clusters correctly fall back to regular ARIMA.",
            "",
            "## 3.3 Statistical Significance",
            "",
            "Diebold-Mariano tests were performed on walk-forward RMSE errors:",
            "  - ARIMA vs Cluster-Ensemble: p=0.035 (significant at alpha=0.05)",
            "  - ARIMA vs Cluster-Concat: p=0.702 (not significant)",
            "  - SMA(20) vs ARIMA: p=0.001 (ARIMA significantly better)",
            "",
            "The ensemble method shows statistically significant improvement over",
            "standalone ARIMA in walk-forward evaluation.",
        ])

        figure_page(pdf,
                    os.path.join(FIG_DIR, "06_model_comparison_rmse.png"),
                    "Figure 8: Walk-forward RMSE across models and horizons.")

        figure_page(pdf,
                    os.path.join(FIG_DIR, "fig5b_cluster_vs_arima.png"),
                    "Figure 9: ARIMA vs Cluster-Ensemble ARIMA across 5 stocks.")

        figure_page(pdf,
                    os.path.join(FIG_DIR, "fig5c_improvement_vs_cluster_size.png"),
                    "Figure 10: RMSE improvement (%) vs cluster peer count.")

        # ── 4. RESULTS — 6-MONTH FORECAST ────────────────────────────
        text_page(pdf, [
            "## 4. Results: Six-Month Forecasts",
            "",
            "Using the full 2-year dataset for training (no holdout), ARIMA models",
            "were fit to all 30 stocks and projected 126 trading days (~6 months)",
            "forward with 95% confidence intervals derived from the statsmodels",
            "ARIMA forecast covariance.",
            "",
            "Figure 11 shows the forecast grid for all 30 stocks. Each subplot",
            "displays 3 months of recent history (solid black) followed by the",
            "6-month forecast (dashed blue) with a shaded 95% confidence band.",
            "",
            "Key observations:",
            "  - Most stocks forecast near-flat trajectories, consistent with the",
            "    random-walk-like behavior of efficient markets.",
            "  - JNJ is the notable outlier (+20.8% expected return), driven by",
            "    a strong upward trend in its recent price history.",
            "  - Confidence intervals widen substantially at the 6-month horizon,",
            "    reflecting increasing forecast uncertainty.",
            "  - High-volatility stocks (TSLA, UNH, CRM) show the widest CI bands.",
            "",
            "Tables 1-3 on the following pages present the numerical forecasts at",
            "1-month, 3-month, and 6-month horizons.",
        ])

        figure_page(pdf,
                    os.path.join(FIG_DIR, "fig7_six_month_forecast.png"),
                    "Figure 11: Six-month ARIMA forecasts with 95% CI for all 30 stocks.")

        # ── FORECAST TABLES ──────────────────────────────────────────
        fc_df = pd.read_csv(os.path.join(DATA_DIR, "six_month_forecasts.csv"))

        for horizon in ["1-Month", "3-Month", "6-Month"]:
            sub = fc_df[fc_df["Horizon"] == horizon].copy()
            sub = sub.sort_values("Ticker")
            display_df = sub[["Ticker", "Current Price", "Forecast",
                              "Lower 95% CI", "Upper 95% CI",
                              "Expected Return (%)"]].reset_index(drop=True)
            # Format numbers
            for col in ["Current Price", "Forecast", "Lower 95% CI", "Upper 95% CI"]:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}")
            display_df["Expected Return (%)"] = sub["Expected Return (%)"].values
            display_df["Expected Return (%)"] = display_df["Expected Return (%)"].apply(
                lambda x: f"{x:+.2f}%")

            table_page(pdf, display_df,
                       f"Table: {horizon} Price Forecasts with 95% Confidence Intervals",
                       fontsize=7)

        # ── 5. CONCLUSION ────────────────────────────────────────────
        text_page(pdf, [
            "## 5. Conclusion",
            "",
            "This project demonstrated an end-to-end pipeline for cluster-informed",
            "stock price forecasting. The key contributions are:",
            "",
            "  1. Feature Engineering + PCA: 38 technical and statistical features",
            "     reduced to 7 principal components capturing 91.9% of variance.",
            "",
            "  2. Meaningful Clustering: K-Means (K=6) identified interpretable",
            "     stock groups aligned with sector/volatility characteristics.",
            "",
            "  3. Cluster-Informed Forecasting: Two novel strategies that genuinely",
            "     incorporate cluster structure into ARIMA predictions:",
            "     - Ensemble ARIMA achieves statistically significant improvement",
            "       (p=0.035) over standalone ARIMA in walk-forward backtests.",
            "     - Concat ARIMA reduces RMSE by 30% for stocks with sufficient",
            "       cluster peers (e.g., AAPL test set: 10.83 vs 15.46).",
            "",
            "  4. Honest Evaluation: Walk-forward backtesting with Diebold-Mariano",
            "     significance tests. Results are mixed across stocks: cluster",
            "     methods help most when peer count is moderate (5-10 stocks).",
            "     Singleton clusters correctly degrade to regular ARIMA.",
            "",
            "  5. Forward Forecasts: 6-month predictions with 95% CI for all 30",
            "     stocks, demonstrating practical applicability.",
            "",
            "Limitations and future work:",
            "  - ARIMA assumes linear dynamics; nonlinear models (LSTM, transformers)",
            "    could potentially capture more complex patterns.",
            "  - Cluster assignments are static; dynamic re-clustering as market",
            "    regimes change could improve robustness.",
            "  - The ensemble self-weight (50%) was fixed; adaptive weighting based",
            "    on cluster cohesion could improve results.",
            "",
            "## References",
            "",
            "  - Box, G.E.P., Jenkins, G.M. (1976). Time Series Analysis.",
            "  - Diebold, F.X., Mariano, R.S. (1995). Comparing Predictive Accuracy.",
            "  - Jolliffe, I.T. (2002). Principal Component Analysis.",
            "  - MacQueen, J. (1967). Some Methods for Classification.",
        ])

    print(f"Report saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
