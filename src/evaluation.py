"""
Evaluation module for StockBuddy Forecast.

Provides metrics, backtesting framework, and comparison utilities
for evaluating forecasting models.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from scipy import stats

from src.config import TEST_SIZE, WALK_FORWARD_WINDOWS, FORECAST_HORIZONS, FIGURES_DIR


# =============================================================================
# Metrics
# =============================================================================

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return np.nan
    return np.sqrt(np.mean((actual[mask] - predicted[mask]) ** 2))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    mask = ~(np.isnan(actual) | np.isnan(predicted))
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(actual[mask] - predicted[mask]))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = ~(np.isnan(actual) | np.isnan(predicted)) & (actual != 0)
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Directional accuracy: what fraction of the time did the model
    correctly predict the direction of price movement?
    """
    if len(actual) < 2 or len(predicted) < 2:
        return np.nan

    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))

    min_len = min(len(actual_dir), len(pred_dir))
    mask = ~(np.isnan(actual_dir[:min_len]) | np.isnan(pred_dir[:min_len]))

    if mask.sum() == 0:
        return np.nan

    return np.mean(actual_dir[:min_len][mask] == pred_dir[:min_len][mask]) * 100


def compute_all_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # Trim to same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    return {
        "RMSE": rmse(actual, predicted),
        "MAE": mae(actual, predicted),
        "MAPE": mape(actual, predicted),
        "Directional_Accuracy": directional_accuracy(actual, predicted),
    }


# =============================================================================
# Backtesting Framework
# =============================================================================

def walk_forward_split(
    series: pd.Series,
    n_windows: int = WALK_FORWARD_WINDOWS,
    test_size: float = TEST_SIZE,
    min_train_size: int = 252,  # At least 1 year of training data
) -> List[Tuple[pd.Series, pd.Series]]:
    """
    Generate walk-forward validation splits.

    Each split uses an expanding training window and a fixed-size test window.

    Returns:
        List of (train_series, test_series) tuples
    """
    n = len(series)
    test_len = max(1, int(n * test_size / n_windows))

    splits = []

    for i in range(n_windows):
        test_end = n - i * test_len
        test_start = test_end - test_len
        train_end = test_start

        if train_end < min_train_size:
            break

        train = series.iloc[:train_end]
        test = series.iloc[test_start:test_end]

        if len(train) > 0 and len(test) > 0:
            splits.append((train, test))

    # Reverse so splits are in chronological order
    splits.reverse()
    return splits


def backtest_model(
    series: pd.Series,
    model_class,
    model_params: Optional[Dict] = None,
    n_windows: int = WALK_FORWARD_WINDOWS,
    horizons: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Backtest a forecasting model using walk-forward validation.

    Args:
        series: Full price series
        model_class: Forecaster class (must have fit() and predict() methods)
        model_params: Parameters to pass to model constructor
        n_windows: Number of walk-forward windows
        horizons: Dict of horizon_name -> steps

    Returns:
        DataFrame with metrics for each window and horizon
    """
    if model_params is None:
        model_params = {}
    if horizons is None:
        horizons = FORECAST_HORIZONS

    splits = walk_forward_split(series, n_windows)
    results = []

    for window_idx, (train, test) in enumerate(splits):
        model = model_class(**model_params)
        model.fit(train)

        for horizon_name, horizon_steps in horizons.items():
            actual_horizon = min(horizon_steps, len(test))
            if actual_horizon == 0:
                continue

            predictions = model.predict(actual_horizon)
            actual = test.values[:actual_horizon]

            metrics = compute_all_metrics(actual, predictions)
            metrics["Window"] = window_idx
            metrics["Horizon"] = horizon_name
            metrics["Model"] = getattr(model, "name", model_class.__name__)

            results.append(metrics)

    return pd.DataFrame(results)


def backtest_cluster_model(
    series: pd.Series,
    ticker: str,
    stock_data: Dict[str, pd.DataFrame],
    cluster_labels: pd.Series,
    method: str = "ensemble",
    self_weight: float = 0.5,
    n_windows: int = WALK_FORWARD_WINDOWS,
    horizons: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    Backtest a cluster-informed model using walk-forward validation.

    Constructs a ClusterARIMAWrapper for each walk-forward window with
    proper date truncation to avoid data leakage from peer stocks.
    """
    from src.forecasters import ClusterARIMAWrapper

    if ticker not in cluster_labels.index:
        raise ValueError(f"Ticker '{ticker}' not found in cluster_labels index")

    if horizons is None:
        horizons = FORECAST_HORIZONS

    splits = walk_forward_split(series, n_windows)
    results = []

    for window_idx, (train, test) in enumerate(splits):
        model = ClusterARIMAWrapper(
            stock_data=stock_data,
            cluster_labels=cluster_labels,
            target_ticker=ticker,
            method=method,
            self_weight=self_weight,
        )
        model.fit(train)

        for horizon_name, horizon_steps in horizons.items():
            actual_horizon = min(horizon_steps, len(test))
            if actual_horizon == 0:
                continue

            predictions = model.predict(actual_horizon)
            actual = test.values[:actual_horizon]

            metrics = compute_all_metrics(actual, predictions)
            metrics["Window"] = window_idx
            metrics["Horizon"] = horizon_name
            metrics["Model"] = model.name

            results.append(metrics)

    return pd.DataFrame(results)


# =============================================================================
# Model Comparison
# =============================================================================

def compare_models(
    results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compare multiple models' backtest results.

    Args:
        results: Dict mapping model_name -> backtest results DataFrame

    Returns:
        Summary DataFrame with mean metrics per model per horizon
    """
    all_results = []

    for model_name, df in results.items():
        df = df.copy()
        df["Model"] = model_name
        all_results.append(df)

    combined = pd.concat(all_results, ignore_index=True)

    # Compute mean metrics per model per horizon
    summary = combined.groupby(["Model", "Horizon"]).agg(
        RMSE_mean=("RMSE", "mean"),
        RMSE_std=("RMSE", "std"),
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        MAPE_mean=("MAPE", "mean"),
        MAPE_std=("MAPE", "std"),
        DirAcc_mean=("Directional_Accuracy", "mean"),
        DirAcc_std=("Directional_Accuracy", "std"),
    ).round(4)

    return summary


def statistical_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    test: str = "diebold-mariano",
) -> Dict[str, float]:
    """
    Statistical significance test between two models' forecast errors.

    Uses a paired t-test on squared errors (simplified Diebold-Mariano).
    """
    mask = ~(np.isnan(errors_a) | np.isnan(errors_b))
    a = errors_a[mask]
    b = errors_b[mask]

    if len(a) < 2:
        return {"statistic": np.nan, "p_value": np.nan, "significant": False}

    # Difference in squared errors
    d = a**2 - b**2

    t_stat, p_value = stats.ttest_1samp(d, 0)

    return {
        "statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
    }


# =============================================================================
# Visualization
# =============================================================================

def plot_forecast_vs_actual(
    actual: pd.Series,
    predictions: Dict[str, np.ndarray],
    title: str = "Forecast vs Actual",
    save: bool = False,
    filename: str = "forecast_vs_actual.png",
):
    """Plot actual prices against model forecasts."""
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot actual
    ax.plot(range(len(actual)), actual.values, "k-", linewidth=2, label="Actual", alpha=0.8)

    # Plot each model's predictions
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
    for (model_name, preds), color in zip(predictions.items(), colors):
        if preds is not None and not np.all(np.isnan(preds)):
            start_idx = len(actual) - len(preds)
            ax.plot(
                range(start_idx, start_idx + len(preds)),
                preds,
                "--",
                color=color,
                linewidth=1.5,
                label=model_name,
                alpha=0.8,
            )

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price ($)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        import os
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

    plt.show()


def plot_model_comparison_heatmap(
    summary: pd.DataFrame,
    metric: str = "RMSE_mean",
    save: bool = False,
    filename: str = "model_comparison.png",
):
    """Plot a heatmap comparing models across horizons."""
    # Pivot for heatmap
    pivot_data = summary.reset_index().pivot(
        index="Model", columns="Horizon", values=metric
    )

    # Reorder horizons
    horizon_order = ["1d", "1w", "1m"]
    available_horizons = [h for h in horizon_order if h in pivot_data.columns]
    pivot_data = pivot_data[available_horizons]

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_data) * 0.6)))
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt=".4f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )

    metric_name = metric.replace("_mean", "").replace("_", " ").upper()
    ax.set_title(f"Model Comparison: {metric_name} by Forecast Horizon")
    ax.set_xlabel("Forecast Horizon")
    ax.set_ylabel("Model")

    plt.tight_layout()

    if save:
        import os
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

    plt.show()


def plot_metrics_bar_chart(
    summary: pd.DataFrame,
    horizon: str = "1d",
    save: bool = False,
    filename: str = "metrics_bar_chart.png",
):
    """Plot bar chart of all metrics for a specific horizon."""
    if isinstance(summary.index, pd.MultiIndex):
        data = summary.xs(horizon, level="Horizon")
    else:
        data = summary[summary["Horizon"] == horizon]

    metrics = ["RMSE_mean", "MAE_mean", "MAPE_mean", "DirAcc_mean"]
    available_metrics = [m for m in metrics if m in data.columns]

    fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 5))
    if len(available_metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available_metrics):
        values = data[metric].sort_values()
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))

        if "DirAcc" in metric:
            # Higher is better for directional accuracy
            values = values.sort_values(ascending=True)
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))

        values.plot(kind="barh", ax=ax, color=colors)
        metric_name = metric.replace("_mean", "").replace("_", " ")
        ax.set_title(f"{metric_name} ({horizon})")
        ax.set_xlabel(metric_name)

    plt.suptitle(f"Model Performance Comparison — {horizon} Horizon", fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        import os
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")

    plt.show()
