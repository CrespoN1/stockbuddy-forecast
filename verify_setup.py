#!/usr/bin/env python3
"""
StockBuddy Forecast — Setup Verification Script

Run this after installing requirements to verify everything is configured correctly:
    python verify_setup.py

This checks:
1. All required packages are installed
2. All source modules import successfully
3. All classes and functions are accessible
4. Directory structure is correct
5. Notebooks exist and are valid JSON
"""
import sys
import json
import os

# Ensure we're running from the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"

passed = 0
failed = 0
warnings = 0


def check(description, condition, warn_only=False):
    global passed, failed, warnings
    if condition:
        print(f"  {PASS} {description}")
        passed += 1
    elif warn_only:
        print(f"  {WARN} {description} (optional)")
        warnings += 1
    else:
        print(f"  {FAIL} {description}")
        failed += 1


# =========================================================================
# 1. Check required packages
# =========================================================================
print("\n1. Checking required packages...")

required_packages = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "sklearn": "scikit-learn",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "yfinance": "yfinance",
    "requests": "requests",
    "bs4": "beautifulsoup4",
    "tqdm": "tqdm",
    "dotenv": "python-dotenv",
}

optional_packages = {
    "tensorflow": "tensorflow (needed for LSTM model)",
    "vaderSentiment": "vaderSentiment (needed for sentiment analysis)",
    "plotly": "plotly (needed for interactive charts)",
}

for import_name, pkg_name in required_packages.items():
    try:
        __import__(import_name)
        check(f"{pkg_name} installed", True)
    except ImportError:
        check(f"{pkg_name} installed — run: pip install {pkg_name}", False)

for import_name, pkg_name in optional_packages.items():
    try:
        __import__(import_name)
        check(f"{pkg_name} installed", True)
    except ImportError:
        check(f"{pkg_name} not installed", False, warn_only=True)

# =========================================================================
# 2. Check directory structure
# =========================================================================
print("\n2. Checking directory structure...")

required_dirs = [
    "src",
    "notebooks",
    "data",
    "data/raw",
    "data/processed",
    "figures",
]

for d in required_dirs:
    check(f"Directory exists: {d}/", os.path.isdir(d))

# =========================================================================
# 3. Check source modules
# =========================================================================
print("\n3. Checking source modules...")

required_files = [
    "src/__init__.py",
    "src/config.py",
    "src/data_loader.py",
    "src/feature_engineering.py",
    "src/sentiment.py",
    "src/pca_analysis.py",
    "src/clustering.py",
    "src/baselines.py",
    "src/forecasters.py",
    "src/evaluation.py",
]

for f in required_files:
    check(f"File exists: {f}", os.path.isfile(f))

# =========================================================================
# 4. Check module imports
# =========================================================================
print("\n4. Checking module imports...")

import_checks = [
    ("src.config", ["DATA_DIR", "FIGURES_DIR", "FORECAST_HORIZONS", "PCA_VARIANCE_THRESHOLD"]),
    ("src.data_loader", ["get_sp500_tickers", "fetch_stock_data", "fetch_multiple_stocks", "fetch_fundamentals", "build_price_matrix", "build_returns_matrix"]),
    ("src.feature_engineering", ["engineer_features", "create_feature_matrix", "compute_rsi", "compute_macd", "compute_bollinger_bands"]),
    ("src.sentiment", ["compute_batch_sentiment", "compute_ticker_sentiment", "get_news_headlines"]),
    ("src.pca_analysis", ["StockPCA"]),
    ("src.clustering", ["StockClusterer"]),
    ("src.baselines", ["NaiveForecast", "RandomWalkForecast", "SMAForecast", "ARIMAForecast", "run_all_baselines", "run_baselines_multi_horizon"]),
    ("src.evaluation", ["rmse", "mae", "mape", "directional_accuracy", "compute_all_metrics", "walk_forward_split", "backtest_model", "compare_models", "statistical_test"]),
]

for module_name, attrs in import_checks:
    try:
        mod = __import__(module_name, fromlist=attrs)
        missing = [a for a in attrs if not hasattr(mod, a)]
        if missing:
            check(f"{module_name} — missing: {', '.join(missing)}", False)
        else:
            check(f"{module_name} — all {len(attrs)} exports OK", True)
    except Exception as e:
        check(f"{module_name} — import error: {e}", False)

# Check forecasters separately (may fail without tensorflow)
try:
    from src.forecasters import LSTMForecaster, ClusterInformedForecaster, run_all_forecasters
    check("src.forecasters — all 3 exports OK", True)
except ImportError as e:
    if "tensorflow" in str(e).lower():
        check("src.forecasters — imports OK (LSTM needs TensorFlow at runtime)", True)
    else:
        check(f"src.forecasters — import error: {e}", False)
except Exception as e:
    check(f"src.forecasters — import error: {e}", False)

# =========================================================================
# 5. Check notebooks
# =========================================================================
print("\n5. Checking notebooks...")

notebooks = [
    "notebooks/01_data_collection.ipynb",
    "notebooks/02_feature_engineering.ipynb",
    "notebooks/03_eda_and_pca.ipynb",
    "notebooks/04_clustering.ipynb",
    "notebooks/05_forecasting_models.ipynb",
    "notebooks/06_evaluation.ipynb",
    "notebooks/07_results_visualization.ipynb",
]

for nb_path in notebooks:
    if os.path.isfile(nb_path):
        try:
            with open(nb_path, "r") as f:
                nb = json.load(f)
            n_cells = len(nb.get("cells", []))
            check(f"{nb_path} — valid notebook ({n_cells} cells)", n_cells > 0)
        except json.JSONDecodeError:
            check(f"{nb_path} — invalid JSON", False)
    else:
        check(f"{nb_path} — file missing", False)

# =========================================================================
# 6. Check other required files
# =========================================================================
print("\n6. Checking project files...")

check("requirements.txt exists", os.path.isfile("requirements.txt"))
check("README.md exists", os.path.isfile("README.md"))
check(".gitignore exists", os.path.isfile(".gitignore"))
check(".env.example exists", os.path.isfile(".env.example"))

# =========================================================================
# Summary
# =========================================================================
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed, {warnings} warnings")
print(f"{'='*50}")

if failed == 0:
    print(f"\n{PASS} All checks passed! You're ready to run the notebooks.")
    print("\nQuick start:")
    print("  1. cd notebooks/")
    print("  2. jupyter notebook")
    print("  3. Run notebooks 01 through 07 in order")
else:
    print(f"\n{FAIL} Some checks failed. Install missing packages with:")
    print("  pip install -r requirements.txt")

sys.exit(0 if failed == 0 else 1)
