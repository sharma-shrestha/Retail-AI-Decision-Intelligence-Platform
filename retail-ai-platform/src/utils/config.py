"""
Retail AI Platform - Configuration
====================================
Central configuration for the entire platform.
"""

import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "m5-data"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

# Ensure directories exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Raw Data Files ──────────────────────────────────────────────────────────
SALES_FILE = DATA_RAW_DIR / "sales_train_validation.csv"
CALENDAR_FILE = DATA_RAW_DIR / "calendar.csv"
PRICES_FILE = DATA_RAW_DIR / "sell_prices.csv"

# ─── Processed Data Files ────────────────────────────────────────────────────
PROCESSED_SALES = DATA_PROCESSED_DIR / "processed_sales.parquet"
FEATURES_FILE = DATA_PROCESSED_DIR / "engineered_features.parquet"

# ─── Sampling Config ─────────────────────────────────────────────────────────
N_SAMPLE_ITEMS = 100       # Number of products to sample (4GB RAM)
RANDOM_SEED = 42
MAX_HISTORY_DAYS = 500     # Use only last N days of history

# ─── Feature Engineering ─────────────────────────────────────────────────────
LAGS = [1, 7, 14, 28, 56]
ROLLING_WINDOWS = [7, 14, 28, 56]
ROLLING_STATS = ["mean", "std", "min", "max", "median"]
EMBEDDING_SIZES = {
    "store_id": 10,
    "item_id": 500,
    "dept_id": 7,
    "cat_id": 3,
    "state_id": 3,
}

# ─── Model Config ────────────────────────────────────────────────────────────
VALIDATION_DAYS = 28       # Last 28 days for validation
HORIZON = 28               # Forecast horizon (28 days)
MODELS = ["lightgbm", "xgboost", "catboost"]

# ─── Model Hyperparameters ──────────────────────────────────────────────────
LIGHTGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 128,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "n_estimators": 500,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": 1,
}

XGBOOST_PARAMS = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.1,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 500,
    "random_state": 42,
    "n_jobs": 1,
    "tree_method": "hist",
}

CATBOOST_PARAMS = {
    "loss_function": "Tweedie:variance_power=1.1",
    "learning_rate": 0.05,
    "depth": 8,
    "random_seed": 42,
    "iterations": 500,
    "verbose": 0,
    "thread_count": 1,
}

# ─── API Config ──────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000