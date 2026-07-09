"""
Module 3 — Forecasting Engine
===============================
Trains LightGBM, XGBoost, and CatBoost models on engineered features.
Evaluates on validation set and saves the best model.
"""

import pandas as pd
import numpy as np
import gc
import json
import logging
import time
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.config import (
    FEATURES_FILE, MODELS_DIR,
    LIGHTGBM_PARAMS, XGBOOST_PARAMS, CATBOOST_PARAMS,
    VALIDATION_DAYS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_data():
    """Load engineered features and split into X/y."""
    logger.info("Loading engineered features...")
    train = pd.read_parquet(FEATURES_FILE)

    exclude_cols = ["sales", "store_id", "item_id", "dept_id", "cat_id", "state_id"]
    feature_cols = [c for c in train.columns if c not in exclude_cols]

    # Fill any remaining NaN with 0
    train[feature_cols] = train[feature_cols].fillna(0)

    X_train = train[train["day_num"] < train["day_num"].max() - VALIDATION_DAYS + 1][feature_cols]
    y_train = train[train["day_num"] < train["day_num"].max() - VALIDATION_DAYS + 1]["sales"]

    X_val = train[train["day_num"] >= train["day_num"].max() - VALIDATION_DAYS + 1][feature_cols]
    y_val = train[train["day_num"] >= train["day_num"].max() - VALIDATION_DAYS + 1]["sales"]

    logger.info(f"  Train: {X_train.shape}, Val: {X_val.shape}, Features: {len(feature_cols)}")
    del train
    gc.collect()

    return X_train, y_train, X_val, y_val, feature_cols


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model."""
    import lightgbm as lgb

    logger.info("Training LightGBM...")
    start = time.time()

    model = lgb.LGBMRegressor(**LIGHTGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    elapsed = time.time() - start
    logger.info(f"  LightGBM trained in {elapsed:.1f}s, best iteration: {model.best_iteration_}")
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model."""
    import xgboost as xgb

    logger.info("Training XGBoost...")
    start = time.time()

    params = {**XGBOOST_PARAMS, "early_stopping_rounds": 50}
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    elapsed = time.time() - start
    best_iter = getattr(model, "best_iteration", getattr(model, "best_ntree_limit", "N/A"))
    logger.info(f"  XGBoost trained in {elapsed:.1f}s, best iteration: {best_iter}")
    return model


def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost model."""
    from catboost import CatBoostRegressor

    logger.info("Training CatBoost...")
    start = time.time()

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )

    elapsed = time.time() - start
    logger.info(f"  CatBoost trained in {elapsed:.1f}s, best iteration: {model.best_iteration_}")
    return model


def evaluate(model, X_val, y_val, name: str) -> dict:
    """Calculate evaluation metrics."""
    preds = model.predict(X_val)
    preds = np.clip(preds, 0, None)  # No negative sales

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    mae = mean_absolute_error(y_val, preds)
    # RMSSE (scaled RMSE) - M5 competition metric approx
    naive_rmse = np.sqrt(mean_squared_error(
        y_val,
        X_val["sales_lag_28"].values if "sales_lag_28" in X_val.columns else np.zeros(len(y_val))
    ))
    rmsse = rmse / naive_rmse if naive_rmse > 0 else float("inf")

    metrics = {"rmse": round(float(rmse), 4), "mae": round(float(mae), 4), "rmsse": round(float(rmsse), 4)}
    logger.info(f"  {name:12s} — RMSE: {rmse:.4f}, MAE: {mae:.4f}, RMSSE: {rmsse:.4f}")
    return metrics


def save_model(model, name: str, feature_cols: list, metrics: dict):
    """Save model and metadata."""
    model_path = MODELS_DIR / f"{name}.pkl"
    meta_path = MODELS_DIR / f"{name}_meta.json"

    import joblib
    joblib.dump(model, model_path)

    meta = {
        "model_name": name,
        "features": feature_cols,
        "metrics": metrics,
        "type": "regressor",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"  Saved model to {model_path}")
    return model_path


def run_training() -> dict:
    """Full training pipeline for all models."""
    logger.info("=" * 60)
    logger.info("  MODULE 3 — FORECASTING ENGINE")
    logger.info("=" * 60)

    X_train, y_train, X_val, y_val, feature_cols = load_data()

    # Train all models
    models = {}
    results = {}

    # LightGBM
    try:
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val)
        metrics = evaluate(lgb_model, X_val, y_val, "LightGBM")
        save_model(lgb_model, "lightgbm", feature_cols, metrics)
        models["lightgbm"] = lgb_model
        results["lightgbm"] = metrics
    except Exception as e:
        logger.error(f"  LightGBM failed: {e}")

    if "lgb_model" in dir():
        del lgb_model
    gc.collect()

    # XGBoost
    try:
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
        metrics = evaluate(xgb_model, X_val, y_val, "XGBoost")
        save_model(xgb_model, "xgboost", feature_cols, metrics)
        models["xgboost"] = xgb_model
        results["xgboost"] = metrics
    except Exception as e:
        logger.error(f"  XGBoost failed: {e}")

    if "xgb_model" in dir():
        del xgb_model
    gc.collect()

    # CatBoost
    try:
        cb_model = train_catboost(X_train, y_train, X_val, y_val)
        metrics = evaluate(cb_model, X_val, y_val, "CatBoost")
        save_model(cb_model, "catboost", feature_cols, metrics)
        models["catboost"] = cb_model
        results["catboost"] = metrics
    except Exception as e:
        logger.error(f"  CatBoost failed: {e}")

    # Summary
    logger.info(f"\n  {'─'*40}")
    logger.info("  Model Comparison:")
    logger.info(f"  {'Model':<12} {'RMSE':>8} {'MAE':>8} {'RMSSE':>8}")
    for name, m in results.items():
        logger.info(f"  {name:<12} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['rmsse']:>8.4f}")

    best = min(results, key=lambda k: results[k]["rmse"])
    logger.info(f"\n  Best model: {best} (RMSE={results[best]['rmse']:.4f})")
    logger.info("=" * 60)

    return {"models": list(results.keys()), "results": results, "best": best}


if __name__ == "__main__":
    run_training()