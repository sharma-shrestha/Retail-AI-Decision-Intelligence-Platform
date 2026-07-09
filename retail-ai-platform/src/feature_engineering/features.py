"""
Module 2 — Feature Engineering Engine
========================================
Creates 100+ business intelligence features from processed sales data.

Feature Groups:
  1. Time Features (~20)
  2. Sales Lags (~10)
  3. Rolling Statistics (~40)
  4. Price Features (~15)
  5. Business / Encoded Features (~15)
  6. Interaction Features (~10)
"""

import pandas as pd
import numpy as np
import gc
import logging
from pathlib import Path

from src.utils.config import (
    PROCESSED_SALES, FEATURES_FILE,
    LAGS, ROLLING_WINDOWS, ROLLING_STATS,
    VALIDATION_DAYS, HORIZON,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Group 1: Time Features
# ══════════════════════════════════════════════════════════════════════════════

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive calendar-based time features."""
    logger.info("Adding time features...")

    df["day_of_month"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["wday"] <= 2).astype(int)  # Sat=1, Sun=2
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # Cyclical encoding for month and day_of_week
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["wday_sin"] = np.sin(2 * np.pi * df["wday"] / 7)
    df["wday_cos"] = np.cos(2 * np.pi * df["wday"] / 7)

    # Event features
    df["has_event1"] = (df["event_type_1"] != "None").astype(int)
    df["has_event2"] = (df["event_type_2"] != "None").astype(int)
    df["has_any_event"] = ((df["event_type_1"] != "None") | (df["event_type_2"] != "None")).astype(int)

    # Event type dummies
    for etype in ["Religious", "National", "Cultural", "Sporting", "Federal"]:
        df[f"event_{etype.lower()}"] = (
            (df["event_type_1"] == etype) | (df["event_type_2"] == etype)
        ).astype(int)

    # SNAP features (already 0/1, keep as-is)

    n_added = 20
    logger.info(f"  Added {n_added} time features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Group 2: Sales Lags
# ══════════════════════════════════════════════════════════════════════════════

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged sales features per product-store series."""
    logger.info(f"Adding lag features: {LAGS}...")

    df = df.sort_values(["store_id", "item_id", "day_num"]).reset_index(drop=True)

    for lag in LAGS:
        col = f"sales_lag_{lag}"
        df[col] = df.groupby(["store_id", "item_id"], observed=True)["sales"].shift(lag)

    logger.info(f"  Added {len(LAGS)} lag features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Group 3: Rolling Statistics
# ══════════════════════════════════════════════════════════════════════════════

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create rolling window statistics per product-store series."""
    logger.info(f"Adding rolling features: windows={ROLLING_WINDOWS}, stats={ROLLING_STATS}...")

    df = df.sort_values(["store_id", "item_id", "day_num"]).reset_index(drop=True)

    count = 0
    for window in ROLLING_WINDOWS:
        for stat in ROLLING_STATS:
            col = f"sales_r{window}_{stat}"
            if stat == "mean":
                df[col] = df.groupby(["store_id", "item_id"], observed=True)["sales"].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
                )
            elif stat == "std":
                df[col] = df.groupby(["store_id", "item_id"], observed=True)["sales"].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).std().fillna(0)
                )
            elif stat == "min":
                df[col] = df.groupby(["store_id", "item_id"], observed=True)["sales"].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
                )
            elif stat == "max":
                df[col] = df.groupby(["store_id", "item_id"], observed=True)["sales"].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
                )
            elif stat == "median":
                df[col] = df.groupby(["store_id", "item_id"], observed=True)["sales"].transform(
                    lambda x: x.shift(1).rolling(window=window, min_periods=1).median()
                )
            count += 1

    # Expanding mean (cumulative average up to previous day)
    df["sales_expanding_mean"] = df.groupby(["store_id", "item_id"], observed=True)["sales"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    count += 1

    logger.info(f"  Added {count} rolling features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Group 4: Price Features
# ══════════════════════════════════════════════════════════════════════════════

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create price-related features."""
    logger.info("Adding price features...")

    df = df.sort_values(["store_id", "item_id", "day_num"]).reset_index(drop=True)

    # Price change from previous day
    df["price_change"] = df.groupby(["store_id", "item_id"], observed=True)["sell_price"].diff()
    df["price_change_pct"] = df.groupby(["store_id", "item_id"], observed=True)["sell_price"].pct_change().fillna(0)

    # Rolling price statistics
    for window in [7, 28]:
        df[f"price_r{window}_mean"] = df.groupby(["store_id", "item_id"], observed=True)["sell_price"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"price_r{window}_std"] = df.groupby(["store_id", "item_id"], observed=True)["sell_price"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std().fillna(0)
        )

    # Discount = relative to rolling max price
    df["price_r28_max"] = df.groupby(["store_id", "item_id"], observed=True)["sell_price"].transform(
        lambda x: x.shift(1).rolling(window=28, min_periods=1).max()
    )
    df["discount"] = (df["sell_price"] - df["price_r28_max"]) / df["price_r28_max"].replace(0, np.nan)
    df["discount"] = df["discount"].fillna(0)

    # Price momentum (is price trending up or down?)
    df["price_7d_ago"] = df.groupby(["store_id", "item_id"], observed=True)["sell_price"].shift(7)
    df["price_momentum"] = (df["sell_price"] - df["price_7d_ago"]) / df["price_7d_ago"].replace(0, np.nan)
    df["price_momentum"] = df["price_momentum"].fillna(0)

    # Drop intermediate columns
    df.drop(columns=["price_7d_ago", "price_r28_max"], inplace=True)

    n_added = 9
    logger.info(f"  Added {n_added} price features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Group 5: Business / Encoded Features
# ══════════════════════════════════════════════════════════════════════════════

def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create business KPI features using label encoding and store/category stats."""
    logger.info("Adding business features...")

    # Label encode categorical columns
    cat_cols = ["store_id", "item_id", "dept_id", "cat_id", "state_id", "weekday"]
    for col in cat_cols:
        df[f"{col}_enc"] = df[col].astype("category").cat.codes

    # Store popularity: avg sales per store (overall)
    store_avg = df.groupby("store_id", observed=True)["sales"].transform("mean")
    df["store_avg_sales"] = store_avg

    # Category avg sales
    cat_avg = df.groupby("cat_id", observed=True)["sales"].transform("mean")
    df["cat_avg_sales"] = cat_avg

    # Item overall avg
    item_avg = df.groupby("item_id", observed=True)["sales"].transform("mean")
    df["item_avg_sales"] = item_avg

    # Sales vs store average (relative demand)
    df["sales_vs_store_avg"] = df["sales"] - df["store_avg_sales"]

    # Sales vs category average
    df["sales_vs_cat_avg"] = df["sales"] - df["cat_avg_sales"]

    # Zero sale streak (how many consecutive zero sales before this day)
    df = df.sort_values(["store_id", "item_id", "day_num"]).reset_index(drop=True)
    df["zero_streak"] = df.groupby(["store_id", "item_id"], observed=True)["sales"].transform(
        lambda x: x.eq(0).groupby((x.ne(0)).cumsum()).cumsum().shift(1).fillna(0)
    )

    # Days since last sale
    df["days_since_sale"] = df.groupby(["store_id", "item_id"], observed=True).apply(
        lambda g: g["day_num"] - g.loc[g["sales"] > 0, "day_num"].shift(1)
    ).reset_index(level=[0, 1], drop=True).fillna(999).astype(int)

    n_added = 15
    logger.info(f"  Added {n_added} business features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Feature Group 6: Interaction Features
# ══════════════════════════════════════════════════════════════════════════════

def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction features combining multiple feature groups."""
    logger.info("Adding interaction features...")

    # Price × event interaction
    df["price_x_event"] = df["sell_price"] * df["has_any_event"]

    # Weekend × lag interaction
    if "sales_lag_7" in df.columns:
        df["weekend_x_lag7"] = df["is_weekend"] * df["sales_lag_7"]

    # SNAP interaction (per state)
    df["snap"] = 0
    df.loc[df["state_id"] == "CA", "snap"] = df.loc[df["state_id"] == "CA", "snap_CA"]
    df.loc[df["state_id"] == "TX", "snap"] = df.loc[df["state_id"] == "TX", "snap_TX"]
    df.loc[df["state_id"] == "WI", "snap"] = df.loc[df["state_id"] == "WI", "snap_WI"]
    df["snap_x_price"] = df["snap"] * df["sell_price"]

    # Trend: ratio of recent to older sales
    if "sales_r7_mean" in df.columns and "sales_r28_mean" in df.columns:
        df["trend_7_28"] = df["sales_r7_mean"] / df["sales_r28_mean"].replace(0, np.nan)
        df["trend_7_28"] = df["trend_7_28"].fillna(1)

    n_added = 5
    logger.info(f"  Added {n_added} interaction features")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Train/Validation Split
# ══════════════════════════════════════════════════════════════════════════════

def split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train (all but last 28 days) and validation (last 28 days)."""
    max_day = df["day_num"].max()
    val_start = max_day - VALIDATION_DAYS + 1

    train = df[df["day_num"] < val_start].copy()
    val = df[df["day_num"] >= val_start].copy()

    # Drop rows with NaN features (from lag/rolling) in training only
    train = train.dropna(subset=[c for c in train.columns if "lag_" in c or "r7_" in c])
    train = train.reset_index(drop=True)

    logger.info(f"  Train: {len(train):,} rows (day 1-{max_day - VALIDATION_DAYS})")
    logger.info(f"  Val:   {len(val):,} rows (day {val_start}-{max_day})")
    return train, val


# ══════════════════════════════════════════════════════════════════════════════
#  Main Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_feature_engineering() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Run full feature engineering pipeline.

    Returns:
        train_df, val_df, feature_columns
    """
    logger.info("=" * 60)
    logger.info("  MODULE 2 — FEATURE ENGINEERING ENGINE")
    logger.info("=" * 60)

    # Load processed data
    logger.info("Loading processed data...")
    df = pd.read_parquet(PROCESSED_SALES)
    logger.info(f"  Input: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # Apply feature groups sequentially
    df = add_time_features(df)
    gc.collect()

    df = add_lag_features(df)
    gc.collect()

    df = add_rolling_features(df)
    gc.collect()

    df = add_price_features(df)
    gc.collect()

    df = add_business_features(df)
    gc.collect()

    df = add_interaction_features(df)
    gc.collect()

    # Drop raw text columns that are now encoded
    drop_cols = ["id", "date", "event_name_1", "event_type_1",
                 "event_name_2", "event_type_2", "weekday"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Split
    logger.info("Splitting train/validation...")
    train_df, val_df = split_train_val(df)

    # Identify feature columns (everything except target and metadata)
    exclude_cols = ["sales", "store_id", "item_id", "dept_id", "cat_id", "state_id"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    # Save engineered features
    train_df.to_parquet(FEATURES_FILE, index=False)
    size_mb = FEATURES_FILE.stat().st_size / 1024 / 1024
    logger.info(f"\n  {'─'*40}")
    logger.info(f"  Total features:    {len(feature_cols)}")
    logger.info(f"  Train shape:       {train_df.shape}")
    logger.info(f"  Val shape:         {val_df.shape}")
    logger.info(f"  Saved train data:  {FEATURES_FILE} ({size_mb:.1f} MB)")
    logger.info("=" * 60)

    return train_df, val_df, feature_cols


if __name__ == "__main__":
    train, val, features = run_feature_engineering()
    print(f"\nFeature columns ({len(features)}):")
    for i, f in enumerate(features):
        print(f"  {i+1:3d}. {f}")
    print(f"\nTrain sample:\n{train.head(2).to_string()}")