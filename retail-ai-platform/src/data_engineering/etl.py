"""
Module 1 — Data Engineering (Memory-Optimized for 4GB RAM)
============================================================
Processes M5 data in stages with aggressive memory management.
Uses chunked melting to avoid OOM.
"""

import pandas as pd
import numpy as np
import gc
import logging
from pathlib import Path

from src.utils.config import (
    SALES_FILE, CALENDAR_FILE, PRICES_FILE,
    PROCESSED_SALES, N_SAMPLE_ITEMS, RANDOM_SEED, MAX_HISTORY_DAYS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sample_items() -> list[str]:
    """Load only metadata columns from sales, sample items."""
    logger.info("Loading sales metadata for sampling...")
    meta = pd.read_csv(SALES_FILE, usecols=["item_id", "cat_id"])
    cat_counts = meta.groupby("cat_id")["item_id"].nunique()
    total = cat_counts.sum()

    np.random.seed(RANDOM_SEED)
    sampled = []
    for cat, cnt in cat_counts.items():
        n = max(5, int(N_SAMPLE_ITEMS * cnt / total))
        items = meta[meta["cat_id"] == cat]["item_id"].unique()
        sampled.extend(np.random.choice(items, size=min(n, len(items)), replace=False).tolist())

    sampled = sorted(set(sampled))
    foods = sum(1 for i in sampled if "FOODS" in i)
    hobbies = sum(1 for i in sampled if "HOBBIES" in i)
    household = sum(1 for i in sampled if "HOUSEHOLD" in i)
    logger.info(f"Sampled {len(sampled)} items: FOODS={foods}, HOBBIES={hobbies}, HOUSEHOLD={household}")
    del meta
    gc.collect()
    return sampled


def melt_in_chunks(sampled_items: list[str], chunk_days: int = 200) -> pd.DataFrame:
    """Melt sales in day-chunks to control memory usage.

    Instead of melting all 1913 day columns at once (creates 9.5M rows),
    we melt 200 days at a time and append.
    """
    logger.info("Loading filtered sales (wide format)...")
    sales = pd.read_csv(SALES_FILE)
    sales = sales[sales["item_id"].isin(sampled_items)].copy()
    logger.info(f"  Filtered to {sales.shape[0]:,} product-store rows, {sales.shape[1]} cols")

    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    n_days = len(day_cols)
    logger.info(f"  Days: {n_days}, Melt chunk size: {chunk_days}")

    chunks = []
    for start in range(0, n_days, chunk_days):
        end = min(start + chunk_days, n_days)
        batch_days = day_cols[start:end]
        batch = sales[id_cols + batch_days].melt(
            id_vars=id_cols, value_vars=batch_days,
            var_name="day_id", value_name="sales",
        )
        chunks.append(batch)
        logger.info(f"    Melted days {start}-{end}: {len(batch):,} rows")

    df_long = pd.concat(chunks, ignore_index=True)
    df_long["day_num"] = df_long["day_id"].str.extract(r"(\d+)").astype(int)
    df_long.drop(columns=["day_id"], inplace=True)

    del sales, chunks
    gc.collect()

    logger.info(f"  Total long rows: {df_long.shape[0]:,}")
    return df_long


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast types to save memory."""
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="unsigned")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique() / max(len(df), 1) < 0.5:
            df[col] = df[col].astype("category")
    return df


def run_etl() -> pd.DataFrame:
    """Full ETL pipeline."""
    logger.info("=" * 60)
    logger.info("  MODULE 1 — DATA ENGINEERING PIPELINE")
    logger.info("=" * 60)

    # Stage 1: Sample items
    sampled_items = sample_items()

    # Stage 2: Load calendar
    logger.info("Loading calendar...")
    calendar = pd.read_csv(CALENDAR_FILE)
    calendar["date"] = pd.to_datetime(calendar["date"])
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        calendar[col] = calendar[col].fillna("None")

    # Stage 3: Load filtered prices
    logger.info("Loading filtered prices...")
    prices = pd.read_csv(PRICES_FILE)
    prices = prices[prices["item_id"].isin(sampled_items)].copy()
    prices = optimize_dtypes(prices)
    logger.info(f"  Filtered prices: {len(prices):,} rows")

    # Stage 4: Melt sales in chunks
    sales_long = melt_in_chunks(sampled_items, chunk_days=200)
    sales_long = optimize_dtypes(sales_long)

    # Stage 5: Merge with calendar
    logger.info("Merging with calendar...")
    calendar["d"] = calendar["d"].str.extract(r"(\d+)").astype(int)
    calendar.rename(columns={"d": "day_num"}, inplace=True)
    df = sales_long.merge(calendar, on="day_num", how="left")
    del sales_long, calendar
    gc.collect()

    # Stage 6: Merge with prices
    logger.info("Merging with prices...")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    del prices
    gc.collect()

    # Stage 7: Final cleanup
    df = optimize_dtypes(df)
    df = df.sort_values(["store_id", "item_id", "day_num"]).reset_index(drop=True)

    # Truncate to last N days to save memory for feature engineering
    max_day = df["day_num"].max()
    min_day = max(1, max_day - MAX_HISTORY_DAYS + 1)
    before = len(df)
    df = df[df["day_num"] >= min_day].copy()
    logger.info(f"  Truncated to last {MAX_HISTORY_DAYS} days: {before:,} → {len(df):,} rows")

    # Fill missing prices
    n_missing = df["sell_price"].isnull().sum()
    if n_missing > 0:
        logger.info(f"Filling {n_missing:,} missing prices with item-store median...")
        df["sell_price"] = df.groupby(["store_id", "item_id"])["sell_price"].transform(
            lambda x: x.fillna(x.median())
        )
        # Any remaining NaN → 0
        df["sell_price"] = df["sell_price"].fillna(0)

    # Save
    df.to_parquet(PROCESSED_SALES, index=False, engine="pyarrow")
    size_mb = PROCESSED_SALES.stat().st_size / 1024 / 1024
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024

    logger.info(f"\n  {'─'*40}")
    logger.info(f"  Final shape:  {df.shape[0]:,} rows × {df.shape[1]} cols")
    logger.info(f"  Memory:       {mem_mb:.1f} MB")
    logger.info(f"  File size:    {size_mb:.1f} MB")
    logger.info(f"  Date range:   {df['date'].min()} → {df['date'].max()}")
    logger.info(f"  Sales:        [{df['sales'].min()}, {df['sales'].max()}]")
    logger.info(f"  Products:     {df['item_id'].nunique()}")
    logger.info(f"  Stores:       {df['store_id'].nunique()}")
    logger.info(f"  Saved to:     {PROCESSED_SALES}")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    df = run_etl()
    print(df.head(3).to_string())
    print(f"\nColumns ({len(df.columns)}): {df.columns.tolist()}")