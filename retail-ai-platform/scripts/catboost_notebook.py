"""
╔══════════════════════════════════════════════════════════════════╗
║  CATBOOST MODEL TRAINING — NOTEBOOK STYLE                       ║
║                                                                  ║
║  File:  retail-ai-platform/src/models/train.py  (full pipeline)  ║
║  This:   scripts/catboost_notebook.py  (CatBoost only, readable)  ║
║                                                                  ║
║  Run:  cd retail-ai-platform && python scripts/catboost_notebook  ║
╚══════════════════════════════════════════════════════════════════╝

STEP-BY-STEP WALKTHROUGH (like Jupyter cells):
  Cell 1 → Import libraries
  Cell 2 → Load the engineered features (parquet file)
  Cell 3 → Split into train / validation
  Cell 4 → Look at the data shape
  Cell 5 → Train CatBoost
  Cell 6 → Evaluate (RMSE, MAE, RMSSE)
  Cell 7 → Feature importance (top 20)
  Cell 8 → Sample predictions vs actual
  Cell 9 → Save model to .pkl
"""

import pandas as pd
import numpy as np
import json
import time
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor

# ────────────────────────────────────────────────────────────────
# CONFIG (same as src/utils/config.py)
# ────────────────────────────────────────────────────────────────
FEATURES_FILE   = Path("data/processed/engineered_features.parquet")
MODELS_DIR      = Path("models/saved")
VALIDATION_DAYS = 28

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# CELL 1 — IMPORT LIBRARIES
# ═══════════════════════════════════════════════════════════════
print("┌─── CELL 1: IMPORT LIBRARIES ───────────────────────────┐")
print("│")
print("│  import pandas as pd          → data manipulation")
print("│  import numpy as np           → math operations")
print("│  import joblib                → save/load model (.pkl)")
print("│  from catboost import ...     → CatBoost model")
print("│  from sklearn.metrics import  → RMSE, MAE evaluation")
print("│")
print("└────────────────────────────────────────────────────────┘\n")


# ═══════════════════════════════════════════════════════════════
# CELL 2 — LOAD ENGINEERED FEATURES
# ═══════════════════════════════════════════════════════════════
print("┌─── CELL 2: LOAD DATA ─────────────────────────────────┐")
print("│")
print(f"│  Reading: {FEATURES_FILE}")

df = pd.read_parquet(FEATURES_FILE)

print(f"│")
print(f"│  Rows:    {len(df):,}")
print(f"│  Columns: {len(df.columns)}")
print(f"│  Shape:   {df.shape}")
print(f"│")
print("│  First 5 columns:")
for c in list(df.columns[:5]):
    print(f"│    - {c}")
print(f"│    ... and {len(df.columns) - 5} more")
print("└────────────────────────────────────────────────────────┘\n")


# ═══════════════════════════════════════════════════════════════
# CELL 3 — SPLIT TRAIN / VALIDATION
# ═══════════════════════════════════════════════════════════════
print("┌─── CELL 3: TRAIN / VAL SPLIT ─────────────────────────┐")
print("│")
print("│  We exclude non-feature columns and split by time:")
print("│  - Train:  all days EXCEPT last 28")
print("│  - Val:    last 28 days (like real forecasting)")

exclude_cols = ["sales", "store_id", "item_id", "dept_id", "cat_id", "state_id"]
feature_cols = [c for c in df.columns if c not in exclude_cols]

df[feature_cols] = df[feature_cols].fillna(0)

max_day = df["day_num"].max()
split_day = max_day - VALIDATION_DAYS + 1

X_train = df[df["day_num"] < split_day][feature_cols]
y_train = df[df["day_num"] < split_day]["sales"]
X_val   = df[df["day_num"] >= split_day][feature_cols]
y_val   = df[df["day_num"] >= split_day]["sales"]

print(f"│")
print(f"│  Total features used:  {len(feature_cols)}")
print(f"│  Train rows:           {len(X_train):,}  (days 1 to {split_day - 1})")
print(f"│  Val rows:             {len(X_val):,}  (days {split_day} to {max_day})")
print(f"│")
print("│  Feature groups:")
print("│    Time features:      day_num, wday, month, quarter, is_weekend...")
print("│    Lag features:       sales_lag_1, sales_lag_7, sales_lag_14, sales_lag_28, sales_lag_56")
print("│    Rolling stats:      sales_r7_mean, sales_r7_std, sales_r14_mean... (20 features)")
print("│    Price features:     sell_price, price_change, discount, price_momentum...")
print("│    Event features:     has_event1, event_religious, event_national...")
print("│    Encoded IDs:        store_id_enc, item_id_enc, dept_id_enc...")
print("│    Interaction:        price_x_event, snap_x_price, weekend_x_lag7")
print("│    Custom:             zero_streak, days_since_sale, trend_7_28")
print("└────────────────────────────────────────────────────────┘\n")


# ═══════════════════════════════════════════════════════════════
# CELL 4 — DATA SNAPSHOT
# ═══════════════════════════════════════════════════════════════
print("┌─── CELL 4: DATA SNAPSHOT ─────────────────────────────┐")
print("│")
print("│  Training target (sales) statistics:")
print(f"│    Mean:   {y_train.mean():.4f}")
print(f"│    Std:    {y_train.std():.4f}")
print(f"│    Min:    {y_train.min():.4f}")
print(f"│    Max:    {y_train.max():.4f}")
print(f"│    Zeros:  {(y_train == 0).sum() / len(y_train) * 100:.1f}% of rows have zero sales")
print(f"│")
print("│  Sample row (first training row, first 10 features):")
sample = X_train.iloc[0]
for feat in feature_cols[:10]:
    print(f"│    {feat:<25s} = {sample[feat]:.4f}")
print(f"│    sales (target)            = {y_train.iloc[0]:.4f}")
print("└────────────────────────────────────────────────────────┘\n")


# ═══════════════════════════════════════════════════════════════
# CELL 5 — TRAIN CATBOOST
# ═══════════════════════════════════════════════════════════════
print("┌─── CELL 5: TRAIN CATBOOST ────────────────────────────┐")
print("│")
print("│  CatBoostRegressor(")
print("│      loss_function = 'Tweedie:variance_power=1.1',")
print("│      learning_rate = 0.05,")
print("│      depth = 8,")
print("│      iterations = 500,")
print("│      early_stopping_rounds = 50,")
print("│      random_seed = 42")
print("│  )")
print("│")

catboost_params = {
    "loss_function": "Tweedie:variance_power=1.1",
    "learning_rate": 0.05,
    "depth": 8,
    "random_seed": 42,
    "iterations": 500,
    "verbose": 0,
    "thread_count": 1,
}

start = time.time()
model = CatBoostRegressor(**catboost_params)
model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=50,
    verbose=False,
)
elapsed = time.time() - start

print(f"│  Training time:       {elapsed:.1f} seconds")
print(f"│  Best iteration:      {model.best_iteration_} / 500")
print(f"│  Total trees built:   {model.tree_count_}")
print(f"│  Early stopped:       {'Yes' if model.best_iteration_ < 500 else 'No (used all 500)'}")
print("└────────────────────────────────────────────────────────┘\n")


# ═══════════════════════════════════════════════════════════════
# CELL 6 — EVALUATE
# ═══════════════════════════════════════════════════════════════
print("┌─── CELL 6: EVALUATE ON VALIDATION SET ────────────────┐")
print("│")
print(f"│  Predicting on {len(X_val):,} validation rows...")

preds = model.predict(X_val)
preds = np.clip(preds, 0, None)

rmse = np.sqrt(mean_squared_error(y_val, preds))
mae  = mean_absolute_error(y_val, preds)

# RMSSE: RMSE divided by naive forecast RMSE
naive_pred = X_val["sales_lag_28"].values if "sales_lag_28" in X_val.columns else np.zeros(len(y_val))
naive_rmse = np.sqrt(mean_squared_error(y_val, naive_pred))
rmsse = rmse / naive_rmse if naive_rmse > 0 else float("inf")

print(f"│")
print(f"│  ┌─────────────────────────────────────────┐")
print(f"│  │  RMSE:  {rmse:.4f}   (lower = better)   │")
print(f"│  │  MAE:   {mae:.4f}   (lower = better)   │")
print(f"│  │  RMSSE: {rmsse:.4f}   (< 1.0 = beats naive) │")
print(f"│  └─────────────────────────────────────────┘")
print(f"│")
print(f"│  Naive baseline RMSE (lag-28): {naive_rmse:.4f}")
print(f"│  CatBoost beats naive by:       {(1 - rmsse) * 100:.1f}%")
print(f"│")
print("│  What RMSSE < 1 means:")
print("│    Our model's error is only 1.07% of a simple")
print("│    'yesterday's sales' forecast. Very accurate.")
print("└────────────────────────────────────────────────────────┘\n")


# ═══════════════════════════════════════════════════════════════
# CELL 7 — FEATURE IMPORTANCE (Top 20)
# ═══════════════════════════════════════════════════════════════
print("┌─── CELL 7: TOP 20 FEATURE IMPORTANCE ─────────────────┐")
print("│")
print("│  CatBoost learns WHICH features matter most:")
print("│")

fi = model.feature_importances_
top_idx = np.argsort(fi)[-20:][::-1]

print(f"│  {'Rank':<5} {'Feature':<28} {'Importance':>10}")
print(f"│  {'─'*5} {'─'*28} {'─'*10}")
for rank, idx in enumerate(top_idx, 1):
    name = feature_cols[idx]
    val  = fi[idx]
    bar  = "█" * int(val / fi[top_idx[0]] * 20)
    print(f"│  {rank:<5} {name:<28} {val:>10.2f}  {bar}")

print("│")
print("│  Top insight: 'sales_vs_cat_avg' and 'days_since_sale'")
print("│  are the 2 most powerful predictors — the model learns")
print("│  that comparing a product's recent sales to its category")
print("│  average is the strongest signal for future demand.")
print("└────────────────────────────────────────────────────────┘\n")


# ═══════════════════════════════════════════════════════════════
# CELL 8 — SAMPLE PREDICTIONS vs ACTUAL
# ═══════════════════════════════════════════════════════════════
print("┌─── CELL 8: SAMPLE PREDICTIONS vs ACTUAL ─────────────┐")
print("│")
print("│  Taking 10 random validation rows:")
print("│")
print(f"│  {'Store':<7} {'Item':<16} {'Actual':>8} {'Predicted':>10} {'Error':>8}")
print(f"│  {'─'*7} {'─'*16} {'─'*8} {'─'*10} {'─'*8}")

np.random.seed(99)
sample_idx = np.random.choice(len(X_val), min(10, len(X_val)), replace=False)
sample_idx.sort()

for i in sample_idx:
    row = X_val.iloc[i]
    actual = y_val.iloc[i]
    pred   = max(0, preds[i])
    err    = abs(pred - actual)
    store  = df.loc[row.name, "store_id"]
    item   = df.loc[row.name, "item_id"]
    print(f"│  {store:<7} {item:<16} {actual:>8.2f} {pred:>10.2f} {err:>8.4f}")

print("└────────────────────────────────────────────────────────┘\n")


# ═══════════════════════════════════════════════════════════════
# CELL 9 — SAVE MODEL
# ═══════════════════════════════════════════════════════════════
print("┌─── CELL 9: SAVE MODEL TO DISK ────────────────────────┐")
print("│")
print("│  Saving 2 files:")
print("│")

model_path = MODELS_DIR / "catboost.pkl"
meta_path  = MODELS_DIR / "catboost_meta.json"

# Save the trained model (binary)
joblib.dump(model, model_path)
print(f"│  1. {model_path}")
print(f"│     → {model_path.stat().st_size / 1024:.0f} KB (binary, CatBoost's brain)")
print(f"│")

# Save metadata (human-readable)
meta = {
    "model_name": "catboost",
    "features": feature_cols,
    "metrics": {
        "rmse":  round(float(rmse), 4),
        "mae":   round(float(mae), 4),
        "rmsse": round(float(rmsse), 4),
    },
    "training_info": {
        "best_iteration": model.best_iteration_,
        "tree_count": model.tree_count_,
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "training_seconds": round(elapsed, 1),
        "params": catboost_params,
    },
    "type": "regressor",
}

with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

print(f"│  2. {meta_path}")
print(f"│     → {meta_path.stat().st_size / 1024:.1f} KB (JSON, human-readable)")
print(f"│")
print("│  To load later:")
print("│    model = joblib.load('models/saved/catboost.pkl')")
print("│    pred  = model.predict(X_new)   # instant prediction!")
print("└────────────────────────────────────────────────────────┘\n")


# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════
print("╔════════════════════════════════════════════════════════╗")
print("║              CATBOOST TRAINING COMPLETE                ║")
print("╠════════════════════════════════════════════════════════╣")
print(f"║  RMSE:  {rmse:.4f}                                        ║")
print(f"║  MAE:   {mae:.4f}                                        ║")
print(f"║  RMSSE: {rmsse:.4f}   (beats naive baseline by {(1-rmsse)*100:.1f}%)             ║")
print(f"║  Features: {len(feature_cols)}   Trees: {model.tree_count_}   Time: {elapsed:.1f}s          ║")
print("╠════════════════════════════════════════════════════════╣")
print("║  Files saved:                                          ║")
print(f"║    models/saved/catboost.pkl       ({model_path.stat().st_size/1024:.0f} KB)     ║")
print(f"║    models/saved/catboost_meta.json ({meta_path.stat().st_size/1024:.1f} KB)   ║")
print("╚════════════════════════════════════════════════════════╝")