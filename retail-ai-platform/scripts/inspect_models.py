"""
Quick model inspection — run like a Jupyter notebook cell.
Usage:  cd /home/z/my-project/retail-ai-platform && python scripts/inspect_models.py
"""

import json
import joblib
import numpy as np
from pathlib import Path

MODELS_DIR = Path("models/saved")

print("=" * 65)
print("  RETAIL AI PLATFORM — MODEL INSPECTION")
print("=" * 65)

for name in ["catboost", "lightgbm", "xgboost"]:
    model_path = MODELS_DIR / f"{name}.pkl"
    meta_path  = MODELS_DIR / f"{name}_meta.json"

    if not model_path.exists():
        print(f"\n  {name.upper()}: NOT FOUND")
        continue

    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    m = meta["metrics"]
    print(f"\n  ┌─ {name.upper()}")
    print(f"  │  RMSE:  {m['rmse']}")
    print(f"  │  MAE:   {m['mae']}")
    print(f"  │  RMSSE: {m['rmsse']}")
    print(f"  │  Features: {len(meta['features'])}")

    # Model-specific info
    if name == "lightgbm":
        print(f"  │  Type: LGBMRegressor")
        print(f"  │  Best iteration: {model.best_iteration_}")
        print(f"  │  Num trees: {model.n_estimators}")
        print(f"  │  Num leaves: {model.num_leaves}")
    elif name == "xgboost":
        print(f"  │  Type: XGBRegressor")
        best = getattr(model, "best_iteration", "N/A")
        print(f"  │  Best iteration: {best}")
    elif name == "catboost":
        print(f"  │  Type: CatBoostRegressor")
        print(f"  │  Best iteration: {model.best_iteration_}")
        print(f"  │  Tree count: {model.tree_count_}")

    # Top 10 features by importance
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        feat_names = meta["features"]
        top_idx = np.argsort(fi)[-10:][::-1]
        print(f"  │  Top 10 features:")
        for rank, idx in enumerate(top_idx, 1):
            print(f"  │    {rank:2d}. {feat_names[idx]:<25s} = {fi[idx]:.4f}")

    print(f"  └{'─' * 62}")

# Compare predictions on a sample
print("\n\n  ┌─ SAMPLE PREDICTION COMPARISON")
print("  │  Loading feature data...")

import pandas as pd
feat_data = pd.read_parquet("data/processed/engineered_features.parquet")
features = meta["features"]

sample = feat_data[feat_data["day_num"] == feat_data["day_num"].max()].iloc[0]
X = sample[features].fillna(0).values.reshape(1, -1)

print(f"  │  Store: {sample['store_id']}, Item: {sample['item_id']}")
print(f"  │  Actual sales (last day): {sample['sales']}")
print("  │")
for name in ["catboost", "lightgbm", "xgboost"]:
    m = joblib.load(MODELS_DIR / f"{name}.pkl")
    pred = max(0, float(m.predict(X)[0]))
    print(f"  │  {name:12s} prediction: {pred:.4f} units/day")

print(f"  └{'─' * 62}")
print("\n  Done.")