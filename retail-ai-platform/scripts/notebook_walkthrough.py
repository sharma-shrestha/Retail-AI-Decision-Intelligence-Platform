"""
Load already-trained models and show all results.
Like a Jupyter notebook — each section is a "cell".
"""

import json, joblib, numpy as np, pandas as pd, time
from pathlib import Path

MODELS_DIR = Path("/home/z/my-project/retail-ai-platform/models/saved")
FEAT_FILE  = Path("/home/z/my-project/retail-ai-platform/data/processed/engineered_features.parquet")

# ── CELL 1: Load models ─────────────────────────────────────────────
print("=" * 70)
print("  CELL 1: Load trained models from .pkl files")
print("=" * 70)

models = {}
metas  = {}
for name in ["catboost", "lightgbm", "xgboost"]:
    print(f"\n  Loading {name}...", end=" ", flush=True)
    t0 = time.time()
    models[name] = joblib.load(MODELS_DIR / f"{name}.pkl")
    with open(MODELS_DIR / f"{name}_meta.json") as f:
        metas[name] = json.load(f)
    print(f"OK ({time.time()-t0:.2f}s)")

features = metas["catboost"]["features"]
print(f"\n  All 3 models loaded. Shared features: {len(features)}")

# ── CELL 2: Model comparison ─────────────────────────────────────────
print("\n" + "=" * 70)
print("  CELL 2: Model Comparison (from validation set)")
print("=" * 70)

print(f"\n  {'Model':<12} {'RMSE':>8} {'MAE':>8} {'RMSSE':>8} {'Features':>8}")
print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
for name in ["catboost", "lightgbm", "xgboost"]:
    m = metas[name]["metrics"]
    print(f"  {name:<12} {m['rmse']:>8.4f} {m['mae']:>8.4f} {m['rmsse']:>8.4f} {len(metas[name]['features']):>8}")

best = min(metas, key=lambda k: metas[k]["metrics"]["rmse"])
print(f"\n  BEST MODEL: {best.upper()} (RMSE = {metas[best]['metrics']['rmse']})")

# ── CELL 3: Feature importance (CatBoost) ───────────────────────────
print("\n" + "=" * 70)
print("  CELL 3: CatBoost Top 15 Feature Importance")
print("=" * 70)

fi = models["catboost"].feature_importances_
ranks = sorted(zip(features, fi), key=lambda x: x[1], reverse=True)

print(f"\n  {'#':<4} {'Feature':<30} {'Importance':>10} {'Type':<15}")
print(f"  {'─'*4} {'─'*30} {'─'*10} {'─'*15}")

# Group features by category
def categorize(f):
    if "lag" in f: return "Lag"
    if "r7_" in f or "r14_" in f or "r28_" in f or "r56_" in f: return "Rolling Stats"
    if "price" in f or "discount" in f: return "Price"
    if "event" in f: return "Calendar Event"
    if "enc" in f: return "Embedding"
    if "snap" in f: return "SNAP"
    if "sin" in f or "cos" in f: return "Cyclical"
    if "month" in f or "wday" in f or "week" in f or "quarter" in f: return "Calendar"
    if "streak" in f or "since" in f: return "Zero Sales"
    if "vs_" in f or "avg_sales" in f: return "Contextual"
    return "Other"

for rank, (feat, imp) in enumerate(ranks[:15], 1):
    print(f"  {rank:<4} {feat:<30} {imp:>10.2f} {categorize(feat):<15}")

print(f"\n  Feature categories used:")
cats = {}
for f in features:
    c = categorize(f)
    cats[c] = cats.get(c, 0) + 1
for c, n in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"    {c:<20} {n} features")

# ── CELL 4: Hyperparameters used ────────────────────────────────────
print("\n" + "=" * 70)
print("  CELL 4: Training Hyperparameters")
print("=" * 70)

print("\n  CatBoost (best):")
print(f"    Loss function:    Tweedie (variance_power=1.1)")
print(f"    Learning rate:    0.05")
print(f"    Depth:            8")
print(f"    Iterations:       500")
print(f"    Early stopping:   50 rounds")

print("\n  LightGBM:")
print(f"    Objective:        Tweedie (variance_power=1.1)")
print(f"    Learning rate:    0.05")
print(f"    Num leaves:       128")
print(f"    N estimators:     500")
print(f"    Feature fraction: 0.8")

print("\n  XGBoost:")
print(f"    Objective:        reg:tweedie (variance_power=1.1)")
print(f"    Learning rate:    0.05")
print(f"    Max depth:        8")
print(f"    N estimators:     500")
print(f"    Tree method:      hist")

# ── CELL 5: Sample predictions ──────────────────────────────────────
print("\n" + "=" * 70)
print("  CELL 5: Sample Predictions (5 random products)")
print("=" * 70)

feat_data = pd.read_parquet(FEAT_FILE)
last_day = feat_data["day_num"].max()
samples = feat_data[feat_data["day_num"] == last_day].sample(5, random_state=42)

print(f"\n  {'Store':<6} {'Item':<16} {'Actual':>7} {'CatBoost':>10} {'LightGBM':>10} {'XGBoost':>10}")
print(f"  {'─'*6} {'─'*16} {'─'*7} {'─'*10} {'─'*10} {'─'*10}")

for _, row in samples.iterrows():
    X = row[features].fillna(0).values.reshape(1, -1)
    actual = row["sales"]
    cb = max(0, float(models["catboost"].predict(X)[0]))
    lg = max(0, float(models["lightgbm"].predict(X)[0]))
    xg = max(0, float(models["xgboost"].predict(X)[0]))
    print(f"  {row['store_id']:<6} {row['item_id']:<16} {actual:>7.1f} {cb:>10.2f} {lg:>10.2f} {xg:>10.2f}")

# ── CELL 6: Model internals ─────────────────────────────────────────
print("\n" + "=" * 70)
print("  CELL 6: Model Internals")
print("=" * 70)

cb = models["catboost"]
print(f"\n  CatBoost:")
print(f"    Type:              {type(cb).__name__}")
print(f"    Total trees:       {cb.tree_count_}")
print(f"    Best iteration:    {cb.best_iteration_}")
print(f"    Learning rate:     {cb.learning_rate_}")

lg = models["lightgbm"]
print(f"\n  LightGBM:")
print(f"    Type:              {type(lg).__name__}")
print(f"    Best iteration:    {lg.best_iteration_}")
print(f"    Num leaves:        {lg.num_leaves}")
print(f"    N estimators:      {lg.n_estimators}")

xg = models["xgboost"]
print(f"\n  XGBoost:")
print(f"    Type:              {type(xg).__name__}")
print(f"    Best iteration:    {getattr(xg, 'best_iteration', 'N/A')}")
print(f"    N estimators:      {xg.n_estimators}")

print("\n" + "=" * 70)
print("  ALL 6 CELLS COMPLETE")
print("=" * 70)