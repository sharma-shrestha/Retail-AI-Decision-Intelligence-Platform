"""
Module 4 — Explainable AI (SHAP)
==================================
Provides SHAP-based explanations for model predictions.
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path

from src.utils.config import MODELS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SHAPExplainer:
    """Wrapper around SHAP for model explanations."""

    def __init__(self, model_name: str = "catboost"):
        self.model_name = model_name
        self.model_path = MODELS_DIR / f"{model_name}.pkl"
        self.meta_path = MODELS_DIR / f"{model_name}_meta.json"

        # Load model
        import joblib
        self.model = joblib.load(self.model_path)

        # Load metadata
        with open(self.meta_path) as f:
            self.meta = json.load(f)

        self.feature_cols = self.meta["features"]
        self.metrics = self.meta["metrics"]

        # Initialize SHAP explainer
        logger.info(f"Initializing SHAP explainer for {model_name}...")
        import shap

        if model_name == "lightgbm":
            self.explainer = shap.TreeExplainer(self.model)
        elif model_name == "xgboost":
            self.explainer = shap.TreeExplainer(self.model)
        elif model_name == "catboost":
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(self.model.predict, shap.sample(X, 100))

        logger.info(f"  Model: {model_name}, Features: {len(self.feature_cols)}")

    def explain_single(self, features_df: pd.DataFrame, row_idx: int = 0) -> dict:
        """Explain a single prediction.

        Returns a dict with:
          - prediction: the forecasted value
          - base_value: SHAP expected value
          - shap_values: per-feature importance
          - top_contributors: top 5 positive/negative contributors
        """
        import shap

        X = features_df[self.feature_cols].fillna(0)
        row = X.iloc[[row_idx]]

        shap_values = self.explainer.shap_values(row)
        prediction = float(self.model.predict(row)[0])
        base_value = float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0

        # Build per-feature explanation
        contributions = []
        for i, feat in enumerate(self.feature_cols):
            contributions.append({
                "feature": feat,
                "value": float(row[feat].values[0]),
                "shap_value": float(shap_values[0][i]) if isinstance(shap_values[0], (list, np.ndarray)) else float(shap_values[i]),
            })

        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        # Top 5 positive and negative
        top_pos = [c for c in contributions if c["shap_value"] > 0][:5]
        top_neg = [c for c in contributions if c["shap_value"] < 0][:5]

        return {
            "model": self.model_name,
            "prediction": round(prediction, 4),
            "base_value": round(base_value, 4),
            "top_positive_contributors": [
                {"feature": c["feature"], "impact": round(c["shap_value"], 4), "value": round(c["value"], 4)}
                for c in top_pos
            ],
            "top_negative_contributors": [
                {"feature": c["feature"], "impact": round(c["shap_value"], 4), "value": round(c["value"], 4)}
                for c in top_neg
            ],
            "all_contributions": contributions[:20],
        }

    def global_feature_importance(self) -> list[dict]:
        """Get global feature importance from SHAP values on a sample of data."""
        from src.models.train import load_data
        import shap

        _, _, X_val, _, _ = load_data()
        X = X_val[self.feature_cols].fillna(0)

        # Sample for speed
        n_sample = min(2000, len(X))
        X_sample = X.sample(n=n_sample, random_state=42)

        logger.info(f"Computing global SHAP importance on {n_sample} samples...")
        shap_values = self.explainer.shap_values(X_sample)

        # Mean absolute SHAP value per feature
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance = []
        for i, feat in enumerate(self.feature_cols):
            importance.append({
                "rank": i + 1,
                "feature": feat,
                "mean_abs_shap": round(float(mean_abs_shap[i]), 6),
            })

        importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
        # Re-rank after sort
        for i, item in enumerate(importance):
            item["rank"] = i + 1

        return importance


def explain_forecast(
    features_df: pd.DataFrame,
    row_idx: int = 0,
    model_name: str = "catboost",
) -> dict:
    """High-level function: explain a forecast prediction."""
    explainer = SHAPExplainer(model_name)
    return explainer.explain_single(features_df, row_idx)


if __name__ == "__main__":
    # Test explainability
    from src.models.train import load_data

    X_train, y_train, X_val, y_val, feature_cols = load_data()

    # Explain first validation sample
    explainer = SHAPExplainer("catboost")
    explanation = explainer.explain_single(X_val, row_idx=0)

    print("\n=== Forecast Explanation ===")
    print(f"Prediction: {explanation['prediction']}")
    print(f"Base Value: {explanation['base_value']}")
    print(f"\nTop Positive Contributors:")
    for c in explanation["top_positive_contributors"]:
        print(f"  +{c['impact']:.4f}  {c['feature']} (value={c['value']})")
    print(f"\nTop Negative Contributors:")
    for c in explanation["top_negative_contributors"]:
        print(f"  {c['impact']:.4f}  {c['feature']} (value={c['value']})")

    # Global importance
    print("\n\n=== Top 15 Global Feature Importance ===")
    importance = explainer.global_feature_importance()
    for item in importance[:15]:
        print(f"  {item['rank']:3d}. {item['feature']:<25} SHAP={item['mean_abs_shap']:.6f}")