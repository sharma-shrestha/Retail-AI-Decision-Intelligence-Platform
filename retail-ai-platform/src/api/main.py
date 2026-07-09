"""
Module 8 — FastAPI Application
================================
REST API for the Retail AI Decision Intelligence Platform.

Endpoints:
  GET  /health                    — Health check
  GET  /api/v1/models             — List trained models & metrics
  POST /api/v1/forecast           — Generate demand forecast
  POST /api/v1/explain            — Explain a forecast with SHAP
  POST /api/v1/inventory          — Get inventory recommendations
  GET  /api/v1/analytics/summary  — Business analytics summary
  GET  /api/v1/analytics/top-products  — Top performing products
  GET  /api/v1/analytics/feature-importance — Global SHAP importance
"""

import json
import os
import logging
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.config import (
    PROCESSED_SALES, FEATURES_FILE, MODELS_DIR,
    VALIDATION_DAYS, HORIZON,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Retail AI Decision Intelligence Platform",
    description="AI-powered demand forecasting, explainability, and inventory recommendations using M5 retail data.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── State ────────────────────────────────────────────────────────────────────
class AppState:
    """Lazy-loaded app state."""
    _models: dict = {}
    _features: list = []
    _processed_data: pd.DataFrame | None = None
    _feature_data: pd.DataFrame | None = None

    @classmethod
    def load_models(cls):
        if cls._models:
            return cls._models
        for name in ["lightgbm", "xgboost", "catboost"]:
            path = MODELS_DIR / f"{name}.pkl"
            meta_path = MODELS_DIR / f"{name}_meta.json"
            if path.exists():
                cls._models[name] = joblib.load(path)
                with open(meta_path) as f:
                    cls._models[f"{name}_meta"] = json.load(f)
                logger.info(f"Loaded model: {name}")
        return cls._models

    @classmethod
    def get_features(cls) -> list:
        if cls._features:
            return cls._features
        cls.load_models()
        for name in ["catboost", "lightgbm", "xgboost"]:
            key = f"{name}_meta"
            if key in cls._models:
                cls._features = cls._models[key]["features"]
                break
        return cls._features

    @classmethod
    def get_processed_data(cls) -> pd.DataFrame:
        if cls._processed_data is None:
            cls._processed_data = pd.read_parquet(PROCESSED_SALES)
        return cls._processed_data

    @classmethod
    def get_feature_data(cls) -> pd.DataFrame:
        if cls._feature_data is None:
            cls._feature_data = pd.read_parquet(FEATURES_FILE)
        return cls._feature_data


# ─── Request/Response Models ──────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    store_id: str
    item_id: str
    model: str = Field(default="catboost", pattern="^(lightgbm|xgboost|catboost)$")
    days: int = Field(default=28, ge=1, le=28)


class ExplainRequest(BaseModel):
    store_id: str
    item_id: str
    model: str = Field(default="catboost", pattern="^(lightgbm|xgboost|catboost)$")


class InventoryRequest(BaseModel):
    store_id: str
    item_id: str
    model: str = Field(default="catboost", pattern="^(lightgbm|xgboost|catboost)$")
    current_stock: Optional[float] = None
    lead_time_days: int = Field(default=7, ge=1)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "healthy", "service": "retail-ai-platform", "version": "1.0.0"}


@app.get("/api/v1/models")
def list_models():
    """List all trained models with their metrics."""
    models = AppState.load_models()
    result = {}
    for name in ["lightgbm", "xgboost", "catboost"]:
        meta_key = f"{name}_meta"
        if meta_key in models:
            result[name] = {
                "status": "loaded",
                "metrics": models[meta_key]["metrics"],
                "features": len(models[meta_key]["features"]),
            }
        else:
            result[name] = {"status": "not_available"}
    return {"models": result}


@app.post("/api/v1/forecast")
def forecast(req: ForecastRequest):
    """Generate demand forecast for a product-store combination.

    Returns predicted sales for the next N days along with the prediction
    and confidence indicators.
    """
    start = time.time()

    # Load model and data
    models = AppState.load_models()
    if req.model not in models:
        raise HTTPException(404, f"Model '{req.model}' not found")

    model = models[req.model]
    feature_cols = AppState.get_features()

    # Get the latest data row for this product-store
    data = AppState.get_processed_data()
    mask = (data["store_id"] == req.store_id) & (data["item_id"] == req.item_id)

    if not mask.any():
        raise HTTPException(404, f"No data found for store={req.store_id}, item={req.item_id}")

    # Get last row as the base for prediction
    latest = data[mask].sort_values("day_num").iloc[-1:]

    # Load feature data for full context
    feat_data = AppState.get_feature_data()
    feat_mask = (feat_data["store_id"] == req.store_id) & (feat_data["item_id"] == req.item_id)
    feat_rows = feat_data[feat_mask].sort_values("day_num")

    if len(feat_rows) == 0:
        raise HTTPException(404, f"No feature data for store={req.store_id}, item={req.item_id}")

    # Use the last row for prediction
    last_row = feat_rows.iloc[[-1]]
    X = last_row[feature_cols].fillna(0)
    prediction = float(model.predict(X)[0])
    prediction = max(0, prediction)  # No negative sales

    # Also get validation metrics
    meta = models[f"{req.model}_meta"]

    # Historical stats
    sales_series = data.loc[mask, "sales"].values
    hist_mean = float(np.mean(sales_series))
    hist_std = float(np.std(sales_series))
    hist_max = float(np.max(sales_series))

    elapsed = time.time() - start

    return {
        "store_id": req.store_id,
        "item_id": req.item_id,
        "model": req.model,
        "forecast": {
            "predicted_daily_sales": round(prediction, 4),
            "predicted_period_sales": round(prediction * req.days, 2),
            "forecast_days": req.days,
            "confidence": "The prediction is based on 80 engineered features including lag sales, rolling statistics, price dynamics, and calendar effects.",
        },
        "historical_context": {
            "mean_daily_sales": round(hist_mean, 2),
            "std_daily_sales": round(hist_std, 2),
            "max_daily_sales": round(hist_max, 2),
            "data_points": len(sales_series),
        },
        "model_metrics": meta["metrics"],
        "latency_ms": round(elapsed * 1000, 1),
    }


@app.post("/api/v1/explain")
def explain(req: ExplainRequest):
    """Explain a forecast using SHAP values.

    Returns the prediction alongside feature-level contributions
    explaining WHY the model made that prediction.
    """
    start = time.time()

    # Lazy import SHAP explainer
    from src.explainability.shap_explainer import SHAPExplainer

    models = AppState.load_models()
    if req.model not in models:
        raise HTTPException(404, f"Model '{req.model}' not found")

    # Get feature data for this product-store
    feat_data = AppState.get_feature_data()
    mask = (feat_data["store_id"] == req.store_id) & (feat_data["item_id"] == req.item_id)

    if not mask.any():
        raise HTTPException(404, f"No data for store={req.store_id}, item={req.item_id}")

    series_data = feat_data[mask].sort_values("day_num").reset_index(drop=True)

    # Use the most recent row
    explainer = SHAPExplainer(req.model)
    explanation = explainer.explain_single(series_data, row_idx=-1)

    # Add product context
    data = AppState.get_processed_data()
    prod_mask = (data["store_id"] == req.store_id) & (data["item_id"] == req.item_id)
    latest_sales = float(data[prod_mask].sort_values("day_num").iloc[-1]["sales"])

    elapsed = time.time() - start

    return {
        "store_id": req.store_id,
        "item_id": req.item_id,
        "model": req.model,
        "latest_actual_sales": latest_sales,
        "explanation": explanation,
        "interpretation": (
            f"The model predicts {explanation['prediction']:.2f} units. "
            f"The primary drivers are: "
            + ", ".join([f"{c['feature']} (+{c['impact']:.4f})" for c in explanation["top_positive_contributors"][:3]])
            + ". "
            + "Key dampening factors: "
            + ", ".join([f"{c['feature']} ({c['impact']:.4f})" for c in explanation["top_negative_contributors"][:3]])
            + "."
        ),
        "latency_ms": round(elapsed * 1000, 1),
    }


@app.post("/api/v1/inventory")
def inventory_recommendation(req: InventoryRequest):
    """Get inventory action recommendation based on demand forecast.

    Combines the forecasting model with safety stock calculations
    to recommend how much to order and why.
    """
    from src.inventory.engine import generate_inventory_recommendation

    # Get forecast first
    models = AppState.load_models()
    if req.model not in models:
        raise HTTPException(404, f"Model '{req.model}' not found")

    model = models[req.model]
    feature_cols = AppState.get_features()

    # Get latest prediction
    feat_data = AppState.get_feature_data()
    mask = (feat_data["store_id"] == req.store_id) & (feat_data["item_id"] == req.item_id)
    if not mask.any():
        raise HTTPException(404, f"No data for store={req.store_id}, item={req.item_id}")

    last_row = feat_data[mask].sort_values("day_num").iloc[[-1]]
    X = last_row[feature_cols].fillna(0)
    daily_prediction = max(0, float(model.predict(X)[0]))

    # Get demand std
    data = AppState.get_processed_data()
    prod_mask = (data["store_id"] == req.store_id) & (data["item_id"] == req.item_id)
    demand_std = float(data[prod_mask]["sales"].std()) if prod_mask.any() else 1.0

    # Generate recommendation
    rec = generate_inventory_recommendation(
        forecasted_demand=daily_prediction * 28,
        current_stock=req.current_stock,
        demand_std=demand_std,
        lead_time_days=req.lead_time_days,
        days_ahead=28,
    )
    rec["store_id"] = req.store_id
    rec["item_id"] = req.item_id
    rec["model"] = req.model
    rec["daily_forecast"] = round(daily_prediction, 4)

    return rec


@app.get("/api/v1/analytics/summary")
def analytics_summary():
    """Business analytics summary across all products and stores."""
    data = AppState.get_processed_data()
    models = AppState.load_models()

    # KPIs
    total_sales = int(data["sales"].sum())
    avg_daily_sales = round(float(data["sales"].mean()), 2)
    total_revenue = round(float((data["sales"] * data["sell_price"]).sum()), 2)
    unique_products = data["item_id"].nunique()
    unique_stores = data["store_id"].nunique()
    zero_sales_pct = round(float((data["sales"] == 0).mean() * 100), 1)

    # Per-store summary
    store_stats = data.groupby("store_id", observed=True).agg(
        total_sales=("sales", "sum"),
        avg_daily_sales=("sales", "mean"),
        avg_price=("sell_price", "mean"),
        n_products=("item_id", "nunique"),
    ).round(2).to_dict(orient="index")

    # Per-category summary
    cat_stats = data.groupby("cat_id", observed=True).agg(
        total_sales=("sales", "sum"),
        avg_daily_sales=("sales", "mean"),
        avg_price=("sell_price", "mean"),
        n_products=("item_id", "nunique"),
    ).round(2).to_dict(orient="index")

    # Model performance
    model_metrics = {}
    for name in ["lightgbm", "xgboost", "catboost"]:
        meta_key = f"{name}_meta"
        if meta_key in models:
            model_metrics[name] = models[meta_key]["metrics"]

    return {
        "overview": {
            "total_sales": total_sales,
            "total_revenue": total_revenue,
            "avg_daily_sales": avg_daily_sales,
            "unique_products": unique_products,
            "unique_stores": unique_stores,
            "zero_sales_pct": zero_sales_pct,
            "date_range": f"{data['date'].min().date()} to {data['date'].max().date()}",
            "total_data_points": len(data),
        },
        "by_store": store_stats,
        "by_category": cat_stats,
        "model_performance": model_metrics,
    }


@app.get("/api/v1/analytics/top-products")
def top_products(
    n: int = Query(default=10, ge=1, le=50),
    metric: str = Query(default="total_sales", pattern="^(total_sales|avg_daily|revenue)$"),
):
    """Get top N performing products."""
    data = AppState.get_processed_data()

    if metric == "total_sales":
        ranking = data.groupby(["item_id", "store_id"], observed=True)["sales"].sum().reset_index()
        ranking = ranking.sort_values("sales", ascending=False).head(n)
        ranking.columns = ["item_id", "store_id", "total_sales"]
    elif metric == "avg_daily":
        ranking = data.groupby(["item_id", "store_id"], observed=True)["sales"].mean().reset_index()
        ranking = ranking.sort_values("sales", ascending=False).head(n)
        ranking.columns = ["item_id", "store_id", "avg_daily_sales"]
    elif metric == "revenue":
        data["revenue"] = data["sales"] * data["sell_price"]
        ranking = data.groupby(["item_id", "store_id"], observed=True)["revenue"].sum().reset_index()
        ranking = ranking.sort_values("revenue", ascending=False).head(n)
        ranking.columns = ["item_id", "store_id", "total_revenue"]

    return {
        "metric": metric,
        "top_n": n,
        "products": ranking.round(2).to_dict(orient="records"),
    }


@app.get("/api/v1/analytics/feature-importance")
def feature_importance(
    model: str = Query(default="catboost", pattern="^(lightgbm|xgboost|catboost)$"),
    top_n: int = Query(default=20, ge=1, le=80),
):
    """Get global SHAP feature importance for a model."""
    from src.explainability.shap_explainer import SHAPExplainer

    try:
        explainer = SHAPExplainer(model)
        importance = explainer.global_feature_importance()
        return {
            "model": model,
            "total_features": len(importance),
            "top_features": importance[:top_n],
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to compute feature importance: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
#  GenAI Endpoints — AI Copilot (RAG)
# ══════════════════════════════════════════════════════════════════════════════

class AskAIRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question about retail data")
    store_id: Optional[str] = None
    item_id: Optional[str] = None
    enrich_shap: bool = False


@app.post("/api/v1/ask-ai")
def ask_ai(req: AskAIRequest):
    """Retail AI Copilot — ask any business question in natural language.

    Uses RAG (Retrieval-Augmented Generation) to answer questions like:
    - "Which products need restocking?"
    - "Why will milk sales increase?"
    - "Which promotions actually worked?"
    - "What is safety stock?"

    Set GEMINI_API_KEY environment variable for AI-generated responses.
    Without it, a built-in NLG engine provides structured data-driven answers.
    """
    start = time.time()

    from src.genai.rag_pipeline import rag_query

    result = rag_query(
        question=req.question,
        top_k=8,
        enrich_with_shap=req.enrich_shap,
        store_id=req.store_id,
        item_id=req.item_id,
    )

    elapsed = time.time() - start

    return {
        "question": req.question,
        "answer": result["answer"],
        "mode": result["mode"],
        "llm_hint": "gemini" if not os.environ.get("GEMINI_API_KEY") else "nlg_fallback",
        "sources_count": result["retrieval_count"],
        "sources": result["sources"],
        "latency_ms": round(elapsed * 1000, 1),
    }


@app.get("/api/v1/ask-ai/report")
def daily_report():
    """Auto-generated Daily Retail Intelligence Report.

    Returns a structured report with:
    - Top products by forecasted demand
    - Risk/underperforming products
    - Store performance summary
    - Category breakdown
    - Model performance metrics
    - Actionable recommendations
    """
    start = time.time()

    from src.genai.rag_pipeline import generate_daily_report

    report = generate_daily_report()

    elapsed = time.time() - start

    return {
        "report": report,
        "generated_at": pd.Timestamp.now().isoformat(),
        "latency_ms": round(elapsed * 1000, 1),
    }


@app.get("/api/v1/ask-ai/underperforming")
def underperforming_products(
    store_id: Optional[str] = Query(default=None, description="Filter by store"),
    n: int = Query(default=10, ge=1, le=50, description="Number of products"),
):
    """Get underperforming products that need attention.

    Returns products in the bottom 10% by average daily sales,
    with recommendations for each (promotion, repricing, or delisting).
    """
    start = time.time()

    from src.genai.rag_pipeline import rag_query

    query = f"underperforming products at {store_id}" if store_id else "underperforming products"
    result = rag_query(query, top_k=5, doc_type="underperformer")

    # Also get structured data
    from src.genai.knowledge_base import get_knowledge_base
    kb = get_knowledge_base()
    all_underperformers = kb.retrieve("underperforming low sales bottom", top_k=n, doc_type="underperformer")

    # Filter by store if specified
    if store_id:
        all_underperformers = [
            d for d in all_underperformers
            if d["metadata"].get("store_id") == store_id
        ]

    products = []
    for d in all_underperformers:
        meta = d["metadata"]
        avg = meta.get("avg_sales", 0)
        if avg < 0.2:
            action = "delist"
            reasoning = "Critical: average < 0.2/day. Consider delisting to free shelf space."
        elif avg < 0.5:
            action = "promote"
            reasoning = "Low: average 0.2-0.5/day. Test price promotions or end-cap placement."
        else:
            action = "monitor"
            reasoning = "Below average: monitor for 2-4 weeks before action."

        products.append({
            "item_id": meta.get("item_id"),
            "store_id": meta.get("store_id"),
            "avg_daily_sales": round(avg, 3),
            "action": action,
            "reasoning": reasoning,
        })

    elapsed = time.time() - start

    return {
        "total_underperforming": len(products),
        "products": products,
        "ai_summary": result["answer"][:500],
        "latency_ms": round(elapsed * 1000, 1),
    }


# ─── Start ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from src.utils.config import API_HOST, API_PORT
    logger.info("Starting Retail AI Platform API...")
    uvicorn.run(app, host=API_HOST, port=API_PORT)