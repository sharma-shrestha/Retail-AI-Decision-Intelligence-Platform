"""
Module 6 — RAG Pipeline
=========================
Orchestrates the full Retrieval-Augmented Generation flow:

  User Query
      │
      ▼
  Knowledge Base (TF-IDF retrieval)
      │
      ▼
  Optional: SHAP / Forecast data enrichment
      │
      ▼
  LLM Interface (NLG fallback or Gemini)
      │
      ▼
  Structured Response
"""

import logging
from typing import Optional

from src.genai.knowledge_base import get_knowledge_base
from src.genai.llm_interface import generate_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def rag_query(
    question: str,
    top_k: int = 8,
    doc_type: Optional[str] = None,
    enrich_with_shap: bool = False,
    store_id: Optional[str] = None,
    item_id: Optional[str] = None,
) -> dict:
    """Full RAG pipeline: retrieve → enrich → generate.

    Args:
        question: User's natural language question
        top_k: Number of documents to retrieve
        doc_type: Optional filter (forecast, catalog, underperformer, etc.)
        enrich_with_shap: If True, add SHAP explanation data for the specified product
        store_id: Optional store filter for enrichment
        item_id: Optional item filter for enrichment

    Returns:
        Full RAG response dict
    """
    # Step 1: Retrieve from knowledge base
    kb = get_knowledge_base()

    # Build enhanced query
    enhanced_query = question
    if store_id:
        enhanced_query += f" store {store_id}"
    if item_id:
        enhanced_query += f" product {item_id}"

    retrieved_docs = kb.retrieve(enhanced_query, top_k=top_k, doc_type=doc_type)

    # Step 2: Optional enrichment
    extra_context = {}

    if enrich_with_shap and store_id and item_id:
        try:
            from src.explainability.shap_explainer import SHAPExplainer
            from src.models.train import load_data
            import json, os

            # Find best model
            best_model = "catboost"
            for name in ["catboost", "lightgbm", "xgboost"]:
                meta_path = os.path.join(os.path.dirname(__file__), "..", "models", "saved", f"{name}_meta.json")
                # Use config path instead
            from src.utils.config import MODELS_DIR
            for name in ["catboost", "lightgbm", "xgboost"]:
                mp = MODELS_DIR / f"{name}_meta.json"
                if mp.exists():
                    best_model = name
                    break

            _, _, X_val, _, _ = load_data()
            mask = (X_val["store_id"] == store_id) & (X_val["item_id"] == item_id)
            if mask.any():
                series_data = X_val[mask].sort_values("day_num").reset_index(drop=True)
                explainer = SHAPExplainer(best_model)
                shap_result = explainer.explain_single(series_data, row_idx=-1)
                extra_context["shap_explanation"] = shap_result
                logger.info(f"Enriched with SHAP for {item_id} @ {store_id}")
        except Exception as e:
            logger.warning(f"SHAP enrichment failed: {e}")

    # Step 3: Generate response
    result = generate_response(question, retrieved_docs, extra_context)

    # Add metadata
    result["retrieval_count"] = len(retrieved_docs)
    result["enriched"] = bool(extra_context)

    return result


def generate_daily_report() -> dict:
    """Auto-generate a daily retail intelligence report.

    Returns structured report with:
      - Top products by forecasted demand
      - Underperforming products needing attention
      - Store performance summary
      - Key metrics and recommendations
    """
    kb = get_knowledge_base()

    # Retrieve data for report sections
    top_forecasts = kb.retrieve("highest predicted demand forecast", top_k=5, doc_type="forecast")
    underperformers = kb.retrieve("underperforming low sales bottom", top_k=10, doc_type="underperformer")
    store_patterns = kb.retrieve("store total sales performance", top_k=10, doc_type="store_pattern")
    cat_patterns = kb.retrieve("category total sales average price", top_k=5, doc_type="category_pattern")

    # Load model metrics
    model_metrics = {}
    from src.utils.config import MODELS_DIR
    import json
    for name in ["catboost", "lightgbm", "xgboost"]:
        meta_path = MODELS_DIR / f"{name}_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                model_metrics[name] = json.load(f)["metrics"]

    # Build report
    report = {
        "title": "Daily Retail Intelligence Report",
        "sections": {
            "top_products": {
                "description": "Products with highest forecasted demand — prioritize for restocking",
                "items": [
                    {
                        "item_id": d["metadata"].get("item_id"),
                        "store_id": d["metadata"].get("store_id"),
                        "predicted_daily": round(d["metadata"].get("prediction", 0), 2),
                        "model": d["metadata"].get("model"),
                    }
                    for d in top_forecasts
                ],
            },
            "risk_products": {
                "description": "Bottom 10% performers — consider promotions, repricing, or delisting",
                "items": [
                    {
                        "item_id": d["metadata"].get("item_id"),
                        "store_id": d["metadata"].get("store_id"),
                        "avg_daily_sales": round(d["metadata"].get("avg_sales", 0), 2),
                    }
                    for d in underperformers[:10]
                ],
            },
            "store_performance": {
                "description": "Store-level sales summary",
                "items": [d["content"] for d in store_patterns[:10]],
            },
            "category_breakdown": {
                "description": "Category-level performance",
                "items": [d["content"] for d in cat_patterns[:5]],
            },
            "model_performance": {
                "description": "Forecasting model accuracy",
                "metrics": model_metrics,
            },
        },
        "recommendations": [
            "Review top 5 products by forecasted demand for immediate restocking",
            "Investigate underperforming products — test price promotions on items with 0.2-0.5 daily avg",
            "Consider delisting products with < 0.2 daily sales to optimize shelf space",
            "Monitor SNAP distribution days for food category demand spikes",
        ],
    }

    return report


if __name__ == "__main__":
    import json

    # Test RAG queries
    queries = [
        "Which products need restocking?",
        "Show me underperforming products",
        "Why would sales increase?",
        "What is safety stock and how is it calculated?",
        "Which store performs best?",
    ]

    for q in queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = rag_query(q, top_k=5)
        print(f"Mode: {result['mode']}")
        print(f"Sources: {result['retrieval_count']}")
        print(f"A: {result['answer'][:300]}...")

    # Test report generation
    print(f"\n{'='*60}")
    print("DAILY REPORT")
    report = generate_daily_report()
    print(json.dumps(report, indent=2, default=str)[:1000])