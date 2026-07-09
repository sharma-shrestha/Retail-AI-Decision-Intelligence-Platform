"""
Module 6 — Modular LLM Interface
====================================
Two modes:
  Mode A (default): NLG Engine — rule-based natural language generation from retrieved context
  Mode B (Gemini):  Google Gemini API — set GEMINI_API_KEY env var to activate

Both modes receive the same structured input:
  - query: user's question
  - retrieved_docs: list of {content, score, metadata, type} from knowledge base
  - extra_context: optional dict with forecast values, SHAP data, etc.
"""

import os
import json
import logging
import re
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"


# ══════════════════════════════════════════════════════════════════════════════
#  Mode B: Gemini LLM
# ══════════════════════════════════════════════════════════════════════════════

def _call_gemini(prompt: str) -> str:
    """Call Gemini API with the given prompt."""
    import google.generativeai as genai

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    response = model.generate_content(prompt)
    return response.text


def _build_gemini_prompt(query: str, retrieved_docs: list[dict], extra_context: dict | None = None) -> str:
    """Build the Gemini prompt with retrieved context."""
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:8]):
        context_parts.append(f"[{i+1}] ({doc['type']}) {doc['content']}")

    context_str = "\n".join(context_parts)

    extra = ""
    if extra_context:
        for k, v in extra_context.items():
            if isinstance(v, (dict, list)):
                extra += f"\n{k}: {json.dumps(v, indent=2, default=str)}"
            else:
                extra += f"\n{k}: {v}"

    prompt = f"""You are an expert retail analytics AI assistant for a Walmart-like retail chain.
Answer the user's question using the retrieved context below. Be specific, data-driven, and actionable.
If the context doesn't contain enough information, say so honestly.

=== RETRIEVED CONTEXT ===
{context_str}
=== ADDITIONAL DATA ===
{extra if extra else "None"}
=== USER QUESTION ===
{query}

Provide a clear, structured answer with specific numbers when available. Use bullet points for lists."""

    return prompt


# ══════════════════════════════════════════════════════════════════════════════
#  Mode A: NLG Engine (Rule-Based Natural Language Generation)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_numbers(text: str) -> list[float]:
    """Extract numbers from text."""
    return [float(x) for x in re.findall(r"[\d]+\.?\d*", text)]


def _generate_restocking_answer(query: str, docs: list[dict]) -> str:
    """Generate answer for restocking-related queries."""
    # If no forecast docs retrieved, get them directly from KB
    forecasts = [d for d in docs if d["type"] == "forecast"]
    catalogs = [d for d in docs if d["type"] == "catalog"]

    if not forecasts:
        try:
            from src.genai.knowledge_base import get_knowledge_base
            kb = get_knowledge_base()
            forecasts = kb.retrieve("highest predicted demand forecast", top_k=10, doc_type="forecast")
            catalogs = kb.retrieve("product store catalog", top_k=20, doc_type="catalog")
        except Exception:
            pass

    # Get top items by predicted sales
    items = []
    for doc in forecasts[:10]:
        meta = doc.get("metadata", {})
        pred = meta.get("prediction", 0)
        if pred > 0:
            items.append({
                "item_id": meta.get("item_id", "unknown"),
                "store_id": meta.get("store_id", "unknown"),
                "prediction": round(pred, 2),
            })

    items.sort(key=lambda x: x["prediction"], reverse=True)
    top_items = items[:5]

    if not top_items:
        return "Based on the current data, all products have low predicted demand. No immediate restocking is needed."

    answer = "Based on the demand forecasting models, here are the products with the **highest predicted demand** that may need restocking:\n\n"
    for i, item in enumerate(top_items, 1):
        # Find matching catalog entry for more context
        matching_cat = next(
            (d for d in catalogs if d["metadata"].get("item_id") == item["item_id"]
             and d["metadata"].get("store_id") == item["store_id"]), None
        )
        extra = ""
        if matching_cat:
            nums = _extract_numbers(matching_cat["content"])
            if len(nums) >= 2:
                extra = f" (historical avg: {nums[0]:.1f}/day)"

        answer += f"**{i}. {item['item_id']} @ {item['store_id']}** — Predicted: {item['prediction']:.1f} units/day{extra}\n"

    total_28d = sum(i["prediction"] * 28 for i in top_items)
    answer += f"\n**Total 28-day demand for top 5:** ~{total_28d:,.0f} units\n"
    answer += "\n> **Recommendation:** Prioritize these high-velocity items for replenishment. Use the `/inventory` endpoint for specific order quantities with safety stock calculations."

    return answer


def _generate_underperforming_answer(query: str, docs: list[dict]) -> str:
    """Generate answer for underperforming product queries."""
    underperformers = [d for d in docs if d["type"] == "underperformer"]

    if not underperformers:
        try:
            from src.genai.knowledge_base import get_knowledge_base
            kb = get_knowledge_base()
            underperformers = kb.retrieve("underperforming low sales bottom", top_k=15, doc_type="underperformer")
        except Exception:
            pass

    if not underperformers:
        return "No underperforming products identified in the current dataset. All products are performing within normal range."

    answer = f"Here are **{len(underperformers)}** products in the **bottom 10%** by average daily sales that may need attention:\n\n"

    # Group by store
    by_store = {}
    for doc in underperformers[:15]:
        store = doc["metadata"].get("store_id", "unknown")
        item = doc["metadata"].get("item_id", "unknown")
        avg = doc["metadata"].get("avg_sales", 0)
        if store not in by_store:
            by_store[store] = []
        by_store[store].append((item, avg))

    for store, items in sorted(by_store.items()):
        answer += f"**Store {store}:**\n"
        for item, avg in sorted(items, key=lambda x: x[1])[:5]:
            if avg < 0.2:
                status = "Critical — consider delisting"
            elif avg < 0.5:
                status = "Low — needs promotion or repricing"
            else:
                status = "Below average — monitor"
            answer += f"  - {item}: {avg:.2f}/day ({status})\n"
        answer += "\n"

    answer += "> **Recommendation:** For products averaging < 0.2/day, consider delisting to free shelf space. For products at 0.2-0.5/day, test price promotions or end-cap placement."

    return answer


def _generate_explanation_answer(query: str, docs: list[dict]) -> str:
    """Generate answer for 'why' questions about demand changes."""
    # Look for business knowledge docs
    biz_docs = [d for d in docs if d["type"] == "business_knowledge"]
    catalog_docs = [d for d in docs if d["type"] == "catalog"]
    forecast_docs = [d for d in docs if d["type"] == "forecast"]

    answer = "Based on the retail knowledge base and forecast data, here are the **key drivers of demand changes**:\n\n"

    # Business knowledge explanations
    factors = []
    for doc in biz_docs[:5]:
        content = doc["content"]
        # Extract key concepts
        if "SNAP" in content:
            factors.append("**SNAP Benefits Distribution:** When SNAP benefits are distributed (1st-15th of month in most states), eligible food product sales typically spike 15-30%.")
        if "weekend" in query.lower() or "weekend" in content.lower():
            factors.append("**Weekend Effect:** Consumer shopping patterns shift on weekends. Many categories see higher foot traffic and impulse purchases on Saturdays and Sundays.")
        if "holiday" in query.lower() or "event" in content.lower():
            factors.append("**Calendar Events:** Holidays and events (Christmas, Thanksgiving, Super Bowl) create significant demand spikes. The model captures these through event indicator features.")
        if "price" in query.lower() or "price" in content.lower():
            factors.append("**Price Dynamics:** Price reductions and promotions directly drive demand increases. The model tracks price changes, discounts, and moving average prices to capture this effect.")
        if "lag" in content.lower() or "seasonal" in query.lower():
            factors.append("**Seasonal & Lag Patterns:** Sales exhibit weekly (7-day), monthly (28-day), and seasonal cycles. The model uses lag features (1, 7, 14, 28, 56 days) and rolling statistics to capture these patterns.")

    if not factors:
        factors.append("**Historical Trend:** The primary driver is recent sales patterns captured through lag features and rolling statistics.")
        factors.append("**Category Dynamics:** Category-level and store-level average sales provide baseline demand expectations.")

    for f in factors:
        answer += f"- {f}\n"

    # Add specific product context if available
    if forecast_docs:
        top_forecast = forecast_docs[0]
        meta = top_forecast["metadata"]
        answer += f"\n**Example:** {meta.get('item_id', 'N/A')} at {meta.get('store_id', 'N/A')} has a predicted daily demand of **{meta.get('prediction', 0):.1f} units**."
        if catalog_docs:
            cat = catalog_docs[0]
            nums = _extract_numbers(cat["content"])
            if nums:
                answer += f" Historical average was {nums[0]:.1f}/day."

    answer += "\n\n> **Tip:** Use the `/explain` endpoint with a specific product-store combination to get the exact SHAP feature contributions for any prediction."

    return answer


def _generate_kpi_answer(query: str, docs: list[dict]) -> str:
    """Generate answer for KPI and metric questions."""
    biz_docs = [d for d in docs if d["type"] == "business_knowledge"]

    answer = "Here are the **key performance indicators** used in this platform:\n\n"

    for doc in biz_docs[:6]:
        content = doc["content"]
        # Extract the main concept
        if "RMSE" in content:
            answer += "- **RMSE (Root Mean Square Error):** Measures forecast accuracy. Lower is better. Penalizes large errors more than MAE.\n"
        elif "RMSSE" in content:
            answer += "- **RMSSE (Root Mean Squared Scaled Error):** Scaled RMSE where < 1.0 means the model beats a naive baseline. The M5 competition metric.\n"
        elif "Safety stock" in content:
            answer += "- **Safety Stock:** z x sigma x sqrt(lead_time). Buffers against demand variability. Uses z=1.645 for 95% service level.\n"
        elif "Reorder Point" in content:
            answer += "- **Reorder Point:** (daily demand x lead time) + safety stock. Triggers replenishment orders.\n"
        elif "SHAP" in content:
            answer += "- **SHAP Values:** Per-feature contribution to each prediction. Positive = increases forecast, negative = decreases.\n"
        elif "Tweedie" in content:
            answer += "- **Tweedie Loss:** Handles zero-inflated continuous retail data (many zero-sale days). Variance power 1.1.\n"

    answer += "\n> **Current Model Performance:** Use the `/api/v1/models` endpoint to see real-time RMSE, MAE, and RMSSE for each trained model."

    return answer


def _generate_store_comparison_answer(query: str, docs: list[dict]) -> str:
    """Generate answer for store comparison queries."""
    store_docs = [d for d in docs if d["type"] == "store_pattern"]
    cat_docs = [d for d in docs if d["type"] == "category_pattern"]

    if not store_docs:
        return "Store data is not available in the knowledge base yet."

    answer = "Here's the **store performance comparison**:\n\n"
    answer += "| Store | Total Sales | Avg Daily | Best Day |\n"
    answer += "|-------|-------------|-----------|----------|\n"

    store_data = []
    for doc in store_docs:
        nums = _extract_numbers(doc["content"])
        store_id = doc["metadata"].get("store_id", "?")
        if len(nums) >= 3:
            store_data.append((store_id, int(nums[0]), round(nums[1], 2), int(nums[2])))

    store_data.sort(key=lambda x: x[1], reverse=True)
    for store, total, avg, best in store_data:
        answer += f"| {store} | {total:,} | {avg} | {best} |\n"

    # Category breakdown
    if cat_docs:
        answer += "\n**Category Performance:**\n"
        for doc in cat_docs:
            answer += f"- {doc['content']}\n"

    return answer


def _generate_general_answer(query: str, docs: list[dict]) -> str:
    """Generate answer for general queries using retrieved context."""
    if not docs:
        return "I couldn't find relevant information for your question. Try asking about specific products, stores, forecasts, underperforming items, or KPIs."

    answer = f"Based on the retail knowledge base, here's what I found for your question:\n\n"

    for i, doc in enumerate(docs[:5], 1):
        score = doc.get("score", 0)
        answer += f"**{i}.** (relevance: {score:.1%}) {doc['content']}\n\n"

    answer += "---\n*Answer generated from RAG retrieval. Set GEMINI_API_KEY for more natural AI-generated responses.*"
    return answer


def generate_nlg_response(query: str, retrieved_docs: list[dict], extra_context: dict | None = None) -> str:
    """Route the query to the appropriate NLG generator.

    This is the Mode A (fallback) response generator.
    """
    q_lower = query.lower()

    # Route based on query intent
    if any(w in q_lower for w in ["restock", "replenish", "order", "stock", "inventory need"]):
        return _generate_restocking_answer(query, retrieved_docs)
    elif any(w in q_lower for w in ["underperform", "worst", "slow", "low sales", "poor"]):
        return _generate_underperforming_answer(query, retrieved_docs)
    elif any(w in q_lower for w in ["why", "explain", "reason", "cause", "driver", "increase", "decrease", "spike", "drop"]):
        return _generate_explanation_answer(query, retrieved_docs)
    elif any(w in q_lower for w in ["kpi", "metric", "rmse", "rmsse", "mae", "accuracy", "measure"]):
        return _generate_kpi_answer(query, retrieved_docs)
    elif any(w in q_lower for w in ["store", "compare", "best store", "which store"]):
        return _generate_store_comparison_answer(query, retrieved_docs)
    elif any(w in q_lower for w in ["safety stock", "reorder", "lead time"]):
        # Direct knowledge retrieval
        biz = [d for d in retrieved_docs if d["type"] == "business_knowledge"]
        if biz:
            return f"**{biz[0]['content']}**\n\n> Use the `/inventory` endpoint with specific product-store data to get actionable order quantities."
    elif any(w in q_lower for w in ["top", "best", "highest", "popular", "trending"]):
        return _generate_restocking_answer(query, [d for d in retrieved_docs if d["type"] in ("forecast", "catalog", "top_performer")])

    # Fallback
    return _generate_general_answer(query, retrieved_docs)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(
    query: str,
    retrieved_docs: list[dict],
    extra_context: dict | None = None,
) -> dict:
    """Generate a response using the best available LLM mode.

    Returns:
        {
            "answer": str,
            "mode": "gemini" | "nlg",
            "sources": list[dict],
            "query": str,
        }
    """
    sources = [
        {"content": d["content"][:200], "score": d["score"], "type": d["type"]}
        for d in retrieved_docs[:5]
    ]

    # Try Gemini first if API key is available
    if GEMINI_API_KEY:
        try:
            logger.info("Using Gemini LLM for response generation")
            prompt = _build_gemini_prompt(query, retrieved_docs, extra_context)
            answer = _call_gemini(prompt)
            return {
                "answer": answer,
                "mode": "gemini",
                "sources": sources,
                "query": query,
            }
        except Exception as e:
            logger.warning(f"Gemini failed, falling back to NLG: {e}")

    # Fallback to NLG
    logger.info("Using NLG engine for response generation")
    answer = generate_nlg_response(query, retrieved_docs, extra_context)

    return {
        "answer": answer,
        "mode": "nlg",
        "sources": sources,
        "query": query,
    }


if __name__ == "__main__":
    # Quick test
    from src.genai.knowledge_base import get_knowledge_base

    kb = get_knowledge_base()

    test_queries = [
        "Which products need restocking?",
        "Show me underperforming products",
        "Why would sales increase?",
    ]

    for q in test_queries:
        docs = kb.retrieve(q, top_k=5)
        result = generate_response(q, docs)
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"Mode: {result['mode']}")
        print(f"A: {result['answer'][:400]}...")
        print(f"Sources: {len(result['sources'])}")