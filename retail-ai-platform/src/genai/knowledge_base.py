"""
Module 6 — Retail Knowledge Base (Vector Store)
===================================================
Builds and manages the retail knowledge base using TF-IDF embeddings.

Knowledge Sources:
  1. Product catalog (item metadata, categories, stores)
  2. Forecast summaries (per product-store predictions)
  3. SHAP explanations (top drivers per product-store)
  4. Business rules (inventory formulas, KPI definitions)
  5. Historical sales summaries

Embedding: TF-IDF (swappable to sentence-transformers)
Retrieval: Cosine similarity (swappable to FAISS)
"""

import pandas as pd
import numpy as np
import json
import logging
import hashlib
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config import (
    PROCESSED_SALES, FEATURES_FILE, MODELS_DIR,
    DATA_PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_DIR = DATA_PROCESSED_DIR / "knowledge_base"
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)


class RetailKnowledgeBase:
    """Manages the retail knowledge base — documents + vector retrieval."""

    def __init__(self):
        self.documents: list[dict] = []
        self.tfidf_matrix = None
        self.vectorizer: TfidfVectorizer | None = None
        self.index_path = KNOWLEDGE_BASE_DIR / "kb_index.npz"
        self.docs_path = KNOWLEDGE_BASE_DIR / "kb_docs.json"
        self.vec_path = KNOWLEDGE_BASE_DIR / "kb_vectorizer.pkl"

    def _add_doc(self, doc_id: str, content: str, metadata: dict, doc_type: str):
        """Add a single document to the knowledge base."""
        self.documents.append({
            "id": doc_id,
            "content": content,
            "metadata": metadata,
            "type": doc_type,
        })

    def build(self):
        """Build the knowledge base from all data sources."""
        logger.info("=" * 60)
        logger.info("  BUILDING RETAIL KNOWLEDGE BASE")
        logger.info("=" * 60)

        # Source 1: Product Catalog
        self._build_product_catalog()
        # Source 2: Forecast Summaries
        self._build_forecast_summaries()
        # Source 3: Business Rules & KPI Definitions
        self._build_business_knowledge()
        # Source 4: Historical Sales Patterns
        self._build_historical_patterns()

        # Build TF-IDF index
        self._build_index()

        # Save
        self.save()

        logger.info(f"  Total documents: {len(self.documents)}")
        logger.info(f"  Index shape: {self.tfidf_matrix.shape}")
        logger.info(f"  Saved to: {KNOWLEDGE_BASE_DIR}")
        logger.info("=" * 60)

    def _build_product_catalog(self):
        """Index all product-store combinations with their metadata."""
        logger.info("  Indexing product catalog...")
        data = pd.read_parquet(PROCESSED_SALES)

        # Unique product-store combos with latest info
        catalog = data.groupby(["store_id", "item_id", "dept_id", "cat_id", "state_id"], observed=True).agg(
            avg_sales=("sales", "mean"),
            total_sales=("sales", "sum"),
            avg_price=("sell_price", "mean"),
            max_sales=("sales", "max"),
            zero_pct=("sales", lambda x: (x == 0).mean() * 100),
            days_observed=("sales", "count"),
        ).reset_index()

        catalog = catalog.copy()
        catalog["avg_price"] = catalog["avg_price"].fillna(0)
        for _, row in catalog.iterrows():
            content = (
                f"Product {row['item_id']} in store {row['store_id']} ({row['state_id']}). "
                f"Category: {row['cat_id']}, Department: {row['dept_id']}. "
                f"Average daily sales: {row['avg_sales']:.2f}, Total sales: {int(row['total_sales'])}. "
                f"Max daily sales: {int(row['max_sales'])}. Average price: ${row['avg_price']:.2f}. "
                f"Zero sales percentage: {row['zero_pct']:.1f}%. "
                f"Observed for {row['days_observed']} days."
            )
            self._add_doc(
                doc_id=f"catalog_{row['store_id']}_{row['item_id']}",
                content=content,
                metadata={"store_id": row["store_id"], "item_id": row["item_id"],
                          "cat_id": row["cat_id"], "type": "catalog"},
                doc_type="catalog",
            )

        logger.info(f"    Added {len(catalog)} product-store documents")

    def _build_forecast_summaries(self):
        """Index forecast summaries from trained models."""
        logger.info("  Indexing forecast summaries...")

        # Load best model predictions on recent data
        feat_data = pd.read_parquet(FEATURES_FILE)

        # Find best model
        best_model = None
        best_rmse = float("inf")
        for name in ["catboost", "lightgbm", "xgboost"]:
            meta_path = MODELS_DIR / f"{name}_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                if meta["metrics"]["rmse"] < best_rmse:
                    best_rmse = meta["metrics"]["rmse"]
                    best_model = name

        if best_model is None:
            logger.warning("    No trained models found, skipping forecast summaries")
            return

        import joblib
        model = joblib.load(MODELS_DIR / f"{best_model}.pkl")
        feature_cols = json.load(open(MODELS_DIR / f"{best_model}_meta.json"))["features"]

        # Get the last day per product-store and predict
        latest = feat_data.groupby(["store_id", "item_id"], observed=True).tail(1).copy()
        X = latest[feature_cols].fillna(0)
        latest["predicted_sales"] = np.clip(model.predict(X), 0, None)

        for _, row in latest.iterrows():
            content = (
                f"Forecast for {row['item_id']} at {row['store_id']}: "
                f"Predicted daily sales: {row['predicted_sales']:.2f}. "
                f"Model: {best_model} (RMSE={best_rmse}). "
                f"Category: {row['cat_id']}, State: {row['state_id']}."
            )
            self._add_doc(
                doc_id=f"forecast_{row['store_id']}_{row['item_id']}",
                content=content,
                metadata={"store_id": row["store_id"], "item_id": row["item_id"],
                          "prediction": float(row["predicted_sales"]), "model": best_model},
                doc_type="forecast",
            )

        logger.info(f"    Added {len(latest)} forecast documents")
        del model, latest, X
        import gc; gc.collect()

    def _build_business_knowledge(self):
        """Index business rules, KPI definitions, and domain knowledge."""
        logger.info("  Indexing business knowledge...")

        knowledge = [
            {
                "id": "biz_kpi_rmse",
                "content": (
                    "RMSE (Root Mean Square Error) measures forecast accuracy. "
                    "Lower is better. It penalizes large errors more than MAE. "
                    "RMSSE is the scaled version where RMSSE < 1 means the model "
                    "beats a naive baseline forecast."
                ),
                "metadata": {"type": "kpi_definition", "kpi": "RMSE"},
            },
            {
                "id": "biz_safety_stock",
                "content": (
                    "Safety stock = z × σ × √(lead_time). "
                    "z=1.645 for 95% service level, z=1.28 for 90%. "
                    "It buffers against demand variability. "
                    "Higher demand variability or longer lead times require more safety stock."
                ),
                "metadata": {"type": "inventory_rule", "concept": "safety_stock"},
            },
            {
                "id": "biz_reorder_point",
                "content": (
                    "Reorder Point = (daily demand × lead time) + safety stock. "
                    "When inventory falls below this point, a new order should be placed. "
                    "This prevents stockouts while minimizing holding costs."
                ),
                "metadata": {"type": "inventory_rule", "concept": "reorder_point"},
            },
            {
                "id": "biz_tweedie",
                "content": (
                    "Tweedie loss is used for sales forecasting because it handles "
                    "zero-inflated continuous data well. Many retail items have "
                    "frequent zero-sales days. Tweedie variance power of 1.1 is "
                    "a good balance between Poisson and Gamma distributions."
                ),
                "metadata": {"type": "ml_knowledge", "concept": "tweedie_loss"},
            },
            {
                "id": "biz_snap",
                "content": (
                    "SNAP (Supplemental Nutrition Assistance Program) is a US "
                    "government program. In M5 data, snap_CA, snap_TX, snap_WI "
                    "indicate if SNAP benefits are distributed that day in each state. "
                    "SNAP days typically see 15-30% higher sales for eligible food products."
                ),
                "metadata": {"type": "domain_knowledge", "concept": "SNAP"},
            },
            {
                "id": "biz_rmsse",
                "content": (
                    "RMSSE (Root Mean Squared Scaled Error) is the M5 competition metric. "
                    "It scales the RMSE by the naive forecast RMSE. "
                    "RMSSE < 1.0 means the model outperforms simply predicting last period's sales. "
                    "Values below 0.5 are considered excellent."
                ),
                "metadata": {"type": "kpi_definition", "kpi": "RMSSE"},
            },
            {
                "id": "biz_shap",
                "content": (
                    "SHAP (SHapley Additive exPlanations) explains ML predictions by "
                    "computing each feature's contribution. Positive SHAP values increase "
                    "the prediction, negative values decrease it. "
                    "Global importance = mean absolute SHAP across all predictions."
                ),
                "metadata": {"type": "ml_knowledge", "concept": "SHAP"},
            },
            {
                "id": "biz_lag_features",
                "content": (
                    "Lag features use past sales values as inputs. Lag-1 = yesterday's sales, "
                    "Lag-7 = same day last week, Lag-28 = same day last month. "
                    "Lag-7 captures weekly seasonality. Lag-28 captures monthly patterns. "
                    "Rolling statistics (mean, std over 7/14/28/56 days) capture trends and volatility."
                ),
                "metadata": {"type": "feature_knowledge", "concept": "lag_features"},
            },
        ]

        for item in knowledge:
            self._add_doc(
                doc_id=item["id"],
                content=item["content"],
                metadata=item["metadata"],
                doc_type="business_knowledge",
            )

        logger.info(f"    Added {len(knowledge)} business knowledge documents")

    def _build_historical_patterns(self):
        """Index aggregated historical patterns for quick retrieval."""
        logger.info("  Indexing historical patterns...")
        data = pd.read_parquet(PROCESSED_SALES)

        # Store-level patterns
        store_patterns = data.groupby("store_id", observed=True).agg(
            total_sales=("sales", "sum"),
            avg_daily=("sales", "mean"),
            best_day=("sales", "max"),
        ).reset_index()

        for _, row in store_patterns.iterrows():
            content = (
                f"Store {row['store_id']}: Total sales {int(row['total_sales'])}, "
                f"Average daily {row['avg_daily']:.2f}, "
                f"Best single day {int(row['best_day'])}."
            )
            self._add_doc(
                doc_id=f"store_pattern_{row['store_id']}",
                content=content,
                metadata={"store_id": row["store_id"], "type": "store_pattern"},
                doc_type="store_pattern",
            )

        # Category-level patterns
        cat_patterns = data.groupby("cat_id", observed=True).agg(
            total_sales=("sales", "sum"),
            avg_daily=("sales", "mean"),
            n_products=("item_id", "nunique"),
            avg_price=("sell_price", "mean"),
        ).reset_index()

        for _, row in cat_patterns.iterrows():
            content = (
                f"Category {row['cat_id']}: {int(row['n_products'])} products, "
                f"Total sales {int(row['total_sales'])}, "
                f"Average daily {row['avg_daily']:.2f}, "
                f"Average price ${row['avg_price']:.2f}."
            )
            self._add_doc(
                doc_id=f"cat_pattern_{row['cat_id']}",
                content=content,
                metadata={"cat_id": row["cat_id"], "type": "category_pattern"},
                doc_type="category_pattern",
            )

        # Underperforming products (bottom 10% by sales)
        item_sales = data.groupby(["store_id", "item_id"], observed=True)["sales"].mean()
        threshold = item_sales.quantile(0.10)
        underperformers = item_sales[item_sales <= threshold].reset_index()

        for _, row in underperformers.iterrows():
            content = (
                f"Underperforming product: {row['item_id']} at {row['store_id']}. "
                f"Average daily sales: {row['sales']:.2f} (bottom 10%). "
                f"This product may need promotion, repricing, or delisting consideration."
            )
            self._add_doc(
                doc_id=f"underperformer_{row['store_id']}_{row['item_id']}",
                content=content,
                metadata={"store_id": row["store_id"], "item_id": row["item_id"],
                          "avg_sales": float(row["sales"]), "type": "underperformer"},
                doc_type="underperformer",
            )

        # Top performers (top 10%)
        top_threshold = item_sales.quantile(0.90)
        top_performers = item_sales[item_sales >= top_threshold].reset_index()

        for _, row in top_performers.iterrows():
            content = (
                f"Top performing product: {row['item_id']} at {row['store_id']}. "
                f"Average daily sales: {row['sales']:.2f} (top 10%). "
                f"This is a high-velocity item that should be kept in stock."
            )
            self._add_doc(
                doc_id=f"top_performer_{row['store_id']}_{row['item_id']}",
                content=content,
                metadata={"store_id": row["store_id"], "item_id": row["item_id"],
                          "avg_sales": float(row["sales"]), "type": "top_performer"},
                doc_type="top_performer",
            )

        logger.info(f"    Added store patterns, category patterns, underperformers, and top performers")
        del data, store_patterns, cat_patterns, item_sales, underperformers, top_performers
        import gc; gc.collect()

    def _build_index(self):
        """Build TF-IDF vector index over all documents."""
        logger.info("  Building TF-IDF index...")
        texts = [doc["content"] for doc in self.documents]

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        logger.info(f"    Vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def retrieve(self, query: str, top_k: int = 5, doc_type: str | None = None) -> list[dict]:
        """Retrieve most relevant documents for a query.

        Args:
            query: Natural language question
            top_k: Number of documents to return
            doc_type: Optional filter by document type

        Returns:
            List of {doc, score, metadata} dicts
        """
        if self.vectorizer is None:
            self.load()

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1]

        results = []
        for idx in top_indices:
            doc = self.documents[idx]

            # Filter by type if specified
            if doc_type and doc["type"] != doc_type:
                continue

            if scores[idx] > 0.01:  # Minimum relevance threshold
                results.append({
                    "content": doc["content"],
                    "score": round(float(scores[idx]), 4),
                    "metadata": doc["metadata"],
                    "type": doc["type"],
                    "id": doc["id"],
                })

            if len(results) >= top_k:
                break

        return results

    def save(self):
        """Save the knowledge base to disk."""
        # Save TF-IDF matrix
        import scipy.sparse as sp
        sp.save_npz(self.index_path, self.tfidf_matrix)

        # Save documents
        with open(self.docs_path, "w") as f:
            json.dump(self.documents, f)

        # Save vectorizer
        import joblib
        joblib.dump(self.vectorizer, self.vec_path)

    def load(self):
        """Load the knowledge base from disk."""
        import scipy.sparse as sp
        import joblib

        self.tfidf_matrix = sp.load_npz(self.index_path)

        with open(self.docs_path) as f:
            self.documents = json.load(f)

        self.vectorizer = joblib.load(self.vec_path)

        logger.info(f"  Loaded knowledge base: {len(self.documents)} docs, matrix {self.tfidf_matrix.shape}")

    def is_built(self) -> bool:
        """Check if the knowledge base has been built and saved."""
        return (
            self.index_path.exists()
            and self.docs_path.exists()
            and self.vec_path.exists()
        )


# Singleton
_kb_instance: RetailKnowledgeBase | None = None


def get_knowledge_base() -> RetailKnowledgeBase:
    """Get or build the knowledge base singleton."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = RetailKnowledgeBase()
        if _kb_instance.is_built():
            _kb_instance.load()
        else:
            _kb_instance.build()
    return _kb_instance


if __name__ == "__main__":
    kb = get_knowledge_base()

    # Test retrieval
    queries = [
        "Which products need restocking?",
        "Why would demand increase?",
        "What is safety stock?",
        "Show me underperforming stores",
        "Which store has highest sales?",
    ]

    for q in queries:
        print(f"\nQ: {q}")
        results = kb.retrieve(q, top_k=2)
        for r in results:
            print(f"  [{r['score']:.3f}] {r['content'][:100]}...")