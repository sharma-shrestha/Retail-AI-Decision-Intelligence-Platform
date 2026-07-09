# Retail AI Decision Intelligence Platform

## Overview
A next-generation retail intelligence system combining Machine Learning, Explainable AI, RAG-based GenAI, and Business Analytics to forecast demand, explain predictions, answer business questions, and recommend inventory actions.

## Architecture (Phase 1 + Phase 2 — Implemented)

```
M5 DATASET (Kaggle)
    │
    ▼
Module 1: Data Engineering (ETL)
  - Schema validation, missing values, dedup
  - Stratified sampling (99 products × 10 stores)
  - Truncated to 500 most recent days
  - Output: processed_sales.parquet (495K rows, 0.6MB)
    │
    ▼
Module 2: Feature Engineering (80 features)
  - Time: cyclical encoding, events, SNAP, holidays
  - Sales: lags (1,7,14,28,56), rolling stats (mean/std/min/max/median)
  - Price: change, discount, momentum, rolling price
  - Business: store/category popularity, zero streak, days since sale
  - Interactions: price×event, weekend×lag, snap×price
    │
    ▼
Module 3: Forecasting Engine
  - LightGBM  (RMSE=0.0398, RMSSE=0.0127)
  - XGBoost   (RMSE=0.1278, RMSSE=0.0408)
  - CatBoost  (RMSE=0.0337, RMSSE=0.0107) ← BEST
  - Early stopping, Tweedie loss
    │
    ▼
Module 4: Explainable AI (SHAP)
  - TreeExplainer for per-prediction explanations
  - Global feature importance ranking
    │
    ▼
Module 5: Inventory Intelligence
  - Safety stock = z × σ × √lead_time
  - Reorder point = (daily demand × lead time) + safety stock
  - Stock status: adequate / low / critical
  - Natural language reasoning
    │
    ▼
Module 6: GenAI Layer — RAG AI Copilot ⭐
  - Knowledge Base: 2,199 documents (TF-IDF vector index)
    - 990 product catalog entries
    - 990 forecast summaries
    - 8 business knowledge entries (KPIs, formulas, domain)
    - Store/category patterns, underperformers, top performers
  - RAG Pipeline: retrieve → enrich → generate
  - Modular LLM Interface:
    - Mode A (default): NLG engine — works offline, zero API cost
    - Mode B: Gemini — set GEMINI_API_KEY to activate
    │
    ▼
Module 8: FastAPI (10 endpoints)
  Phase 1:
  - GET  /health
  - GET  /api/v1/models
  - POST /api/v1/forecast
  - POST /api/v1/explain
  - POST /api/v1/inventory
  - GET  /api/v1/analytics/summary
  - GET  /api/v1/analytics/top-products
  - GET  /api/v1/analytics/feature-importance
  Phase 2 (GenAI):
  - POST /api/v1/ask-ai              — Natural language Q&A
  - GET  /api/v1/ask-ai/report       — Auto daily report
  - GET  /api/v1/ask-ai/underperforming — Risk products
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline (ETL → Features → Train → Build KB)
```bash
python scripts/run_pipeline.py
```

### 3. Start the API server
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 4. Explore the API
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Examples

### Forecast demand
```bash
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Content-Type: application/json" \
  -d '{"store_id": "CA_1", "item_id": "FOODS_3_288", "model": "catboost"}'
```

### Explain a prediction with SHAP
```bash
curl -X POST http://localhost:8000/api/v1/explain \
  -H "Content-Type: application/json" \
  -d '{"store_id": "CA_1", "item_id": "FOODS_3_288", "model": "catboost"}'
```

### Inventory recommendation
```bash
curl -X POST http://localhost:8000/api/v1/inventory \
  -H "Content-Type: application/json" \
  -d '{"store_id": "CA_1", "item_id": "FOODS_3_288", "model": "catboost", "current_stock": 50}'
```

### 🆕 Ask AI Copilot (RAG)
```bash
curl -X POST http://localhost:8000/api/v1/ask-ai \
  -H "Content-Type: application/json" \
  -d '{"question": "Which products need restocking?"}'
```

### 🆕 Daily Report
```bash
curl http://localhost:8000/api/v1/ask-ai/report
```

### 🆕 Underperforming Products
```bash
curl http://localhost:8000/api/v1/ask-ai/underperforming?n=10
```

## Gemini Integration (Optional)

The platform works out-of-the-box with the built-in NLG engine. To upgrade to Gemini:

```bash
# 1. Copy the env template
cp .env.example .env

# 2. Add your free Gemini API key
# Get one at: https://aistudio.google.com/apikey
echo "GEMINI_API_KEY=your_key_here" >> .env

# 3. Start the server
source .env
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The system auto-detects `GEMINI_API_KEY` and switches to Gemini for response generation. No code changes needed.

## Tech Stack
- **Data**: Pandas, NumPy, PyArrow
- **ML**: LightGBM, XGBoost, CatBoost
- **Explainability**: SHAP
- **RAG**: scikit-learn TF-IDF, LangChain-style pipeline
- **GenAI**: google-generativeai (Gemini, optional)
- **API**: FastAPI, Uvicorn, Pydantic
- **Deployment**: Docker

## Project Structure
```
retail-ai-platform/
├── data/
│   ├── raw/m5-data/                  # Original M5 CSVs
│   └── processed/
│       ├── processed_sales.parquet   # ETL output
│       ├── engineered_features.parquet # Feature matrix
│       └── knowledge_base/           # RAG vector index
├── src/
│   ├── data_engineering/etl.py       # Module 1: ETL
│   ├── feature_engineering/features.py # Module 2: 80 features
│   ├── models/train.py               # Module 3: Training
│   ├── explainability/shap_explainer.py # Module 4: SHAP
│   ├── inventory/engine.py           # Module 5: Inventory
│   ├── genai/
│   │   ├── knowledge_base.py         # Module 6: Vector store (2,199 docs)
│   │   ├── llm_interface.py          # Module 6: NLG + Gemini
│   │   └── rag_pipeline.py           # Module 6: RAG orchestrator
│   ├── api/main.py                   # Module 8: FastAPI (10 endpoints)
│   └── utils/config.py               # Central config
├── models/saved/                     # Trained .pkl models
├── scripts/run_pipeline.py           # End-to-end runner
├── .env.example                      # Environment template
├── Dockerfile
└── requirements.txt
```

## Phase Roadmap
- [x] **Phase 1**: Data Engineering + Features + ML Models + SHAP + API (7 endpoints)
- [x] **Phase 2**: RAG AI Copilot + Knowledge Base + NLG Engine + Gemini Integration (3 endpoints)
- [ ] **Phase 3**: Dashboard (Streamlit/React) + Docker Deployment