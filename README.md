<div align="center">
# AI-Powered Retail Decision Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-Best_Model-FF6F00?logo=catboost&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-1E88E5)
![License](https://img.shields.io/badge/License-MIT-green)
![M5 Dataset](https://img.shields.io/badge/Dataset-M5_Kaggle-9C27B0)

**End-to-end demand forecasting, explainability, and intelligent inventory recommendations — powered by 3 gradient boosting models, SHAP, and a RAG AI Copilot.**

[Features](#-key-features) · [Architecture](#-system-architecture) · [Demo](#-api-examples) · [Quick Start](#-quick-start)

</div>

---

<img src="docs/images/dashboard_screenshot.png" alt="Dashboard Screenshot" width="100%">

<br>

## 📋 Overview

Retailers lose **$1.1 trillion annually** to overstocking and stockouts. Traditional forecasting relies on simple moving averages and gut feeling. This platform solves that with a **full-stack ML pipeline** that transforms raw retail data into actionable business intelligence.

**What it does:**
- Ingests M5 Kaggle retail data (99 products × 10 stores × 500 days, ~411K rows)
- Engineers **80 predictive features** across 5 categories (temporal, sales, price, business, interaction)
- Trains **3 gradient boosting models** with Tweedie loss — CatBoost wins with **RMSE = 0.0337**
- Explains every prediction with **SHAP TreeExplainer**
- Generates inventory actions via **safety stock + reorder point formulas**
- Answers natural language business questions via a **RAG AI Copilot** with 2,199-doc knowledge base
- Serves everything through a **FastAPI backend (10 endpoints)** and **Next.js 16 interactive dashboard (6 tabs)**

---

## ✨ Key Features

<table>
<tr>
<td width="33%">

### 📊 Demand Forecasting
3 production-grade models (CatBoost, LightGBM, XGBoost) with early stopping, Tweedie loss, and time-based validation.

</td>
<td width="33%">

### 🔍 SHAP Explainability
Per-prediction waterfall plots + global feature importance rankings via TreeExplainer.

</td>
<td width="33%">

### 🤖 RAG AI Copilot
Ask "Which products need restocking?" — retrieves from 2,199 vector-indexed docs and generates an answer.

</td>
</tr>
<tr>
<td width="33%">

### 📦 Inventory Intelligence
Safety stock = z × σ × √lead_time. Reorder point, stock status, and NL recommendations.

</td>
<td width="33%">

### 🧠 80 Engineered Features
Cyclical time encoding, multi-window rolling stats, price momentum, SNAP/event flags, cross-feature interactions.

</td>
<td width="33%">

### 🖥️ Interactive Dashboard
6-tab React dashboard: Overview, Forecast, Analytics, Explainability, Inventory, AI Copilot — all real-time.

</td>
</tr>
</table>

---

## 🏗️ System Architecture

<img src="docs/images/architecture.png" alt="System Architecture" width="100%">

The platform follows a **modular pipeline architecture** with 9 independent modules connected through shared data artifacts (Parquet files, pickle models, vector indices):

```
M5 Kaggle Dataset
       │
       ▼
┌──────────────────┐
│  Module 1: ETL   │  Schema validation · Dedup · Stratified sampling · 500-day truncation
└──────┬───────────┘
       │  processed_sales.parquet (411K rows)
       ▼
┌──────────────────────┐
│  Module 2: Features  │  80 features · 5 groups · Lags · Rolling stats · Price · Interactions
└──────┬───────────────┘
       │  engineered_features.parquet (411K × 80 cols)
       ▼
┌──────────────────────┐
│  Module 3: Training  │  LightGBM · XGBoost · CatBoost · Tweedie loss · Early stopping
└──────┬───────────────┘
       │  saved .pkl models + _meta.json
       ├──────────────────────────────────┐
       ▼                                  ▼
┌──────────────────┐            ┌──────────────────┐
│  Module 4: SHAP  │            │  Module 5: Stock │
│  TreeExplainer   │            │  Safety stock    │
└──────────────────┘            │  Reorder point   │
                                └──────────────────┘
       │                                  │
       ▼                                  ▼
┌──────────────────┐            ┌──────────────────┐
│  Module 6: RAG   │            │  Module 8: API   │
│  KB (2,199 docs) │◄──────────►│  FastAPI 10 eps  │
│  TF-IDF + Cosine │            └──────┬───────────┘
└──────────────────┘                   │
                                       ▼
                                ┌──────────────────┐
                                │  Module 9: UI    │
                                │  Next.js 16      │
                                │  React 19        │
                                │  6 Dashboard tabs│
                                └──────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why This Choice |
|-------|-----------|-----------------|
| **Language (Backend)** | Python 3.12 | Dominant ML/DS ecosystem, best library support for gradient boosting & SHAP |
| **Language (Frontend)** | JavaScript (JSX) | Faster iteration for dashboard UI, no TypeScript compilation overhead in dev |
| **ML Framework** | LightGBM + XGBoost + CatBoost | Top 3 gradient boosting libraries; CatBoost handles categorical features natively, LightGBM is fastest, XGBoost is most configurable |
| **Loss Function** | Tweedie (variance_power=1.1) | Ideal for right-skewed, zero-inflated retail sales data (between Poisson & Gamma) |
| **Explainability** | SHAP TreeExplainer | Exact Shapley values for tree models, industry standard for model interpretability |
| **RAG Retrieval** | scikit-learn TF-IDF + Cosine Similarity | Zero-cost, no GPU needed, works offline, sufficient for structured retail domain docs |
| **GenAI (Optional)** | Google Gemini via `generativeai` SDK | Free tier available, swappable LLM interface — NLG fallback works with zero API cost |
| **API Framework** | FastAPI + Uvicorn | Auto-generated Swagger docs, Pydantic validation, async-capable, 2-3x faster than Flask |
| **ASGI Server** | Uvicorn | Production-grade ASGI server, lightweight, handles concurrent requests efficiently |
| **Frontend Framework** | Next.js 16 + React 19 | App Router, API routes for proxying, SSR capability, largest React ecosystem |
| **UI Library** | shadcn/ui + Radix Primitives | Accessible, unstyled components, full customization via Tailwind CSS |
| **Charts** | Recharts | React-native charting, composable API, good for time-series and bar charts |
| **Data Fetching** | TanStack React Query | Automatic caching, refetching, loading/error states — ideal for polling dashboards |
| **Styling** | Tailwind CSS 4 | Utility-first, consistent design system, small bundle size |
| **Data Format** | Parquet (PyArrow) | Columnar storage, 5-10x smaller than CSV, faster read/write for tabular data |
| **Containerization** | Docker | Reproducible builds, easy deployment, isolates Python + Node environments |
| **API Gateway** | Caddy | Automatic HTTPS, simple reverse proxy config via `XTransformPort` |

---

## 📊 Data Pipeline

<img src="docs/images/data_pipeline.png" alt="Data Pipeline" width="100%">

### Module 1: ETL (Extract → Transform → Load)

Raw M5 Kaggle data goes through a rigorous cleaning pipeline:

| Step | Description |
|------|-------------|
| **Schema Validation** | Verify column types, required fields (`item_id`, `store_id`, `d`, `sales`) |
| **Missing Value Handling** | Calendar events filled with "NoEvent", prices forward-filled |
| **Deduplication** | Remove exact duplicate rows (same item + store + day) |
| **Stratified Sampling** | 99 items selected across 3 categories (FOODS, HOBBIES, HOUSEHOLDS), ensuring category balance |
| **Store Coverage** | All 10 stores (CA_1-3, TX_1-3, WI_1-3) included for every sampled item |
| **Time Truncation** | Only the most recent 500 days used (memory constraint: 4GB RAM target) |
| **Wide-to-Long Melt** | M5's wide format (d_1, d_2, ... d_1913) melted to long format with `date` column |
| **Calendar Join** | Merge date features: weekday, month, year, events, SNAP flags |
| **Price Join** | Merge `sell_prices` for each item-store-date combination |
| **Output** | `processed_sales.parquet` — 411K rows × 15 columns, ~0.6 MB |

---

## 🧠 Feature Engineering

<img src="docs/images/feature_groups.png" alt="Feature Groups" width="100%">

80 engineered features across **5 groups**, designed to capture every demand signal:

| Group | Count | Features | Rationale |
|-------|-------|----------|-----------|
| **Temporal** | ~15 | `day_of_week_sin/cos`, `month_sin/cos`, `day_num`, `year`, `is_weekend`, `is_month_start/end`, `quarter`, `week_of_year` | Cyclical encoding preserves periodic patterns (weekly seasonality, monthly trends) |
| **Sales Lags** | 5 | `lag_1`, `lag_7`, `lag_14`, `lag_28`, `lag_56` | Past sales at key intervals — yesterday, last week, 2 weeks, 4 weeks, 8 weeks |
| **Rolling Statistics** | 40 | For windows [7, 14, 28, 56] × stats [mean, std, min, max, median] | Capture trend, volatility, baseline demand, and seasonal patterns |
| **Price** | ~10 | `price`, `price_change`, `price_momentum_7/14/28`, `rolling_price_mean_7/14/28`, `discount`, `is_promo` | Price elasticity signals — demand often spikes on discounts and drops on hikes |
| **Business / Encoded** | ~7 | `store_id_enc`, `item_id_enc`, `dept_id_enc`, `cat_id_enc`, `state_id_enc`, `snap_CA/TX/WI`, `event_type_1/2`, `is_event_day` | Categorical embeddings + government assistance program (SNAP) flags |
| **Interactions** | ~8 | `price × event`, `weekend × lag_7`, `snap × price`, `lag_7 × rolling_mean_28`, etc. | Cross-feature signals that capture compound effects (e.g., weekend + recent demand) |

### Target Variable
- **`sales`** — daily unit sales per item per store (non-negative, right-skewed, zero-inflated)
- This distribution is why **Tweedie loss** (variance_power=1.1) was chosen over MSE

---

## 🎯 Model Performance

<img src="docs/images/model_comparison.png" alt="Model Comparison" width="100%">

### Metrics Summary

| Model | RMSE ↓ | MAE ↓ | RMSSE ↓ | Training Time | Status |
|-------|--------|-------|---------|---------------|--------|
| **CatBoost** | **0.0337** | **0.0187** | **0.0107** | ~45s | 🏆 Best |
| **LightGBM** | 0.0398 | 0.0215 | 0.0127 | ~12s | ⚡ Fastest |
| XGBoost | 0.1278 | 0.0738 | 0.0408 | ~18s | Baseline |

### Why CatBoost Wins
- **Ordered Boosting** prevents target leakage — critical for time-series
- **Native categorical feature handling** — no manual encoding needed
- **Symmetric trees** — more robust generalization on noisy retail data

### Training Configuration
```python
# All 3 models use identical data split
VALIDATION_DAYS = 28          # Last 28 days for validation (time-based, no leakage)
HORIZON = 28                  # Forecast 28 days ahead
LOSS = "Tweedie"              # variance_power=1.1 (between Poisson & Gamma)
EARLY_STOPPING = 50 rounds    # Prevent overfitting
N_ESTIMATORS = 500            # Max boosting rounds
```

### Validation Strategy
Time-based train/val split (NOT random):
- **Train**: Days 1–472 (90.4% of data)
- **Validation**: Days 473–500 (last 28 days, 9.6%)

This mimics real-world deployment where the model predicts the future using only past data.

---

## 🔍 Explainability (SHAP)

<img src="docs/images/shap_waterfall.png" alt="SHAP Waterfall" width="48%"> <img src="docs/images/feature_importance.png" alt="Feature Importance" width="48%">

### How It Works
Every prediction comes with a **SHAP explanation** showing which features pushed the forecast up or down:

- **SHAP TreeExplainer** computes exact Shapley values in O(TLD) time for tree ensembles
- **Waterfall plot**: Shows top 10 features for a single prediction — red = increased demand, blue = decreased
- **Global importance**: Mean absolute SHAP values across all predictions reveal the most influential features

### Top 5 Global Feature Drivers
1. `lag_7` — Sales from exactly one week ago (strong weekly seasonality)
2. `rolling_mean_28` — 28-day average demand (baseline trend)
3. `price` — Current selling price (demand elasticity)
4. `rolling_std_14` — 14-day demand volatility (uncertainty signal)
5. `day_of_week_sin` — Cyclical day-of-week encoding (weekend vs weekday patterns)

---

## 🤖 RAG AI Copilot

<img src="docs/images/rag_pipeline.png" alt="RAG Pipeline" width="100%">

### Architecture

The AI Copilot uses a **Retrieve-Augment-Generate** pipeline to answer natural language questions about your retail data:

```
User Question
     │
     ▼
┌─────────────────┐
│  TF-IDF Vectorizer │  Convert question to sparse vector
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Cosine Similarity │  Find top-k (k=5) relevant documents from KB
└───────┬─────────┘
        │
        ▼
┌─────────────────┐
│  Context Builder  │  Merge retrieved docs into structured prompt
└───────┬─────────┘
        │
        ▼
┌─────────────────────────────────────┐
│  LLM Interface                       │
│  ├─ Mode A (default): NLG Engine     │  ← Works offline, zero cost
│  └─ Mode B (optional): Google Gemini  │  ← Set GEMINI_API_KEY
└───────┬───────────────────────────┘
        │
        ▼
   Natural Language Answer
```

### Knowledge Base: 2,199 Documents

| Source | Count | Content |
|--------|-------|---------|
| Product Catalog | 990 | Item metadata, category, department, store mapping |
| Forecast Summaries | 990 | Per product-store: predicted demand, trend direction, confidence |
| Business Knowledge | 8 | KPI definitions, inventory formulas, domain terminology |
| Sales Patterns | 100+ | Store/category performance, underperformers, top movers |
| SHAP Explanations | 100+ | Top feature drivers per product-store combination |

### Example Interaction

**User:** *"Which products in CA_1 need urgent restocking?"*

**AI Copilot:**
> Based on current forecasts for store CA_1:
> 1. **FOODS_3_288** — Predicted 28-day demand: 145 units. If stock < 100, reorder immediately.
> 2. **FOODS_1_012** — Demand trending +23% week-over-week. Safety stock at 38 units.
> 3. **HOUSEHOLDS_2_047** — Zero sales streak: 5 days. Consider markdown strategy.
>
> *Retrieved from 5 knowledge base documents in 12ms.*

---

## 📦 Inventory Intelligence

<img src="docs/images/inventory_flow.png" alt="Inventory Flow" width="100%">

### Formulas Used

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Safety Stock** | `z × σ × √lead_time` | Buffer against demand variability (z=1.645 for 95% SL) |
| **Reorder Point** | `(daily_demand × lead_time) + safety_stock` | Trigger point to place a new order |
| **Days of Supply** | `current_stock / daily_demand` | How many days until stockout |
| **Stock Status** | Based on days of supply thresholds | `adequate` (>21d) / `low` (14-21d) / `medium` (7-14d) / `high` (3-7d) / `critical` (<3d) / `urgent` (0) |

### Example API Response
```json
{
  "store_id": "CA_1",
  "item_id": "FOODS_3_288",
  "forecasted_demand": 145.2,
  "daily_demand": 5.19,
  "demand_std": 3.82,
  "safety_stock": 44.30,
  "reorder_point": 80.63,
  "days_of_supply": 9.6,
  "stock_status": "medium",
  "recommendation": "Consider placing a reorder soon. Current stock covers ~10 days; reorder point is 81 units."
}
```

---

## 🖥️ Dashboard

<img src="docs/images/dashboard_screenshot.png" alt="Dashboard Screenshot" width="100%">

Built with **Next.js 16 + React 19 + shadcn/ui + Tailwind CSS 4 + Recharts + TanStack Query**.

### 6 Interactive Tabs

| Tab | Features |
|-----|----------|
| **Overview** | KPI cards (total products, active models, avg RMSE), model comparison chart, store performance heatmap |
| **Forecast** | Select store + item + model → 28-day demand forecast line chart with confidence bands |
| **Analytics** | Category distribution pie chart, top products table, on-demand SHAP feature importance bar chart |
| **Explainability** | SHAP waterfall chart for any prediction, top feature drivers with +/- contribution |
| **Inventory** | Store-item inventory status table with urgency badges, safety stock & reorder point display |
| **AI Copilot** | Chat interface with auto-scroll, asks natural language questions, displays retrieved context + AI answer |

### API Proxy Architecture
The Next.js frontend includes a **catch-all API proxy** (`src/app/api/[...slug]/route.ts`) that forwards all `/api/*` requests to the FastAPI backend on port 8000 — enabling seamless full-stack deployment without CORS issues.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check — returns status + loaded models |
| `GET` | `/api/v1/models` | List all trained models with RMSE, MAE, RMSSE metrics |
| `POST` | `/api/v1/forecast` | Generate 28-day demand forecast for a product-store |
| `POST` | `/api/v1/explain` | SHAP-based explanation for a specific prediction |
| `POST` | `/api/v1/inventory` | Inventory recommendation (safety stock, reorder point, status) |
| `GET` | `/api/v1/analytics/summary` | Business analytics summary (total sales, top categories, trends) |
| `GET` | `/api/v1/analytics/top-products` | Top/bottom performing products by forecasted demand |
| `GET` | `/api/v1/analytics/feature-importance` | Global SHAP feature importance ranking (on-demand) |
| `POST` | `/api/v1/ask-ai` | Ask the RAG AI Copilot a natural language question |
| `GET` | `/api/v1/ask-ai/report` | Auto-generated daily business intelligence report |
| `GET` | `/api/v1/ask-ai/underperforming` | List underperforming products at risk of stockout |

---

## 💻 API Examples

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

### Get inventory recommendation
```bash
curl -X POST http://localhost:8000/api/v1/inventory \
  -H "Content-Type: application/json" \
  -d '{"store_id": "CA_1", "item_id": "FOODS_3_288", "model": "catboost", "current_stock": 50}'
```

### Ask the AI Copilot
```bash
curl -X POST http://localhost:8000/api/v1/ask-ai \
  -H "Content-Type: application/json" \
  -d '{"question": "Which products need restocking in TX_2?"}'
```

### Auto daily report
```bash
curl http://localhost:8000/api/v1/ask-ai/report
```

### Interactive API docs
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.12+** with pip
- **Node.js 18+** and npm/bun
- **4GB RAM** minimum (platform is optimized for constrained environments)

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/retail-ai-platform.git
cd retail-ai-platform
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Download M5 data — place CSVs in data/raw/m5-data/
# Files needed: sales_train_validation.csv, calendar.csv, sell_prices.csv

# Run full pipeline: ETL → Feature Engineering → Train Models → Build Knowledge Base
python scripts/run_pipeline.py

# Start FastAPI server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 3. Frontend Setup (separate terminal)

```bash
cd frontend

# Install dependencies
npm install   # or: bun install

# Build for production (recommended — next dev OOMs on 4GB)
npm run build

# Start production server
npm run start
```

### 4. Open the Dashboard

Navigate to **[http://localhost:3000](http://localhost:3000)** — the dashboard will proxy all API calls to the FastAPI backend on port 8000.

### 5. (Optional) Gemini Integration

```bash
# In the backend directory:
cp .env.example .env

# Add your free Gemini API key
# Get one at: https://aistudio.google.com/apikey
echo "GEMINI_API_KEY=your_key_here" >> .env

# Restart the server — auto-detects and switches to Gemini
source .env
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

The system works **fully offline** with the built-in NLG engine by default. Gemini is an optional upgrade — no code changes needed.

---

## 📁 Project Structure

```
retail-ai-platform/
│
├── 📂 docs/
│   └── 📂 images/                          # README diagrams & screenshots
│       ├── architecture.png
│       ├── dashboard_screenshot.png
│       ├── model_comparison.png
│       ├── feature_importance.png
│       ├── shap_waterfall.png
│       ├── data_pipeline.png
│       ├── rag_pipeline.png
│       ├── inventory_flow.png
│       ├── store_performance.png
│       ├── feature_groups.png
│       └── category_distribution.png
│
├── 📂 backend/                              # Python FastAPI Backend
│   ├── 📂 data/
│   │   ├── 📂 raw/m5-data/                 # Kaggle M5 CSVs (gitignored)
│   │   └── 📂 processed/
│   │       ├── processed_sales.parquet      # ETL output
│   │       ├── engineered_features.parquet  # Feature matrix (80 cols)
│   │       └── 📂 knowledge_base/
│   │           ├── kb_docs.json             # 2,199 documents
│   │           ├── kb_vectorizer.pkl        # TF-IDF vectorizer
│   │           └── kb_index.npz             # Cosine similarity index
│   ├── 📂 src/
│   │   ├── 📂 data_engineering/
│   │   │   └── etl.py                      # Module 1: ETL pipeline
│   │   ├── 📂 feature_engineering/
│   │   │   └── features.py                 # Module 2: 80 features
│   │   ├── 📂 models/
│   │   │   └── train.py                    # Module 3: Model training
│   │   ├── 📂 explainability/
│   │   │   └── shap_explainer.py           # Module 4: SHAP
│   │   ├── 📂 inventory/
│   │   │   └── engine.py                   # Module 5: Inventory formulas
│   │   ├── 📂 genai/
│   │   │   ├── knowledge_base.py           # Module 6: Vector store
│   │   │   ├── llm_interface.py            # Module 6: NLG + Gemini
│   │   │   └── rag_pipeline.py             # Module 6: RAG orchestrator
│   │   ├── 📂 api/
│   │   │   └── main.py                     # Module 8: FastAPI (10 endpoints)
│   │   └── 📂 utils/
│   │       └── config.py                   # Central configuration
│   ├── 📂 models/saved/                    # Trained .pkl models (gitignored)
│   │   ├── catboost.pkl + catboost_meta.json
│   │   ├── lightgbm.pkl + lightgbm_meta.json
│   │   └── xgboost.pkl + xgboost_meta.json
│   ├── 📂 scripts/
│   │   ├── run_pipeline.py                 # End-to-end runner
│   │   ├── catboost_notebook.py            # CatBoost experiment
│   │   ├── lightgbm_xgboost_notebook.py    # LightGBM/XGBoost experiment
│   │   └── inspect_models.py               # Model inspection utility
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
│
├── 📂 frontend/                             # Next.js 16 Dashboard
│   ├── 📂 src/
│   │   ├── 📂 app/
│   │   │   ├── page.tsx                    # Main dashboard page
│   │   │   ├── layout.tsx                  # Root layout
│   │   │   ├── globals.css                 # Global styles
│   │   │   └── 📂 api/[...slug]/
│   │   │       └── route.ts                # Catch-all API proxy to backend
│   │   ├── 📂 components/
│   │   │   ├── 📂 dashboard/
│   │   │   │   ├── overview-tab.jsx        # KPIs + model comparison
│   │   │   │   ├── forecast-tab.jsx        # 28-day forecast chart
│   │   │   │   ├── analytics-tab.jsx       # Category pie + top products
│   │   │   │   ├── explain-tab.jsx         # SHAP waterfall
│   │   │   │   ├── inventory-tab.jsx       # Stock status table
│   │   │   │   ├── copilot-tab.jsx         # AI chat interface
│   │   │   │   ├── query-provider.jsx      # React Query provider
│   │   │   │   └── data-selectors.jsx      # Store/item/model dropdowns
│   │   │   └── 📂 ui/                      # shadcn/ui components
│   │   ├── 📂 hooks/                       # Custom React hooks
│   │   └── 📂 lib/                         # Utilities, API client, constants
│   ├── package.json
│   ├── next.config.ts
│   ├── tailwind.config.ts
│   ├── postcss.config.mjs
│   └── tsconfig.json
│
├── docker-compose.yml                      # One-command full-stack deployment
├── start.sh                                # Convenience startup script
├── .gitignore
├── README.md
└── LICENSE
```

---

## ⚙️ Environment Variables

Create a `.env` file in the `backend/` directory (or copy from `.env.example`):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | No | — | Google Gemini API key for enhanced AI responses. Without it, the built-in NLG engine is used (free, offline). |

All other configuration (paths, model params, feature lists) is centralized in `backend/src/utils/config.py`.

---

## 🔮 Roadmap

- [x] **Phase 1**: Data Engineering + Feature Engineering + ML Models + SHAP + FastAPI (7 endpoints)
- [x] **Phase 2**: RAG AI Copilot + Knowledge Base + NLG Engine + Gemini Integration (3 endpoints)
- [x] **Phase 3**: Next.js 16 Interactive Dashboard with 6 tabs + API Proxy
- [ ] **Phase 4**: Docker Compose full-stack deployment + CI/CD pipeline
- [ ] **Phase 5**: Real-time data streaming (Kafka/WebSocket) for live dashboard updates
- [ ] **Phase 6**: Multi-model ensemble (stacking/blending) for improved accuracy
- [ ] **Phase 7**: User authentication + role-based access control
- [ ] **Phase 8**: Automated retraining pipeline (daily/weekly model refresh)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Built with** Python · FastAPI · CatBoost · SHAP · Next.js · React · Tailwind CSS

</div>