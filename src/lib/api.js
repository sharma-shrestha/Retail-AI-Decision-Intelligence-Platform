// API Service Layer — pure JS, no TypeScript
// All requests proxy through Next.js API route → localhost:8000

const PROXY = "/api";

function buildUrl(path, query) {
  const q = query ? "?" + query : "";
  if (path.startsWith("/")) return PROXY + path + q;
  return PROXY + "/" + path + q;
}

// ── Health ──
export async function fetchHealth() {
  const res = await fetch(buildUrl("/health"));
  return res.json();
}

// ── Models ──
export async function fetchModels() {
  const res = await fetch(buildUrl("/api/v1/models"));
  return res.json();
}

// ── Forecast ──
export async function fetchForecast(body) {
  const res = await fetch(buildUrl("/api/v1/forecast"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Forecast failed");
  }
  return res.json();
}

// ── SHAP Explain ──
export async function fetchExplain(body) {
  const res = await fetch(buildUrl("/api/v1/explain"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Explain failed");
  }
  return res.json();
}

// ── Inventory ──
export async function fetchInventory(body) {
  const res = await fetch(buildUrl("/api/v1/inventory"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Inventory check failed");
  }
  return res.json();
}

// ── Analytics Summary ──
export async function fetchAnalyticsSummary() {
  const res = await fetch(buildUrl("/api/v1/analytics/summary"));
  return res.json();
}

// ── Top Products ──
export async function fetchTopProducts(n = 10, metric = "total_sales") {
  const res = await fetch(buildUrl("/api/v1/analytics/top-products", "n=" + n + "&metric=" + metric));
  return res.json();
}

// ── Feature Importance ──
export async function fetchFeatureImportance(model = "catboost", topN = 20) {
  const res = await fetch(buildUrl("/api/v1/analytics/feature-importance", "model=" + model + "&top_n=" + topN));
  return res.json();
}

// ── AI Copilot ──
export async function fetchAskAI(body) {
  const res = await fetch(buildUrl("/api/v1/ask-ai"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "AI query failed");
  }
  return res.json();
}

// ── Daily Report ──
export async function fetchDailyReport() {
  const res = await fetch(buildUrl("/api/v1/ask-ai/report"));
  return res.json();
}

// ── Underperforming Products ──
export async function fetchUnderperforming(storeId, n = 10) {
  const params = new URLSearchParams({ n: String(n) });
  if (storeId) params.set("store_id", storeId);
  const res = await fetch(buildUrl("/api/v1/ask-ai/underperforming", params.toString()));
  return res.json();
}