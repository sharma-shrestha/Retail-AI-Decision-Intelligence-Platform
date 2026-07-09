import { NextRequest, NextResponse } from "next/server";

const BACKEND = "http://localhost:8000";

/**
 * Catch-all API proxy — forwards any request under /api/* to the FastAPI backend.
 *
 * Examples:
 *   GET  /api/health                      → http://localhost:8000/health
 *   GET  /api/v1/models                   → http://localhost:8000/api/v1/models
 *   POST /api/v1/forecast                 → http://localhost:8000/api/v1/forecast
 *   GET  /api/v1/analytics/summary        → http://localhost:8000/api/v1/analytics/summary
 *   POST /api/v1/ask-ai                   → http://localhost:8000/api/v1/ask-ai
 *   GET  /api/v1/ask-ai/report            → http://localhost:8000/api/v1/ask-ai/report
 */
async function proxy(request: NextRequest, method: string) {
  const slug = request.nextUrl.pathname.replace(/^\/api\//, "");
  const search = request.nextUrl.search;
  // Backend routes: /health (root) or /api/v1/* (prefixed)
  // Frontend calls /api/health → slug "health" → backend /health
  // Frontend calls /api/v1/... → slug "v1/..." → backend /api/v1/...
  const backendPath = slug.startsWith("v1/") ? `/api/${slug}` : `/${slug}`;
  const url = `${BACKEND}${backendPath}${search}`;

  const headers = new Headers();
  const contentType = request.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);

  let body: string | undefined;
  if (method !== "GET" && method !== "HEAD") {
    body = await request.text();
  }

  try {
    const res = await fetch(url, { method, headers, body, cache: "no-store" , signal: AbortSignal.timeout(120000) });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch {
    return NextResponse.json(
      { detail: "Backend service unavailable. Is the FastAPI server running on port 8000?" },
      { status: 502 }
    );
  }
}

export async function GET(req: NextRequest) {
  return proxy(req, "GET");
}
export async function POST(req: NextRequest) {
  return proxy(req, "POST");
}
export async function PUT(req: NextRequest) {
  return proxy(req, "PUT");
}
export async function DELETE(req: NextRequest) {
  return proxy(req, "DELETE");
}