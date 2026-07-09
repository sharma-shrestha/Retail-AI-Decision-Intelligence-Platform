#!/bin/bash
# Start Retail AI Platform — Backend + Frontend
set -e

echo "=== Starting Retail AI Platform ==="

# 1. Start FastAPI backend
echo "[1/2] Starting FastAPI backend on port 8000..."
cd "$(dirname "$0")/retail-ai-platform"
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "       Backend PID: $BACKEND_PID"

# Wait for backend to be ready
for i in $(seq 1 15); do
  if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "       Backend is healthy ✓"
    break
  fi
  sleep 1
done

# 2. Start Next.js frontend
echo "[2/2] Starting Next.js frontend on port 3000..."
cd "$(dirname "$0")"
npx next dev --port 3000 &
NEXT_PID=$!
echo "       Next.js PID: $NEXT_PID"

echo ""
echo "=== Platform is running ==="
echo "  Frontend: http://localhost:3000"
echo "  Backend:  http://localhost:8000"
echo "  Docs:     http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers."

# Graceful shutdown
trap "kill $BACKEND_PID $NEXT_PID 2>/dev/null; echo 'Stopped.'; exit 0" INT TERM
wait