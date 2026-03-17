#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_local.sh — Start the F1 dashboard without Docker (development mode)
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "=== F1 Analytics Dashboard — Local Startup ==="

# 1. Copy .env if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "[INFO] Created .env from .env.example — edit it if needed."
fi

# 2. Create cache directories
mkdir -p cache/fastf1 cache/models

# 3. Start FastAPI in the background
echo "[INFO] Starting FastAPI on http://localhost:8000 ..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!
echo "[INFO] API PID: $API_PID"

# Give the API a moment to start
sleep 2

# 4. Start Streamlit
echo "[INFO] Starting Streamlit dashboard on http://localhost:8501 ..."
streamlit run dashboard/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0

# Cleanup on exit
trap "kill $API_PID 2>/dev/null" EXIT
