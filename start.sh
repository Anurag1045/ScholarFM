#!/bin/bash
set -e

# Install deps if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
  pip install -r requirements.txt
fi

export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:?Set ANTHROPIC_API_KEY first}"

echo ""
echo "  ScholarFM running at → http://localhost:8000"
echo ""

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
