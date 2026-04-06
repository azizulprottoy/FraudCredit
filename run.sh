#!/bin/bash
cd "$(dirname "$0")"

echo "==================================================="
echo "  CreditCardFraudRnD - Starting Application"
echo "==================================================="
echo ""

# Check for venv
if [ ! -f "venv/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found. Please run 'setup.sh' first."
    exit 1
fi

# Activate venv
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Start Backend in a separate Terminal window
echo "[INFO] Starting FastAPI Backend on http://localhost:8000 ..."
osascript -e 'tell application "Terminal"
    do script "cd \"'"$(pwd)"'\" && source venv/bin/activate && python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000"
    activate
end tell'

# Wait for backend to initialize
echo "[INFO] Waiting for backend to initialize (5 seconds)..."
sleep 5

# Open Frontend Dashboard in default browser
echo "[INFO] Opening Dashboard in browser..."
open "frontend/index.html"

echo ""
echo "[INFO] Application is running!"
echo "[INFO] Keep the Backend Terminal window open."
echo ""
read -p "Press Enter to exit this script (this will NOT stop the backend)..."