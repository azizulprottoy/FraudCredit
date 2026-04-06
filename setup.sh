#!/bin/bash
cd "$(dirname "$0")"

echo "==================================================="
echo "  CreditCardFraudRnD - Environment Setup"
echo "==================================================="
echo ""

# Check if Python is installed (try python3 first, then python)
if command -v python3 &>/dev/null; then
    PYTHON=python3
    PIP=pip3
elif command -v python &>/dev/null; then
    PYTHON=python
    PIP=pip
else
    echo "[ERROR] Python not found. Please install Python 3.9+ (brew install python)."
    exit 1
fi

echo "[INFO] Using: $($PYTHON --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    $PYTHON -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        exit 1
    fi
else
    echo "[INFO] Virtual environment already exists."
fi

# Activate venv
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Upgrading pip..."
python -m pip install --upgrade pip

echo "[INFO] Installing dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies."
    exit 1
fi

echo ""
echo "==================================================="
echo "  SETUP COMPLETE!"
echo "  You can now run the app using: bash run.sh"
echo "==================================================="
echo ""
read -p "Press Enter to continue..."