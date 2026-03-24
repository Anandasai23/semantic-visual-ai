#!/usr/bin/env bash
# ─────────────────────────────────────────────
#  SVIS Quick Start Script
#  Usage: bash run.sh
# ─────────────────────────────────────────────

set -e

echo "🧠 Semantic Visual Intelligence System"
echo "────────────────────────────────────────"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "✅ Python $PYTHON_VERSION detected"

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install deps
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Copy .env if needed
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo "⚠️  .env created from template. Add your API keys to enable LLM features."
fi

# Load env
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs) 2>/dev/null || true
fi

echo ""
echo "🚀 Starting SVIS at http://localhost:8501 ..."
echo "────────────────────────────────────────"
streamlit run app.py
