@echo off
echo 🧠 Semantic Visual Intelligence System
echo ────────────────────────────────────────

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

if not exist "venv\" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate

echo 📥 Installing dependencies...
pip install -q -r requirements.txt

if not exist ".env" (
    copy .env.example .env
    echo ⚠️  .env created. Add your API keys for LLM features.
)

echo.
echo 🚀 Starting SVIS at http://localhost:8501
echo ────────────────────────────────────────
streamlit run app.py
pause
