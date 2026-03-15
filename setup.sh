#!/bin/bash
# ============================================================
# ScholarGuard — Environment Setup Script (Mac/Linux)
# ============================================================

set -e

echo "============================================"
echo "   ScholarGuard — Environment Setup"
echo "============================================"
echo ""

# --- Step 1: Create virtual environment ---
echo "[1/4] Creating virtual environment (.venv)..."
if [ -d ".venv" ]; then
    echo "  ⚠  .venv already exists. Skipping creation."
else
    python3 -m venv .venv
    echo "  ✅ Virtual environment created."
fi

# --- Step 2: Activate virtual environment ---
echo "[2/4] Activating virtual environment..."
source .venv/bin/activate
echo "  ✅ Virtual environment activated."

# --- Step 3: Install dependencies ---
echo "[3/4] Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt
echo "  ✅ All dependencies installed."

# --- Step 4: Generate .env template ---
echo "[4/4] Generating .env template..."
if [ -f ".env" ]; then
    echo "  ⚠  .env already exists. Skipping generation."
else
    cat > .env << 'EOF'
# ============================================================
# ScholarGuard — Environment Variables
# ============================================================
# Fill in your API keys below. Do NOT commit this file to git.

# Groq API Key (https://console.groq.com/keys)
GROQ_API_KEY=your_groq_api_key_here

# Tavily Search API Key (https://tavily.com/)
TAVILY_API_KEY=your_tavily_api_key_here

# LangSmith Monitoring (https://smith.langchain.com/)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ScholarGuard
EOF
    echo "  ✅ .env template created."
fi

echo ""
echo "============================================"
echo "   Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your actual API keys"
echo "  2. Add PDFs to data/raw/ for RAG ingestion"
echo "  3. Run:  streamlit run app.py"
echo ""
