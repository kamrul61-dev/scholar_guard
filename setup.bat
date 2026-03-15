@echo off
REM ============================================================
REM ScholarGuard — Environment Setup Script (Windows)
REM ============================================================

echo ============================================
echo    ScholarGuard — Environment Setup
echo ============================================
echo.

REM --- Step 1: Create virtual environment ---
echo [1/4] Creating virtual environment (.venv)...
if exist ".venv" (
    echo   WARNING: .venv already exists. Skipping creation.
) else (
    python -m venv .venv
    echo   DONE: Virtual environment created.
)

REM --- Step 2: Activate virtual environment ---
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat
echo   DONE: Virtual environment activated.

REM --- Step 3: Install dependencies ---
echo [3/4] Installing dependencies from requirements.txt...
pip install --upgrade pip
pip install -r requirements.txt
echo   DONE: All dependencies installed.

REM --- Step 4: Generate .env template ---
echo [4/4] Generating .env template...
if exist ".env" (
    echo   WARNING: .env already exists. Skipping generation.
) else (
    (
        echo # ============================================================
        echo # ScholarGuard — Environment Variables
        echo # ============================================================
        echo # Fill in your API keys below. Do NOT commit this file to git.
        echo.
        echo # Groq API Key (https://console.groq.com/keys)
        echo GROQ_API_KEY=your_groq_api_key_here
        echo.
        echo # Tavily Search API Key (https://tavily.com/)
        echo TAVILY_API_KEY=your_tavily_api_key_here
        echo.
        echo # LangSmith Monitoring (https://smith.langchain.com/)
        echo LANGSMITH_API_KEY=your_langsmith_api_key_here
        echo LANGCHAIN_TRACING_V2=true
        echo LANGCHAIN_PROJECT=ScholarGuard
    ) > .env
    echo   DONE: .env template created.
)

echo.
echo ============================================
echo    Setup complete!
echo ============================================
echo.
echo Next steps:
echo   1. Edit .env with your actual API keys
echo   2. Add PDFs to data\raw\ for RAG ingestion
echo   3. Run:  streamlit run app.py
echo.
pause
