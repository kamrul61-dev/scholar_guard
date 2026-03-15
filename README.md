# 🛡️ ScholarGuard

**Academic Integrity & Fact-Checking Multi-Agent System**

ScholarGuard is an AI-powered chatbot that verifies academic claims using local documents (RAG) and live web search, then generates a structured audit report with verdict and confidence score.

---

## 🏗️ Architecture

```
User Input (Text / Image)
        │
        ├── Image ──→ 👁️ Vision Clerk (OCR via Gemini Vision)
        │                        │
        │                        ▼
        └── Text ───→ 📖 Librarian (RAG via ChromaDB)
                                 │
                                 ▼
                      🔍 Fact-Checker (Tavily + Google Search)
                                 │
                                 ▼
                      📋 Verification Auditor (Structured Report)
                                 │
                                 ▼
                         Audit Report
                  (Verdict • Evidence • Confidence)
```

### Agents

| Agent | Role | Tools |
|-------|------|-------|
| **Vision Clerk** | Extracts text from images of claims | Gemini Vision OCR |
| **Librarian** | Retrieves evidence from local documents | ChromaDB RAG |
| **Fact-Checker** | Verifies claims on the internet | Tavily Search, Google Grounding |
| **Auditor** | Compiles a structured audit report | LLM Synthesis |

### Tech Stack

- **Framework:** LangChain / LangGraph
- **LLM:** Google Gemini 1.5 Pro
- **Vector DB:** ChromaDB (persistent)
- **Search:** Tavily API + Google Search Grounding
- **UI:** Streamlit
- **Monitoring:** LangSmith (full tracing)

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Scholar_Guard
```

### 2. Run the Setup Script

**Windows:**
```cmd
setup.bat
```

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a `.venv` virtual environment
- Install all dependencies from `requirements.txt`
- Generate a `.env` template

### 3. Configure API Keys


### 4. Activate the Virtual Environment

**Windows:**
```cmd
.venv\Scripts\activate
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 5. (Optional) Add Documents for RAG

Place PDF files in the `data/raw/` folder. These will be ingested into ChromaDB when you click **"Ingest Documents"** in the app sidebar.

### 6. Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📁 Project Structure

```
ScholarGuard/
├── .venv/                 # Virtual environment (git-ignored)
├── .env                   # API Keys & LangSmith config
├── .gitignore
├── setup.sh               # Mac/Linux setup
├── setup.bat              # Windows setup
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── app.py                 # Streamlit UI
├── main.py                # LangGraph orchestration
├── src/
│   ├── __init__.py
│   ├── agents.py          # 4 Agents: Vision Clerk, Librarian, Fact-Checker, Auditor
│   ├── tools.py           # OCR, RAG, Tavily Search, Google Grounding tools
│   ├── database.py        # ChromaDB setup and PDF ingestion
│   └── utils.py           # LangSmith config, LLM factory, helpers
└── data/
    ├── raw/               # PDF storage for RAG ingestion
    └── vector_store/      # Persistent ChromaDB data (git-ignored)
```

---

## 🔍 Usage

### Text Claims
1. Open the app in your browser
2. Select the **"📝 Text Claim"** tab
3. Enter an academic claim (e.g., *"The human genome contains approximately 20,000-25,000 protein-coding genes."*)
4. Click **"🔍 Verify Claim"**
5. Review the audit report

### Image Claims
1. Select the **"🖼️ Image Claim"** tab
2. Upload an image (PNG, JPG, JPEG, WebP) containing an academic claim
3. Click **"🔍 Extract & Verify"**
4. The Vision Clerk will OCR the text, then the pipeline runs as normal

### CLI Mode
```bash
python main.py "The speed of light is approximately 3 × 10^8 m/s"
```

---

## 📊 Monitoring with LangSmith

All agent executions and tool calls are automatically traced in LangSmith:

1. Open [smith.langchain.com](https://smith.langchain.com/)
2. Find the **"ScholarGuard"** project
3. Browse runs to see the full pipeline trace: inputs, outputs, tokens, latency

---

## 📝 Audit Report Format

Each verification produces a report with:

| Field | Description |
|-------|-------------|
| **Claim** | The original text under review |
| **RAG Evidence** | Passages from local documents |
| **Web Evidence** | Tavily + Google Search results |
| **Verdict** | ✅ Verified · ❌ Refuted · ⚠️ Inconclusive |
| **Confidence** | Score from 0% to 100% |

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| `GOOGLE_API_KEY` error | Verify key at [aistudio.google.com](https://aistudio.google.com/) |
| No RAG results | Upload PDFs via sidebar and click "Ingest Documents" |
| Tavily search fails | Check `TAVILY_API_KEY` in `.env` |
| LangSmith not tracing | Ensure `LANGCHAIN_TRACING_V2=true` and key is set |

---

## 📄 License

This project is part of an AI course final project.
