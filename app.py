"""
ScholarGuard — Streamlit UI

Premium academic integrity & fact-checking interface with:
  • Text input and image upload for claims
  • Sidebar for PDF ingestion and system status
  • Real-time pipeline progress tracking
  • Structured audit report display
"""

import os
import sys
import base64
import streamlit as st

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import initialize_langsmith, get_confidence_color
from src.database import ingest_pdfs, get_collection_stats
from main import run_pipeline


# ------------------------------------------------------------------ #
#  Page Config & Custom Styles
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="ScholarGuard — Academic Integrity Checker",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium dark-themed CSS
st.markdown("""
<style>
    /* ---- Import Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ---- Global Theme ---- */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Hero Header ---- */
    .hero-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(48, 43, 99, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-header h1 {
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-header .subtitle {
        color: rgba(255, 255, 255, 0.7);
        font-size: 1.05rem;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    .hero-header .badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.8rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    /* ---- Status Cards ---- */
    .status-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }
    .status-card .label {
        color: rgba(255, 255, 255, 0.55);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
    }
    .status-card .value {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 4px;
    }

    /* ---- Pipeline Steps ---- */
    .pipeline-step {
        background: linear-gradient(135deg, #1e1e30 0%, #252540 100%);
        border-left: 4px solid #667eea;
        border-radius: 0 10px 10px 0;
        padding: 0.8rem 1.2rem;
        margin-bottom: 0.6rem;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }
    .pipeline-step.active {
        border-left-color: #22c55e;
        background: linear-gradient(135deg, #1a2e1a 0%, #1e3a1e 100%);
    }
    .pipeline-step.completed {
        border-left-color: #22c55e;
        opacity: 0.85;
    }
    .pipeline-step .step-icon {
        font-size: 1.3rem;
    }
    .pipeline-step .step-name {
        color: #ffffff;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .pipeline-step .step-desc {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.78rem;
    }

    /* ---- Verdict Banner ---- */
    .verdict-banner {
        border-radius: 14px;
        padding: 1.8rem 2rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.2);
    }
    .verdict-verified {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .verdict-refuted {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .verdict-inconclusive {
        background: linear-gradient(135deg, #78350f, #92400e);
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .verdict-banner h2 {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
    }
    .verdict-banner .conf {
        color: rgba(255, 255, 255, 0.75);
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    /* ---- Sidebar Styling ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff;
    }

    /* ---- Input Area ---- */
    .input-section {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Initialize
# ------------------------------------------------------------------ #

initialize_langsmith()


# ------------------------------------------------------------------ #
#  Sidebar
# ------------------------------------------------------------------ #

with st.sidebar:
    # Logo / Branding
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 3rem;">🛡️</div>
        <h2 style="margin: 0; color: #ffffff; font-weight: 800;">ScholarGuard</h2>
        <p style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 4px;">
            Academic Integrity System
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- API Key Status ---
    st.markdown("### 🔑 API Configuration")

    groq_key = os.environ.get("GROQ_API_KEY", "")
    tavily_key = os.environ.get("TAVILY_API_KEY", "")
    langsmith_key = os.environ.get("LANGSMITH_API_KEY", "")

    def key_status(key: str, placeholder: str = "your_") -> str:
        if key and placeholder not in key:
            return "✅ Configured"
        return "❌ Missing"

    st.markdown(f"""
    <div class="status-card">
        <div class="label">Groq LLM</div>
        <div class="value">{key_status(groq_key)}</div>
    </div>
    <div class="status-card">
        <div class="label">Tavily Search</div>
        <div class="value">{key_status(tavily_key)}</div>
    </div>
    <div class="status-card">
        <div class="label">LangSmith</div>
        <div class="value">{key_status(langsmith_key)}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # --- Knowledge Base / PDF Ingestion ---
    st.markdown("### 📚 Knowledge Base")

    stats = get_collection_stats()
    doc_count = stats.get("document_count", 0)
    db_status = stats.get("status", "unknown")

    st.markdown(f"""
    <div class="status-card">
        <div class="label">Vector Store Status</div>
        <div class="value">{"🟢 " + str(doc_count) + " chunks" if doc_count > 0 else "🔴 Empty"}</div>
    </div>
    """, unsafe_allow_html=True)

    # PDF Upload
    uploaded_files = st.file_uploader(
        "Upload PDFs for RAG",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload academic papers or reference documents",
    )

    if uploaded_files:
        if st.button("📥 Ingest Documents", use_container_width=True):
            with st.spinner("Processing PDFs..."):
                # Save uploaded files to data/raw/
                raw_dir = os.path.join(os.path.dirname(__file__), "data", "raw")
                os.makedirs(raw_dir, exist_ok=True)

                for uploaded_file in uploaded_files:
                    file_path = os.path.join(raw_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                # Run ingestion
                result = ingest_pdfs(raw_dir)
                st.success(
                    f"✅ Processed **{result['files_processed']}** files → "
                    f"**{result['chunks_added']}** chunks added"
                )
                st.rerun()

    st.divider()

    # --- Pipeline Info ---
    st.markdown("### 🔄 Verification Pipeline")
    pipeline_steps = [
        ("👁️", "Vision Clerk", "OCR extraction"),
        ("📖", "Librarian", "RAG retrieval"),
        ("🔍", "Fact-Checker", "Web verification"),
        ("📋", "Auditor", "Report generation"),
    ]
    for icon, name, desc in pipeline_steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <span class="step-icon">{icon}</span>
            <div>
                <div class="step-name">{name}</div>
                <div class="step-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------ #
#  Main Content Area
# ------------------------------------------------------------------ #

# Hero Header
st.markdown("""
<div class="hero-header">
    <h1>🛡️ ScholarGuard</h1>
    <div class="subtitle">AI-Powered Academic Integrity & Fact-Checking System</div>
    <div class="badge">Multi-Agent Verification Pipeline</div>
</div>
""", unsafe_allow_html=True)

# Input Tabs
tab_text, tab_image = st.tabs(["📝 Text Claim", "🖼️ Image Claim"])

with tab_text:
    st.markdown("#### Enter an academic claim to verify")
    claim_text = st.text_area(
        "Claim",
        placeholder="e.g., 'The human genome contains approximately 20,000-25,000 protein-coding genes.'",
        height=120,
        label_visibility="collapsed",
    )
    submit_text = st.button(
        "🔍 Verify Claim",
        key="verify_text",
        use_container_width=True,
        type="primary",
    )

with tab_image:
    st.markdown("#### Upload an image of an academic claim")
    uploaded_image = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
    )
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded claim image", use_container_width=True)
    submit_image = st.button(
        "🔍 Extract & Verify",
        key="verify_image",
        use_container_width=True,
        type="primary",
    )

# ------------------------------------------------------------------ #
#  Verification Execution
# ------------------------------------------------------------------ #

def display_results(result: dict):
    """Display the verification results with styled components."""
    verdict = result.get("verdict", "Inconclusive")
    confidence = result.get("confidence", 0.5)
    report = result.get("report", "")
    reasoning = result.get("reasoning", "")

    # Verdict Banner
    verdict_class = {
        "Verified": "verdict-verified",
        "Refuted": "verdict-refuted",
    }.get(verdict, "verdict-inconclusive")

    verdict_emoji = {
        "Verified": "✅",
        "Refuted": "❌",
    }.get(verdict, "⚠️")

    st.markdown(f"""
    <div class="verdict-banner {verdict_class}">
        <h2>{verdict_emoji} {verdict}</h2>
        <div class="conf">Confidence: {confidence * 100:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence Progress Bar
    conf_color = get_confidence_color(confidence)
    st.progress(confidence)

    # Reasoning
    if reasoning:
        st.markdown("### 💡 Reasoning")
        st.info(reasoning)

    # Full Report
    st.markdown("---")
    st.markdown(report)

    # Evidence Expanders
    with st.expander("📚 RAG Evidence (Local Documents)", expanded=False):
        st.markdown(result.get("rag_evidence", "_No RAG evidence available._"))

    with st.expander("🌐 Web Search Evidence", expanded=False):
        st.markdown(result.get("search_evidence", "_No web evidence available._"))


# Handle Text Submission
if submit_text and claim_text:
    with st.spinner(""):
        # Show progress
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Verification in Progress...")

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.markdown("**Step 1/3:** 📖 Librarian — Searching local documents...")
            progress_bar.progress(15)

            import time
            # Run pipeline
            status_text.markdown("**Step 2/3:** 🔍 Fact-Checker — Searching the web...")
            progress_bar.progress(45)

            result = run_pipeline(claim_text=claim_text)

            status_text.markdown("**Step 3/3:** 📋 Auditor — Generating report...")
            progress_bar.progress(85)

            time.sleep(0.5)
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

        # Check for errors
        if result.get("error"):
            st.error(f"❌ Pipeline Error: {result['error']}")
        else:
            display_results(result)

# Handle Image Submission
elif submit_image and uploaded_image:
    with st.spinner(""):
        progress_container = st.container()
        with progress_container:
            st.markdown("### 🔄 Verification in Progress...")

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.markdown("**Step 1/4:** 👁️ Vision Clerk — Extracting text from image...")
            progress_bar.progress(10)

            # Encode image to base64
            image_bytes = uploaded_image.getvalue()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            status_text.markdown("**Step 2/4:** 📖 Librarian — Searching local documents...")
            progress_bar.progress(30)

            status_text.markdown("**Step 3/4:** 🔍 Fact-Checker — Searching the web...")
            progress_bar.progress(55)

            result = run_pipeline(image_data=image_b64)

            status_text.markdown("**Step 4/4:** 📋 Auditor — Generating report...")
            progress_bar.progress(90)

            import time
            time.sleep(0.5)
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()

        # Show extracted text if OCR was performed
        if result.get("ocr_performed"):
            with st.expander("👁️ Extracted Text (OCR)", expanded=True):
                st.markdown(result.get("claim_text", ""))

        # Check for errors
        if result.get("error"):
            st.error(f"❌ Pipeline Error: {result['error']}")
        else:
            display_results(result)

# Empty State
elif not submit_text and not submit_image:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="status-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📝</div>
            <div class="label">Step 1</div>
            <div style="color: #fff; margin-top: 4px; font-size: 0.9rem;">
                Enter a claim or upload an image
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="status-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔍</div>
            <div class="label">Step 2</div>
            <div style="color: #fff; margin-top: 4px; font-size: 0.9rem;">
                AI agents verify using RAG & Web
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="status-card" style="text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📋</div>
            <div class="label">Step 3</div>
            <div style="color: #fff; margin-top: 4px; font-size: 0.9rem;">
                Get a structured audit report
            </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: rgba(255,255,255,0.3); font-size: 0.8rem; padding: 1rem;'>"
    "ScholarGuard — Academic Integrity & Fact-Checking System • "
    "Built with LangGraph, Groq, ChromaDB & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
