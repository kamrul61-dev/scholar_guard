"""
ScholarGuard — Agent Definitions

Defines 4 specialized agents that form the verification pipeline:
  1. Vision Clerk   — OCR extraction from claim images (Groq Vision)
  2. Librarian      — RAG retrieval from local documents
  3. Fact-Checker   — Web search & grounding verification
  4. Auditor        — Compiles a structured audit report
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.utils import get_llm, format_report
from src.tools import ocr_tool, rag_search_tool, tavily_search_tool, google_grounding_tool


# ------------------------------------------------------------------ #
#  Agent 1: Vision Clerk (OCR)
# ------------------------------------------------------------------ #

VISION_CLERK_SYSTEM = """You are the Vision Clerk, a specialized OCR agent for ScholarGuard.

Your role:
- Extract text accurately from images of academic claims
- Preserve formatting: paragraphs, headings, lists, mathematical notation
- Identify and flag any text that is unclear or partially legible
- Output ONLY the extracted text — do not interpret or verify it

If the input is already text (not an image), simply pass it through unchanged.
"""


def create_vision_clerk():
    """Create the Vision Clerk agent for OCR text extraction.

    Returns:
        A callable agent that processes images and extracts text.
    """
    llm = get_llm()

    def run_vision_clerk(state: dict) -> dict:
        """Execute the Vision Clerk agent.

        Args:
            state: Pipeline state containing 'image_data' and/or 'claim_text'.

        Returns:
            Updated state with 'claim_text' populated.
        """
        # If text is already provided, skip OCR
        if state.get("claim_text") and not state.get("image_data"):
            return state

        # If image is provided, extract text via OCR tool
        if state.get("image_data"):
            extracted_text = ocr_tool.invoke({"image_base64": state["image_data"]})
            state["claim_text"] = extracted_text
            state["ocr_performed"] = True

        return state

    return run_vision_clerk


# ------------------------------------------------------------------ #
#  Agent 2: Librarian (RAG)
# ------------------------------------------------------------------ #

LIBRARIAN_SYSTEM = """You are the Librarian, a RAG specialist agent for ScholarGuard.

Your role:
- Search the local academic document database for evidence related to the claim
- Retrieve the most relevant passages from ingested PDFs
- Summarize findings with source citations (document name, page number)
- Clearly state if no relevant evidence was found locally

Present evidence objectively — do not form your own verdict on the claim.
"""


def create_librarian():
    """Create the Librarian agent for RAG-based evidence retrieval.

    Returns:
        A callable agent that searches local documents.
    """
    llm = get_llm()

    def run_librarian(state: dict) -> dict:
        """Execute the Librarian agent.

        Args:
            state: Pipeline state with 'claim_text'.

        Returns:
            Updated state with 'rag_evidence' populated.
        """
        claim = state.get("claim_text", "")
        if not claim:
            state["rag_evidence"] = "No claim text provided for RAG search."
            return state

        # Retrieve evidence from the vector store
        raw_evidence = rag_search_tool.invoke({"query": claim})

        # Use LLM to synthesize the retrieved evidence
        messages = [
            SystemMessage(content=LIBRARIAN_SYSTEM),
            HumanMessage(content=(
                f"The following claim needs evidence from our document database:\n\n"
                f"**Claim:** \"{claim}\"\n\n"
                f"**Retrieved Documents:**\n{raw_evidence}\n\n"
                f"Summarize the relevant evidence found. Include source citations. "
                f"If no relevant evidence was found, clearly state that."
            )),
        ]

        response = llm.invoke(messages)
        state["rag_evidence"] = response.content

        return state

    return run_librarian


# ------------------------------------------------------------------ #
#  Agent 3: Fact-Checker (Search + Grounding)
# ------------------------------------------------------------------ #

FACT_CHECKER_SYSTEM = """You are the Fact-Checker, a web verification agent for ScholarGuard.

Your role:
- Verify academic claims using internet search and web grounding
- Cross-reference multiple sources to assess claim accuracy
- Identify supporting and contradicting evidence
- Note the credibility and type of each source (journal, institution, news, etc.)
- Present findings objectively with clear source attribution

Provide a thorough and balanced analysis. Do not provide a final verdict —
that is the Auditor's responsibility.
"""


def create_fact_checker():
    """Create the Fact-Checker agent for web-based verification.

    Returns:
        A callable agent that verifies claims using Tavily and web grounding.
    """
    llm = get_llm()

    def run_fact_checker(state: dict) -> dict:
        """Execute the Fact-Checker agent.

        Args:
            state: Pipeline state with 'claim_text'.

        Returns:
            Updated state with 'search_evidence' populated.
        """
        claim = state.get("claim_text", "")
        if not claim:
            state["search_evidence"] = "No claim text provided for fact-checking."
            return state

        # --- Tavily Internet Search ---
        tavily_results = tavily_search_tool.invoke({"query": claim})

        # --- Web Grounding ---
        grounding_results = google_grounding_tool.invoke({"claim": claim})

        # --- Synthesize with LLM ---
        messages = [
            SystemMessage(content=FACT_CHECKER_SYSTEM),
            HumanMessage(content=(
                f"Verify the following academic claim using the evidence gathered:\n\n"
                f"**Claim:** \"{claim}\"\n\n"
                f"**Internet Search Results (Tavily):**\n{tavily_results}\n\n"
                f"**Web Grounding Results:**\n{grounding_results}\n\n"
                f"Analyze the evidence from both sources. Identify which evidence "
                f"supports or contradicts the claim. Note source credibility."
            )),
        ]

        response = llm.invoke(messages)
        state["search_evidence"] = response.content

        return state

    return run_fact_checker


# ------------------------------------------------------------------ #
#  Agent 4: Verification Auditor (Report)
# ------------------------------------------------------------------ #

AUDITOR_SYSTEM = """You are the Verification Auditor, the final report agent for ScholarGuard.

Your role:
- Review ALL evidence gathered by the Librarian (RAG) and Fact-Checker (Web)
- Synthesize a clear, structured verdict on the academic claim
- Provide a confidence score between 0.0 and 1.0
- Classify the verdict as: "Verified", "Refuted", or "Inconclusive"

Verdict Guidelines:
- "Verified" (confidence >= 0.7): Strong, consistent evidence supports the claim
- "Refuted"  (confidence >= 0.7): Strong, consistent evidence contradicts the claim
- "Inconclusive" (any confidence): Mixed, insufficient, or conflicting evidence

You MUST respond in EXACTLY this JSON format:
{
  "verdict": "Verified" | "Refuted" | "Inconclusive",
  "confidence": 0.0 to 1.0,
  "reasoning": "Brief explanation of your verdict..."
}
"""


def create_auditor():
    """Create the Verification Auditor agent for final report generation.

    Returns:
        A callable agent that produces the final audit report.
    """
    llm = get_llm()

    def run_auditor(state: dict) -> dict:
        """Execute the Verification Auditor agent.

        Args:
            state: Pipeline state with 'claim_text', 'rag_evidence',
                   and 'search_evidence'.

        Returns:
            Updated state with 'verdict', 'confidence', and 'report'.
        """
        claim = state.get("claim_text", "No claim provided.")
        rag_evidence = state.get("rag_evidence", "No RAG evidence available.")
        search_evidence = state.get("search_evidence", "No search evidence available.")

        messages = [
            SystemMessage(content=AUDITOR_SYSTEM),
            HumanMessage(content=(
                f"Review all evidence and produce a final verdict.\n\n"
                f"**Claim:** \"{claim}\"\n\n"
                f"**Local Document Evidence (RAG):**\n{rag_evidence}\n\n"
                f"**Web Search Evidence:**\n{search_evidence}\n\n"
                f"Respond ONLY with the JSON format specified in your instructions."
            )),
        ]

        response = llm.invoke(messages)

        # Parse the LLM response to extract verdict and confidence
        import json
        try:
            # Clean up potential markdown code fences
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1]  # Remove first line
                content = content.rsplit("```", 1)[0]  # Remove last fence
            content = content.strip()

            result = json.loads(content)
            verdict = result.get("verdict", "Inconclusive")
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")
        except (json.JSONDecodeError, ValueError):
            verdict = "Inconclusive"
            confidence = 0.5
            reasoning = response.content

        # Build the formatted report
        state["verdict"] = verdict
        state["confidence"] = confidence
        state["reasoning"] = reasoning
        state["report"] = format_report(
            claim=claim,
            rag_evidence=rag_evidence,
            search_evidence=search_evidence,
            verdict=verdict,
            confidence=confidence,
        )

        return state

    return run_auditor
