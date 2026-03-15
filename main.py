"""
ScholarGuard — LangGraph Orchestration

Defines the multi-agent verification pipeline using LangGraph's StateGraph.
Routes claims through Vision Clerk → Librarian → Fact-Checker → Auditor
with conditional logic to skip OCR when input is plain text.
"""

import operator
from typing import TypedDict, Optional, Annotated
from langgraph.graph import StateGraph, END
from src.utils import initialize_langsmith
from src.agents import (
    create_vision_clerk,
    create_librarian,
    create_fact_checker,
    create_auditor,
)


# ------------------------------------------------------------------ #
#  Pipeline State Schema
# ------------------------------------------------------------------ #

class ScholarGuardState(TypedDict):
    """State that flows through the entire verification pipeline.

    Attributes:
        claim_text: The extracted or provided claim text.
        image_data: Base64-encoded image data (None if text input).
        input_type: Either 'text' or 'image'.
        ocr_performed: Whether OCR was executed.
        rag_evidence: Evidence from local document retrieval.
        search_evidence: Evidence from web search and grounding.
        verdict: Final verdict (Verified / Refuted / Inconclusive).
        confidence: Confidence score 0.0–1.0.
        reasoning: Explanation for the verdict.
        report: Formatted audit report string.
        error: Error message if something goes wrong.
    """
    claim_text: Optional[str]
    image_data: Optional[str]
    input_type: str
    ocr_performed: bool
    rag_evidence: Optional[str]
    search_evidence: Optional[str]
    verdict: Optional[str]
    confidence: Optional[float]
    reasoning: Optional[str]
    report: Optional[str]
    error: Optional[str]


# ------------------------------------------------------------------ #
#  Conditional Router
# ------------------------------------------------------------------ #

def route_input(state: ScholarGuardState) -> str:
    """Determine the first processing step based on input type.

    If the input is an image, route to the Vision Clerk for OCR.
    If the input is text, skip directly to the Librarian.

    Args:
        state: Current pipeline state.

    Returns:
        Name of the next node: 'vision_clerk' or 'librarian'.
    """
    if state.get("input_type") == "image" and state.get("image_data"):
        return "vision_clerk"
    return "librarian"


# ------------------------------------------------------------------ #
#  Graph Builder
# ------------------------------------------------------------------ #

def build_graph() -> StateGraph:
    """Construct the LangGraph StateGraph for the verification pipeline.

    Pipeline Flow:
        [Input] → route_input → Vision Clerk (if image)
                               ↓
                           Librarian (RAG)
                               ↓
                          Fact-Checker (Search)
                               ↓
                            Auditor (Report)
                               ↓
                             [END]

    Returns:
        Compiled LangGraph application ready for invocation.
    """
    # Instantiate agents
    vision_clerk = create_vision_clerk()
    librarian = create_librarian()
    fact_checker = create_fact_checker()
    auditor = create_auditor()

    # Build the state graph
    workflow = StateGraph(ScholarGuardState)

    # Add nodes
    workflow.add_node("vision_clerk", vision_clerk)
    workflow.add_node("librarian", librarian)
    workflow.add_node("fact_checker", fact_checker)
    workflow.add_node("auditor", auditor)

    # Entry point — conditional routing based on input type
    workflow.set_conditional_entry_point(
        route_input,
        {
            "vision_clerk": "vision_clerk",
            "librarian": "librarian",
        },
    )

    # Define edges — the pipeline flows linearly after routing
    workflow.add_edge("vision_clerk", "librarian")
    workflow.add_edge("librarian", "fact_checker")
    workflow.add_edge("fact_checker", "auditor")
    workflow.add_edge("auditor", END)

    # Compile the graph
    app = workflow.compile()
    return app


# ------------------------------------------------------------------ #
#  Pipeline Runner
# ------------------------------------------------------------------ #

def run_pipeline(
    claim_text: Optional[str] = None,
    image_data: Optional[str] = None,
) -> dict:
    """Execute the full ScholarGuard verification pipeline.

    Args:
        claim_text: Plain text claim to verify (provide this OR image_data).
        image_data: Base64-encoded image of a claim (provide this OR claim_text).

    Returns:
        Final pipeline state dict containing the report and all evidence.

    Raises:
        ValueError: If neither claim_text nor image_data is provided.
    """
    if not claim_text and not image_data:
        raise ValueError("Provide either 'claim_text' or 'image_data'.")

    # Initialize LangSmith tracing
    initialize_langsmith()

    # Determine input type
    input_type = "image" if image_data else "text"

    # Build initial state
    initial_state: ScholarGuardState = {
        "claim_text": claim_text or "",
        "image_data": image_data,
        "input_type": input_type,
        "ocr_performed": False,
        "rag_evidence": None,
        "search_evidence": None,
        "verdict": None,
        "confidence": None,
        "reasoning": None,
        "report": None,
        "error": None,
    }

    # Build and run the graph
    app = build_graph()

    try:
        final_state = app.invoke(initial_state)
        return final_state
    except Exception as e:
        initial_state["error"] = str(e)
        initial_state["report"] = f"## ❌ Pipeline Error\n\n{e}"
        return initial_state


# ------------------------------------------------------------------ #
#  CLI Entry Point (for testing)
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py \"<your academic claim>\"")
        sys.exit(1)

    claim = " ".join(sys.argv[1:])
    print(f"\n🔍 Verifying claim: \"{claim}\"\n")
    print("=" * 60)

    result = run_pipeline(claim_text=claim)

    if result.get("report"):
        print(result["report"])
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")
