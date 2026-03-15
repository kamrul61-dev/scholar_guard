"""
ScholarGuard — Tool Definitions

Defines all LangChain-compatible tools used by the agents:
  • OCR Tool          — extract text from images via Groq Vision (Llama 3.2 Vision)
  • RAG Search Tool   — semantic retrieval from ChromaDB
  • Tavily Search     — internet search for fact-checking
  • Google Grounding  — web-based claim verification via LLM + Tavily
"""

import os
import base64
from typing import Optional
from langchain_core.tools import tool
from src.utils import traced_tool_call, get_llm, get_vision_llm
from src.database import get_retriever


# ------------------------------------------------------------------ #
#  1. OCR Tool — Vision Clerk's primary tool
# ------------------------------------------------------------------ #

@tool
@traced_tool_call
def ocr_tool(image_base64: str) -> str:
    """Extract text from an image of an academic claim using Groq Vision.

    Args:
        image_base64: Base64-encoded image string.

    Returns:
        Extracted text from the image, or an error message.
    """
    try:
        from langchain_core.messages import HumanMessage

        llm = get_vision_llm()

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "You are an OCR specialist. Extract ALL text from this image "
                        "accurately and completely. Preserve the original structure "
                        "(paragraphs, headings, lists, tables). If there are "
                        "mathematical formulas, represent them in plain text. "
                        "Return ONLY the extracted text, nothing else."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                    },
                },
            ]
        )

        response = llm.invoke([message])
        return response.content

    except Exception as e:
        return f"OCR extraction failed: {e}"


# ------------------------------------------------------------------ #
#  2. RAG Search Tool — Librarian's primary tool
# ------------------------------------------------------------------ #

@tool
@traced_tool_call
def rag_search_tool(query: str) -> str:
    """Search the local academic document database for relevant evidence.

    Performs semantic similarity search over PDFs ingested into ChromaDB.

    Args:
        query: The claim or search query to find evidence for.

    Returns:
        Concatenated relevant document excerpts, or a message if
        no documents are available.
    """
    retriever = get_retriever(k=5)

    if retriever is None:
        return (
            "No documents found in the knowledge base. "
            "Please upload PDFs via the sidebar to enable RAG search."
        )

    try:
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant passages found in the knowledge base."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            # Extract just the filename from the full path
            source_name = os.path.basename(source)
            results.append(
                f"**[Source {i}]** _{source_name}_ (Page {page})\n"
                f"{doc.page_content}\n"
            )

        return "\n---\n".join(results)

    except Exception as e:
        return f"RAG search failed: {e}"


# ------------------------------------------------------------------ #
#  3. Tavily Search Tool — Fact-Checker's web search tool
# ------------------------------------------------------------------ #

@tool
@traced_tool_call
def tavily_search_tool(query: str) -> str:
    """Search the internet for evidence related to an academic claim.

    Uses the Tavily API for high-quality, AI-optimized web search results.

    Args:
        query: The claim or search query to verify on the internet.

    Returns:
        Formatted search results with titles, URLs, and content snippets.
    """
    try:
        from tavily import TavilyClient

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key or api_key == "your_tavily_api_key_here":
            return "Tavily API key not configured. Please set TAVILY_API_KEY in .env."

        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_answer=True,
        )

        results = []

        # Include Tavily's generated answer if available
        if response.get("answer"):
            results.append(f"**🔍 AI Summary:** {response['answer']}\n")

        # Format individual results
        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "No content available.")
            score = result.get("score", 0)

            results.append(
                f"**[Result {i}]** [{title}]({url})\n"
                f"_Relevance: {score:.2f}_\n"
                f"{content}\n"
            )

        return "\n---\n".join(results) if results else "No search results found."

    except Exception as e:
        return f"Tavily search failed: {e}"


# ------------------------------------------------------------------ #
#  4. Web Grounding Tool — Fact-Checker's grounding tool
# ------------------------------------------------------------------ #

@tool
@traced_tool_call
def google_grounding_tool(claim: str) -> str:
    """Verify an academic claim using web grounding via LLM + Tavily.

    Performs a focused Tavily search and uses the LLM to synthesize
    a grounded verification of the claim with source citations.

    Args:
        claim: The specific academic claim to verify.

    Returns:
        Grounded verification response with source citations.
    """
    try:
        from tavily import TavilyClient
        from langchain_core.messages import HumanMessage, SystemMessage

        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key or api_key == "your_tavily_api_key_here":
            return "Tavily API key not configured. Please set TAVILY_API_KEY in .env."

        # Perform a focused search specifically for grounding
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=f"fact check: {claim}",
            search_depth="advanced",
            max_results=5,
            include_raw_content=False,
        )

        # Compile search context
        search_context = ""
        for i, result in enumerate(response.get("results", []), 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")
            search_context += f"[{i}] {title} ({url})\n{content}\n\n"

        # Use LLM to synthesize a grounded verdict
        llm = get_llm()
        messages = [
            SystemMessage(content=(
                "You are an academic fact-checker performing web grounding. "
                "Based on the search results provided, verify the given claim. "
                "Cite specific sources by number. Be objective and thorough."
            )),
            HumanMessage(content=(
                f"Claim to verify: \"{claim}\"\n\n"
                f"Search Results:\n{search_context}\n\n"
                f"Provide:\n"
                f"1. Whether the claim is accurate, inaccurate, or partially accurate\n"
                f"2. Key evidence supporting your assessment with source citations\n"
                f"3. Any nuances or caveats"
            )),
        ]

        result = llm.invoke(messages)
        return result.content

    except Exception as e:
        return f"Web grounding verification failed: {e}"
