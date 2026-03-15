"""
ScholarGuard — ChromaDB Vector Store & PDF Ingestion

Handles persistent vector database creation, PDF document loading,
text chunking, embedding (via HuggingFace), and retriever construction for RAG.
"""

import os
from typing import Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #

VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "vector_store")
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
COLLECTION_NAME = "scholar_guard_docs"


# ------------------------------------------------------------------ #
#  Embeddings (HuggingFace — runs locally, no API key needed)
# ------------------------------------------------------------------ #

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a HuggingFace embedding model that runs locally.

    Uses 'all-MiniLM-L6-v2' which provides 384-dimensional vectors,
    runs fast on CPU, and works well for academic text retrieval.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ------------------------------------------------------------------ #
#  Vector Store Management
# ------------------------------------------------------------------ #

def initialize_vector_store() -> Chroma:
    """Create or load the persistent ChromaDB vector store.

    Returns:
        Chroma instance backed by persistent storage in data/vector_store/.
    """
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=_get_embeddings(),
        persist_directory=os.path.abspath(VECTOR_STORE_DIR),
    )
    return vector_store


def get_retriever(k: int = 5) -> Optional[object]:
    """Return a LangChain retriever over the vector store.

    Args:
        k: Number of top documents to retrieve.

    Returns:
        A LangChain retriever, or None if the vector store is empty.
    """
    vector_store = initialize_vector_store()

    # Check if there are any documents in the store
    try:
        collection = vector_store._collection
        if collection.count() == 0:
            return None
    except Exception:
        return None

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )


# ------------------------------------------------------------------ #
#  PDF Ingestion Pipeline
# ------------------------------------------------------------------ #

def ingest_pdfs(directory: Optional[str] = None) -> dict:
    """Load PDFs from a directory, chunk them, and add to the vector store.

    Args:
        directory: Path to folder containing PDF files.
                   Defaults to data/raw/.

    Returns:
        dict with keys 'files_processed' (int) and 'chunks_added' (int).
    """
    if directory is None:
        directory = RAW_DATA_DIR

    os.makedirs(directory, exist_ok=True)

    # Discover PDF files
    pdf_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        return {"files_processed": 0, "chunks_added": 0}

    # Load all PDFs
    all_documents = []
    for pdf_path in pdf_files:
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            print(f"⚠  Failed to load {pdf_path}: {e}")

    if not all_documents:
        return {"files_processed": len(pdf_files), "chunks_added": 0}

    # Chunk text for optimal retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(all_documents)

    # Embed and store
    vector_store = initialize_vector_store()
    vector_store.add_documents(chunks)

    return {
        "files_processed": len(pdf_files),
        "chunks_added": len(chunks),
    }


def get_collection_stats() -> dict:
    """Return basic statistics about the current vector store.

    Returns:
        dict with 'document_count' and 'status'.
    """
    try:
        vector_store = initialize_vector_store()
        count = vector_store._collection.count()
        return {
            "document_count": count,
            "status": "active" if count > 0 else "empty",
        }
    except Exception as e:
        return {
            "document_count": 0,
            "status": f"error: {e}",
        }
