"""Pairing 2 pipeline builder for per-user indexes.

Builds a section-based hybrid (vector + BM25) retriever for a single uploaded
PDF, using OpenRouter embeddings via the existing model factories. Each user
gets their own Chroma persist directory under ``data/users/{user_id}/index/``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config import CHROMA_PERSIST_DIR, HYBRID_BM25_WEIGHT, HYBRID_VECTOR_WEIGHT, TOP_K
from ingestion.chunking_pair2 import create_section_based_chunks
from ingestion.loader import load_pdf_structured
from models.embeddings import get_embeddings
from retrieval.pair2_retriever import build_metadata_filtered_hybrid_retriever
from vectorstore.store import ensure_indexed
from webapp.config import USERS_DIR


def user_index_dir(user_id: int) -> Path:
    """Return the per-user Chroma persist directory."""
    return USERS_DIR / str(user_id) / "index"


def user_pdf_dir(user_id: int) -> Path:
    """Return the per-user PDF storage directory."""
    return USERS_DIR / str(user_id) / "pdfs"


def build_pairing2_pipeline(
    user_id: int,
    pdf_path: str | Path,
    document_id: str | None = None,
) -> tuple[Any, Any, list[Document]]:
    """Build a Pairing 2 hybrid retriever for a single user's PDF.

    Args:
        user_id: The owning user (used for the per-user index directory).
        pdf_path: Path to the uploaded ISO PDF.
        document_id: Optional document id override (defaults to the PDF stem).

    Returns:
        ``(retriever, vectorstore, chunks)`` where ``retriever`` is the hybrid
        EnsembleRetriever, ``vectorstore`` is the Chroma store, and ``chunks``
        is the list of section-based chunks.
    """
    pdf_path = Path(pdf_path)
    if document_id is None:
        document_id = pdf_path.stem

    persist_dir = str(user_index_dir(user_id))
    collection_name = f"user_{user_id}_pair2"

    # 1. Parse the PDF into a structured document (Docling, section-aware).
    structured = load_pdf_structured(pdf_path, use_ocr=False)
    if structured is None:
        raise RuntimeError(f"Failed to parse PDF: {pdf_path}")

    # 2. Chunk into section-based units (tables kept atomic).
    chunks = create_section_based_chunks([structured])
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
        chunk.metadata["source_file"] = pdf_path.name
        chunk.metadata["user_id"] = user_id

    # 3. Build the hybrid retriever with OpenRouter embeddings.
    #    get_embeddings() reads EMBEDDING_PROVIDER=openai from webapp/config.py
    #    and routes to the OpenAI-compatible endpoint.
    retriever, vectorstore = build_metadata_filtered_hybrid_retriever(
        chunks,
        persist_directory=persist_dir,
        collection_name=collection_name,
        vector_weight=HYBRID_VECTOR_WEIGHT,
        bm25_weight=HYBRID_BM25_WEIGHT,
        k=TOP_K,
    )

    return retriever, vectorstore, chunks


def reset_user_index(user_id: int) -> None:
    """Delete a user's persisted index directory (for re-uploads)."""
    index_dir = user_index_dir(user_id)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
