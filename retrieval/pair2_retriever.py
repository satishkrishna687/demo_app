"""Pairing 2 retrieval: metadata-filtered hybrid search.

Combines dense vector search (Chroma) with sparse keyword search (BM25) via an
``EnsembleRetriever`` (Reciprocal Rank Fusion), and exposes a metadata-filtered query
path so callers can scope retrieval to a specific ISO clause number.
"""

from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import (
    CHROMA_PERSIST_DIR,
    HYBRID_BM25_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    TOP_K,
)
from models.embeddings import get_embeddings
from vectorstore.store import ensure_indexed


def build_metadata_filtered_hybrid_retriever(
    chunks: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = "pair2_section_hybrid",
    vector_weight: float = HYBRID_VECTOR_WEIGHT,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    k: int = TOP_K,
) -> tuple[EnsembleRetriever, Chroma]:
    """Build a hybrid (vector + BM25) retriever and return it with its vector store."""
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    ids = [f"{c.metadata.get('document_id', 'doc')}_s2_{i}" for i, c in enumerate(chunks)]
    ensure_indexed(vectorstore, chunks, ids=ids)

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    ensemble = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )
    return ensemble, vectorstore


def filtered_retrieve(
    vectorstore: Chroma,
    query: str,
    section_filter: str | None = None,
    k: int = TOP_K,
) -> list[Document]:
    """Run a metadata-filtered similarity search.

    Args:
        vectorstore: The Chroma store built by :func:`build_metadata_filtered_hybrid_retriever`.
        query: Natural-language query.
        section_filter: Optional exact ``section_number`` (e.g. ``"4.1"``).
        k: Number of results.

    Returns:
        Matching documents, filtered by ``section_number`` when provided.
    """
    if not section_filter:
        return vectorstore.similarity_search(query, k=k)

    try:
        return vectorstore.similarity_search(
            query, k=k, filter={"section_number": {"$eq": section_filter}}
        )
    except Exception:
        # Older/newer Chroma builds may prefer `where` over `filter`.
        return vectorstore.similarity_search(
            query, k=k, where={"section_number": section_filter}
        )