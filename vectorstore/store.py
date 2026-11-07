"""Shared vector store utilities.

Chroma rejects a single upsert larger than its max batch size (~5461), so all
ingestion goes through :func:`add_documents_in_batches`.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.documents import Document

from config import VECTORSTORE_BATCH_SIZE


def _chroma_safe_metadata(metadata: dict) -> dict:
    """Drop metadata values Chroma rejects (``None`` and empty lists).

    Chroma only accepts scalar values or *non-empty* lists. We keep the original
    ``Document`` metadata intact for in-memory use (e.g. the graph builder) and only
    sanitise the copies sent to the vector store.
    """
    safe: dict = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                continue
            safe[key] = list(value)
        else:
            safe[key] = value
    return safe


def ensure_indexed(
    vectorstore: Any,
    documents: Sequence[Document],
    ids: Sequence[str] | None = None,
    batch_size: int = VECTORSTORE_BATCH_SIZE,
) -> bool:
    """Index ``documents`` only if the collection does not already hold them.

    Uses the collection count as a cheap freshness check so repeated pipeline builds
    (e.g. after a Streamlit cache clear or process restart) do **not** re-embed the whole
    corpus. Returns ``True`` if documents were added, ``False`` if the existing index was
    reused.

    Note: this is a count-based heuristic. Delete the persist directory to force a rebuild
    after changing chunking or embedding settings.
    """
    try:
        existing = vectorstore._collection.count()
    except Exception:
        existing = None
    if existing == len(documents):
        return False
    add_documents_in_batches(vectorstore, documents, ids=ids, batch_size=batch_size)
    return True


def add_documents_in_batches(
    vectorstore: Any,
    documents: Sequence[Document],
    ids: Sequence[str] | None = None,
    batch_size: int = VECTORSTORE_BATCH_SIZE,
) -> None:
    """Add documents to a vector store in batches to respect backend limits.

    Args:
        vectorstore: A LangChain vector store exposing ``add_documents``.
        documents: Documents to index.
        ids: Optional ids aligned with ``documents`` (enables idempotent upserts).
        batch_size: Maximum documents per upsert.

    Raises:
        ValueError: If ``ids`` is provided but its length differs from ``documents``.
    """
    if ids is not None and len(ids) != len(documents):
        raise ValueError("ids length must match documents length")

    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        batch = [
            Document(page_content=d.page_content, metadata=_chroma_safe_metadata(d.metadata))
            for d in documents[start:end]
        ]
        batch_ids = list(ids[start:end]) if ids is not None else None
        vectorstore.add_documents(batch, ids=batch_ids)
