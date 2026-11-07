"""Cross-encoder reranker factory and helper (Pairing 3, Stage 4).

A cross-encoder scores each (query, chunk) pair jointly, which is far more accurate than
the bi-encoder similarity used for first-stage retrieval. It is intentionally applied only
to the small candidate set produced by hybrid search + graph expansion.

Example:
    from models.reranker import rerank
    top = rerank(query, docs, top_k=5)
"""

from __future__ import annotations

from typing import Any, Sequence

from config import RERANKER_MAX_LENGTH, RERANKER_MODEL
from models.device import get_device

_CACHE: dict[tuple[str, str], Any] = {}


def get_reranker(model_name: str | None = None, device: str | None = None) -> Any:
    """Return a cached ``sentence_transformers.CrossEncoder`` instance."""
    from sentence_transformers import CrossEncoder

    name = model_name or RERANKER_MODEL
    dev = device or get_device()
    key = (name, dev)

    if key not in _CACHE:
        _CACHE[key] = CrossEncoder(name, max_length=RERANKER_MAX_LENGTH, device=dev)

    return _CACHE[key]


def rerank(query: str, docs: Sequence[Any], top_k: int) -> list[Any]:
    """Reorder documents by cross-encoder relevance to the query.

    Args:
        query: The user question.
        docs: LangChain ``Document`` objects to score.
        top_k: Number of top documents to return.

    Returns:
        The ``top_k`` documents ordered by descending relevance score.
    """
    if not docs:
        return []

    reranker = get_reranker()
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked[:top_k]


def clear_reranker_cache() -> None:
    """Drop cached reranker models (useful for tests)."""
    _CACHE.clear()
