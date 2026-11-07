"""Deterministic retrieval and chunking metrics (no model calls).

Relevance is **answer-term based**: a retrieved chunk is relevant to a query if its text
contains any of the query's ``answer_terms`` (case-insensitive). This works uniformly across
all three pairings, because Pairing 1 chunks carry no section metadata. An additional
section-based relevance signal is computed when both the query gold sections and the chunk
section metadata are available.

Implements (per RAG_EVAL_METRICS.md §1–§2): Recall@k, Precision@k, MRR, nDCG@k, chunk
statistics, and index size.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Sequence


def _terms(entry: dict) -> list[str]:
    terms = list(entry.get("answer_terms") or [])
    gold = entry.get("gold_answer") or ""
    if gold:
        terms.append(gold)
    return [t.lower() for t in terms if t]


def relevance_flags(docs: Sequence[Any], entry: dict) -> list[bool]:
    """Binary relevance per retrieved doc, using answer-term matching."""
    terms = _terms(entry)
    if not terms:
        return [False] * len(docs)
    flags = []
    for doc in docs:
        text = (getattr(doc, "page_content", "") or "").lower()
        flags.append(any(t in text for t in terms))
    return flags


def section_flags(docs: Sequence[Any], entry: dict) -> list[bool]:
    """Section-based relevance: chunk ``document_id:section_number`` in gold sections."""
    gold = set(entry.get("gold_relevant_sections") or [])
    if not gold:
        return [False] * len(docs)
    flags = []
    for doc in docs:
        meta = getattr(doc, "metadata", {}) or {}
        key = f"{meta.get('document_id')}:{meta.get('section_number')}"
        flags.append(key in gold)
    return flags


def recall_at_k(flags: Sequence[bool], k: int) -> float:
    """1.0 if any relevant doc is in the top-k, else 0.0."""
    return 1.0 if any(flags[:k]) else 0.0


def precision_at_k(flags: Sequence[bool], k: int) -> float:
    """Fraction of the top-k docs that are relevant."""
    if k <= 0:
        return 0.0
    return sum(1 for f in flags[:k] if f) / k


def reciprocal_rank(flags: Sequence[bool]) -> float:
    """1 / rank of the first relevant doc (rank is 1-based); 0 if none."""
    for i, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / i
    return 0.0


def ndcg_at_k(flags: Sequence[bool], k: int) -> float:
    """Binary-relevance nDCG@k."""
    gains = [1.0 if f else 0.0 for f in flags[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_relevant = int(sum(gains))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))
    return dcg / idcg if idcg > 0 else 0.0


def mean(values: Iterable[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def aggregate_retrieval(per_query_flags: list[list[bool]], k_list: Sequence[int] = (1, 3, 5, 10)) -> dict:
    """Aggregate retrieval metrics across queries for each k."""
    out: dict[str, Any] = {"num_queries": len(per_query_flags)}
    for k in k_list:
        out[f"recall@{k}"] = mean(recall_at_k(f, k) for f in per_query_flags)
        out[f"precision@{k}"] = mean(precision_at_k(f, k) for f in per_query_flags)
        out[f"ndcg@{k}"] = mean(ndcg_at_k(f, k) for f in per_query_flags)
    out["mrr"] = mean(reciprocal_rank(f) for f in per_query_flags)
    return out


def chunk_statistics(chunks: Sequence[Any]) -> dict:
    """Chunk count and size statistics for a pairing's corpus."""
    sizes = [len(getattr(c, "page_content", "") or "") for c in chunks]
    if not sizes:
        return {"chunk_count": 0, "avg_chunk_size_chars": None, "min_chunk_size_chars": None,
                "max_chunk_size_chars": None}
    return {
        "chunk_count": len(sizes),
        "avg_chunk_size_chars": sum(sizes) / len(sizes),
        "min_chunk_size_chars": min(sizes),
        "max_chunk_size_chars": max(sizes),
    }


def dir_size_bytes(path: str) -> int | None:
    """Recursive size of a directory in bytes (index storage size)."""
    if not path or not os.path.isdir(path):
        return None
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total
