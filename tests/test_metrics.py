"""Deterministic unit tests for eval.metrics (no model loads)."""

from __future__ import annotations

import math

from langchain_core.documents import Document

from eval import metrics


def _doc(text: str, section: str = "", document_id: str = "ISO-15118-2") -> Document:
    return Document(page_content=text, metadata={"section_number": section, "document_id": document_id})


def test_relevance_flags_term_matching():
    entry = {"answer_terms": ["SLAC"]}
    docs = [_doc("no match"), _doc("the SLAC procedure"), _doc("slac lowercase")]
    assert metrics.relevance_flags(docs, entry) == [False, True, True]


def test_relevance_flags_includes_gold_answer():
    entry = {"answer_terms": [], "gold_answer": "230 V"}
    docs = [_doc("nominal is 230 V"), _doc("unrelated")]
    assert metrics.relevance_flags(docs, entry) == [True, False]


def test_section_flags():
    entry = {"gold_relevant_sections": ["ISO-15118-2:4.1"]}
    docs = [_doc("x", "4.1"), _doc("y", "4.2"), _doc("z", "4.1", "ISO-15118-3")]
    assert metrics.section_flags(docs, entry) == [True, False, False]


def test_recall_and_precision():
    flags = [False, True, True, False]
    assert metrics.recall_at_k(flags, 1) == 0.0
    assert metrics.recall_at_k(flags, 2) == 1.0
    assert metrics.precision_at_k(flags, 4) == 0.5
    assert metrics.precision_at_k(flags, 2) == 0.5


def test_reciprocal_rank():
    assert metrics.reciprocal_rank([False, False, True]) == 1 / 3
    assert metrics.reciprocal_rank([True, False]) == 1.0
    assert metrics.reciprocal_rank([False, False]) == 0.0


def test_ndcg_at_k():
    # Perfect ranking of 2 relevant at top -> 1.0
    assert metrics.ndcg_at_k([True, True, False], 3) == 1.0
    # No relevant -> 0.0
    assert metrics.ndcg_at_k([False, False], 2) == 0.0
    # One relevant at rank 2; the ideal (only 1 relevant) DCG is 1.0 at rank 1.
    got = metrics.ndcg_at_k([False, True, False], 3)
    expected = (1.0 / math.log2(3)) / 1.0
    assert abs(got - expected) < 1e-9


def test_aggregate_retrieval():
    per_query = [[True, False], [False, True]]
    agg = metrics.aggregate_retrieval(per_query, k_list=(1, 2))
    assert agg["num_queries"] == 2
    # Query 1: relevant at rank 1. Query 2: relevant at rank 2.
    assert agg["recall@1"] == 0.5
    assert agg["recall@2"] == 1.0
    assert agg["precision@2"] == 0.5
    assert agg["mrr"] == 0.75


def test_chunk_statistics():
    chunks = [_doc("12345"), _doc("1234567890")]
    stats = metrics.chunk_statistics(chunks)
    assert stats["chunk_count"] == 2
    assert stats["avg_chunk_size_chars"] == 7.5
    assert stats["min_chunk_size_chars"] == 5
    assert stats["max_chunk_size_chars"] == 10


def test_chunk_statistics_empty():
    stats = metrics.chunk_statistics([])
    assert stats["chunk_count"] == 0
    assert stats["avg_chunk_size_chars"] is None


def test_mean_ignores_none():
    assert metrics.mean([1.0, None, 3.0]) == 2.0
    assert metrics.mean([None, None]) is None
