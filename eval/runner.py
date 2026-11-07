"""Run one pairing over the gold set and compute all RAG_EVAL_METRICS categories.

Design notes:
- Retrieval metrics are deterministic (answer-term relevance).
- Generation/chunking metrics use the LLM judge (see :mod:`eval.judge`); unavailable scores are
  recorded as ``None`` and surfaced in ``flags`` rather than dropped.
- Efficiency metrics are measured around retrieval/generation and supplied by the caller for
  indexing time and index size (which are properties of the build step).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from eval import metrics
from eval.judge import LLMJudge, citation_accuracy

logger = logging.getLogger(__name__)

K_LIST = (1, 3, 5, 10)
JUDGE_CHUNKS = 2  # bound judge cost: score density/coherence on the top 2 chunks


def _call_retriever(retriever: Any, query: str) -> list:
    invoke = getattr(retriever, "invoke", None)
    return invoke(query) if callable(invoke) else retriever(query)


def _invoke_chain(chain: Any, query: str) -> tuple[str, dict | None]:
    """Invoke the chain, capturing token usage when the callback API is available."""
    try:
        from langchain_core.callbacks import UsageMetadataCallbackHandler

        handler = UsageMetadataCallbackHandler()
        answer = chain.invoke(query, config={"callbacks": [handler]})
        return answer, getattr(handler, "usage_metadata", None)
    except Exception:
        invoke = getattr(chain, "invoke", None)
        return (invoke(query) if callable(invoke) else chain(query)), None


def _sum_tokens(usage: dict | None) -> int | None:
    if not usage:
        return None
    total = 0
    found = False
    for model_usage in usage.values():
        if isinstance(model_usage, dict):
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                if isinstance(model_usage.get(key), int):
                    total += model_usage[key]
                    found = True
    return total if found else None


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _std(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    mu = sum(vals) / len(vals)
    return float((sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5)


def evaluate_pairing(
    pairing_name: str,
    pipeline: dict,
    gold_set: list[dict],
    *,
    corpus_chunks: Sequence[Any] | None = None,
    indexing_time_ms: float | None = None,
    index_size_bytes: int | None = None,
    k_list: Sequence[int] = K_LIST,
    runs: int = 1,
    use_judge: bool = True,
    judge: LLMJudge | None = None,
    max_judge_queries: int | None = None,
) -> dict:
    """Evaluate a single pairing and return a structured metrics dict."""
    retriever = pipeline["retriever"]
    chain = pipeline["chain"]
    judge = judge or (LLMJudge() if use_judge else None)

    flags: list[dict] = []
    per_query: list[dict] = []
    section_metadata_seen = False
    term_flags_per_query: list[list[bool]] = []
    section_flags_per_query: list[list[bool]] = []

    retrieval_ms_all: list[float] = []
    generation_ms_all: list[float] = []
    e2e_ms_all: list[float] = []
    tokens_all: list[int | None] = []

    judge_used = 0
    judge_total = 0

    for i, entry in enumerate(gold_set):
        query = entry["query"]

        t0 = time.perf_counter()
        try:
            docs = _call_retriever(retriever, query) or []
        except Exception as exc:
            logger.warning("[%s] retrieval failed for %s: %s", pairing_name, entry["id"], exc)
            docs = []
        retrieval_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        try:
            answer, usage = _invoke_chain(chain, query)
        except Exception as exc:
            logger.warning("[%s] generation failed for %s: %s", pairing_name, entry["id"], exc)
            answer, usage = "", None
        e2e_ms = (time.perf_counter() - t0) * 1000
        generation_ms = max(0.0, e2e_ms - retrieval_ms)

        term_flags = metrics.relevance_flags(docs, entry)
        sec_flags = metrics.section_flags(docs, entry)
        if not section_metadata_seen and any((d.metadata or {}).get("section_number") for d in docs):
            section_metadata_seen = True
        term_flags_per_query.append(term_flags)
        section_flags_per_query.append(sec_flags)

        retrieved_sections = [d.metadata.get("section_number", "") for d in docs]
        cited_acc = citation_accuracy(answer, retrieved_sections)

        row: dict[str, Any] = {
            "query_id": entry["id"],
            "query": query,
            "category": entry.get("category"),
            "answer": answer,
            "num_retrieved": len(docs),
            "relevance_flags": term_flags,
            "section_flags": sec_flags,
            "retrieved_sections": retrieved_sections,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "e2e_ms": e2e_ms,
            "tokens": _sum_tokens(usage),
            "citation_accuracy": cited_acc,
        }

        # Judge metrics (optionally limited to save time).
        if judge is not None and (max_judge_queries is None or i < max_judge_queries):
            contexts = [d.page_content for d in docs]
            row["relevance_density"] = _mean(
                [judge.relevance_density(query, c) for c in contexts[:JUDGE_CHUNKS]]
            )
            row["boundary_coherence"] = _mean(
                [judge.boundary_coherence(c) for c in contexts[:JUDGE_CHUNKS]]
            )
            row["faithfulness"] = judge.faithfulness(query, answer, contexts)
            row["answer_relevance"] = judge.answer_relevance(query, answer)
            row["answer_correctness"] = judge.answer_correctness(answer, entry.get("gold_answer", ""))
            judge_used += 1
            judge_total += 5

        retrieval_ms_all.append(retrieval_ms)
        generation_ms_all.append(generation_ms)
        e2e_ms_all.append(e2e_ms)
        tokens_all.append(_sum_tokens(usage))
        per_query.append(row)

    retrieval = metrics.aggregate_retrieval(term_flags_per_query, k_list)

    if section_metadata_seen:
        retrieval_section = metrics.aggregate_retrieval(section_flags_per_query, k_list)
    else:
        # No section metadata in this pairing's chunks (e.g. Pairing 1 recursive chunks):
        # the section-based metric is not applicable, not zero.
        retrieval_section = {
            **{f"recall@{k}": None for k in k_list},
            **{f"precision@{k}": None for k in k_list},
            **{f"ndcg@{k}": None for k in k_list},
            "mrr": None,
            "num_queries": len(per_query),
        }
        flags.append({
            "metric": "retrieval_section",
            "reason": "not applicable: retrieved chunks carry no section_number (recursive chunking)",
        })

    chunking: dict[str, Any] = {}
    if corpus_chunks is not None:
        chunking.update(metrics.chunk_statistics(corpus_chunks))
    chunking["relevance_density"] = _mean([r.get("relevance_density") for r in per_query])
    chunking["boundary_coherence"] = _mean([r.get("boundary_coherence") for r in per_query])

    generation = {
        "faithfulness": _mean([r.get("faithfulness") for r in per_query]),
        "answer_relevance": _mean([r.get("answer_relevance") for r in per_query]),
        "answer_correctness": _mean([r.get("answer_correctness") for r in per_query]),
        "citation_accuracy": _mean([r.get("citation_accuracy") for r in per_query]),
    }

    efficiency = {
        "indexing_time_ms": indexing_time_ms,
        "avg_retrieval_ms": _mean(retrieval_ms_all),
        "avg_generation_ms": _mean(generation_ms_all),
        "avg_e2e_ms": _mean(e2e_ms_all),
        "index_size_bytes": index_size_bytes,
        "llm_calls_per_query": 1.0,
        "tokens_per_query": _mean(tokens_all) if any(t is not None for t in tokens_all) else None,
    }

    # Flag metrics that could not be computed, per RAG_EVAL_METRICS.md Rules.
    if all(chunking.get(k) is None for k in ("avg_chunk_size_chars",)):
        flags.append({"metric": "chunking.avg_chunk_size", "reason": "corpus_chunks not provided"})
    for name, value in generation.items():
        if value is None:
            flags.append({
                "metric": f"generation.{name}",
                "reason": "judge unavailable or not applicable (e.g. no cited sections / no gold answer)",
            })
    if efficiency["index_size_bytes"] is None:
        flags.append({"metric": "efficiency.index_size_bytes", "reason": "not supplied by caller"})
    if efficiency["tokens_per_query"] is None:
        flags.append({"metric": "efficiency.tokens_per_query", "reason": "usage metadata unavailable"})

    return {
        "pairing": pairing_name,
        "num_queries": len(gold_set),
        "retrieval": retrieval,
        "retrieval_section": retrieval_section,
        "chunking": chunking,
        "generation": generation,
        "efficiency": efficiency,
        "judge_coverage": {"queries_judged": judge_used, "metrics_attempted": judge_total},
        "flags": flags,
        "per_query": per_query,
    }
