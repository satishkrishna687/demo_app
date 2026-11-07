"""Load evaluation results and shape them for the dashboard.

Reads ``eval/result_pair{1,2,3}.json`` (produced by ``eval.run_pairing``) and
``eval/RAG_EVAL_REPORT.md``. Kept free of heavy model imports so the dashboard is fast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Any

import pandas as pd

RESULT_PATHS = {
    1: "eval/result_pair1.json",
    2: "eval/result_pair2.json",
    3: "eval/result_pair3.json",
}
REPORT_PATH = "eval/RAG_EVAL_REPORT.md"

SHORT_LABELS = {
    "Pairing 1: Recursive + Parent-Child": "P1 Recursive+Parent-Child",
    "Pairing 2: Section + Hybrid": "P2 Section+Hybrid",
    "Pairing 3: Advanced ISO + Graph": "P3 Advanced+Graph",
}

RETRIEVAL_METRICS = {"Recall@5": "recall@5", "nDCG@5": "ndcg@5", "MRR": "mrr"}
GENERATION_METRICS = {
    "Faithfulness": "faithfulness",
    "Answer relevance": "answer_relevance",
    "Answer correctness": "answer_correctness",
    "Citation accuracy": "citation_accuracy",
}


def load_results() -> list[dict]:
    """Load available result JSONs, adding a short ``label`` per result."""
    results: list[dict] = []
    for number, path in RESULT_PATHS.items():
        p = Path(path)
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        data["number"] = number
        data["label"] = SHORT_LABELS.get(data.get("pairing", ""), data.get("pairing", f"Pairing {number}"))
        results.append(data)
    return results


def missing_result_numbers() -> list[int]:
    return [n for n, p in RESULT_PATHS.items() if not Path(p).exists()]


def report_text() -> str | None:
    p = Path(REPORT_PATH)
    return p.read_text(encoding="utf-8") if p.exists() else None


def summary_df(results: list[dict]) -> pd.DataFrame:
    """One row per pairing with headline metrics across all four categories."""
    rows = []
    for r in results:
        ret = r.get("retrieval", {})
        sec = r.get("retrieval_section", {}) or {}
        ch = r.get("chunking", {})
        gen = r.get("generation", {})
        eff = r.get("efficiency", {})
        rows.append(
            {
                "Pair": r["label"],
                "Queries": r.get("num_queries"),
                "Recall@5": ret.get("recall@5"),
                "nDCG@5": ret.get("ndcg@5"),
                "MRR": ret.get("mrr"),
                "Section Recall@5": sec.get("recall@5"),
                "Faithfulness": gen.get("faithfulness"),
                "Answer correctness": gen.get("answer_correctness"),
                "Citation accuracy": gen.get("citation_accuracy"),
                "Chunks": ch.get("chunk_count"),
                "Avg chunk (chars)": ch.get("avg_chunk_size_chars"),
                "Rel. density": ch.get("relevance_density"),
                "Boundary coherence": ch.get("boundary_coherence"),
                "Indexing (s)": (eff.get("indexing_time_ms") or 0) / 1000 if eff.get("indexing_time_ms") else None,
                "Retrieval (ms)": eff.get("avg_retrieval_ms"),
                "E2E (ms)": eff.get("avg_e2e_ms"),
                "Index (MiB)": (eff.get("index_size_bytes") or 0) / (1024 * 1024) if eff.get("index_size_bytes") else None,
                "Tokens/query": eff.get("tokens_per_query"),
            }
        )
    return pd.DataFrame(rows)


def tidy_metric_df(results: list[dict], metrics: dict[str, str]) -> pd.DataFrame:
    """Long-format frame ``[Pair, Metric, Value]`` for grouped bar charts."""
    rows = []
    for r in results:
        for label, key in metrics.items():
            value = (r.get("retrieval") or {}).get(key) if key in r.get("retrieval", {}) else (r.get("generation") or {}).get(key)
            rows.append({"Pair": r["label"], "Metric": label, "Value": value})
    return pd.DataFrame(rows)


def category_recall_df(results: list[dict], k: int = 5) -> pd.DataFrame:
    """Per-category Recall@k for each pairing (from ``per_query`` relevance flags)."""
    frames = {}
    for r in results:
        buckets: dict[str, list[float]] = {}
        for row in r.get("per_query", []):
            flags = row.get("relevance_flags") or []
            hit = 1.0 if any(flags[:k]) else 0.0
            buckets.setdefault(row.get("category", "uncategorized"), []).append(hit)
        if buckets:
            frames[r["label"]] = {cat: sum(v) / len(v) for cat, v in buckets.items()}
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).fillna(0.0).round(3)
    df.index.name = "category"
    return df


def flags_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        for flag in r.get("flags", []):
            rows.append({"Pair": r["label"], "Metric": flag.get("metric"), "Reason": flag.get("reason")})
    return pd.DataFrame(rows)