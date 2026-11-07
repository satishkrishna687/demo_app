"""Generate a Markdown comparison report for the evaluated pipelines."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

_CUSTOM_COLUMNS = [
    "pipeline",
    "num_queries",
    "hit_rate",
    "citation_accuracy",
    "avg_retrieval_latency_ms",
    "avg_e2e_latency_ms",
    "avg_chunks_retrieved",
]


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def generate_comparison_report(results: list[dict], output_path: str = "eval/report.md") -> str:
    """Write a Markdown comparison report and return the output path.

    Args:
        results: One entry per pipeline, each with ``rows`` and ``aggregate`` keys.
        output_path: Destination Markdown file.
    """
    aggregates = pd.DataFrame([r["aggregate"] for r in results])

    # Per-category hit rate per pipeline.
    category_frames = {}
    for r in results:
        df = pd.DataFrame(r["rows"])
        if df.empty:
            continue
        df["retrieval_hit"] = df["retrieval_hit"].astype("boolean")
        category_frames[r["aggregate"]["pipeline"]] = (
            df.groupby("category")["retrieval_hit"].mean().astype(float)
        )
    category_df = pd.DataFrame(category_frames).round(2) if category_frames else pd.DataFrame()

    lines: list[str] = []
    lines.append("# Chunking + Retrieval Strategy Comparison Report\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")
    lines.append("## Executive Summary\n")
    lines.append(_summary(results))
    lines.append("\n## Aggregate Metrics (custom)\n")
    lines.append(_custom_table(aggregates))
    lines.append("\n## RAGAS Metrics\n")
    lines.append(_ragas_table(aggregates))
    lines.append("\n## Hit Rate by Query Category\n")
    lines.append(category_df.to_markdown() if not category_df.empty else "_No expected sections provided._")
    lines.append("\n## Per-Pipeline Details\n")
    for r in results:
        lines.append(_pipeline_details(r))
    lines.append("\n## Recommendations\n")
    lines.append(_recommendations(aggregates))
    lines.append("\n## Reproducibility Appendix\n")
    lines.append(_reproducibility())

    content = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return output_path


def _summary(results: list[dict]) -> str:
    if not results:
        return "_No results._"
    best_hit = max(
        results,
        key=lambda r: r["aggregate"]["hit_rate"] if r["aggregate"]["hit_rate"] is not None else -1,
    )
    fastest = min(
        results,
        key=lambda r: r["aggregate"]["avg_e2e_latency_ms"]
        if r["aggregate"]["avg_e2e_latency_ms"] is not None
        else float("inf"),
    )
    ragas_ok = any(
        any(isinstance(v, float) and v == v for v in (r["aggregate"].get("ragas") or {}).values())
        for r in results
    )
    parts = [
        f"Evaluated **{len(results)} pipelines** over "
        f"**{results[0]['aggregate']['num_queries']} queries**.",
        f"Best retrieval hit rate: **{best_hit['aggregate']['pipeline']}** "
        f"({_fmt(best_hit['aggregate']['hit_rate'])}).",
        f"Fastest end-to-end: **{fastest['aggregate']['pipeline']}** "
        f"({_fmt(fastest['aggregate']['avg_e2e_latency_ms'])} ms).",
        "RAGAS metrics computed." if ragas_ok else "RAGAS metrics unavailable (see appendix).",
    ]
    return " ".join(parts)


def _custom_table(aggregates: pd.DataFrame) -> str:
    cols = [c for c in _CUSTOM_COLUMNS if c in aggregates.columns]
    return aggregates[cols].round(4).to_markdown(index=False)


def _ragas_table(aggregates: pd.DataFrame) -> str:
    rows = []
    has_any = False
    for _, row in aggregates.iterrows():
        entry = {"pipeline": row["pipeline"]}
        ragas = row.get("ragas") or {}
        if ragas:
            entry.update(ragas)
            has_any = True
        if row.get("ragas_error"):
            entry["error"] = str(row["ragas_error"])[:120]
            has_any = True
        rows.append(entry)
    if not has_any:
        return (
            "_RAGAS metrics not computed (skipped or unavailable). See "
            "`docs/MILESTONE_7.md` for the local-LLM structured-output limitation._"
        )
    return pd.DataFrame(rows).to_markdown(index=False)


def _pipeline_details(result: dict) -> str:
    agg = result["aggregate"]
    lines = [
        f"\n### {agg['pipeline']}\n",
        "| Metric | Value |",
        "|---|---|",
        f"| Queries | {agg['num_queries']} |",
        f"| Hit Rate | {_fmt(agg['hit_rate'])} |",
        f"| Citation Accuracy | {_fmt(agg['citation_accuracy'])} |",
        f"| Avg Retrieval Latency | {_fmt(agg['avg_retrieval_latency_ms'])} ms |",
        f"| Avg End-to-End Latency | {_fmt(agg['avg_e2e_latency_ms'])} ms |",
        f"| Avg Chunks Retrieved | {_fmt(agg['avg_chunks_retrieved'])} |",
    ]
    for metric, value in (agg.get("ragas") or {}).items():
        lines.append(f"| RAGAS {metric} | {_fmt(value)} |")
    if agg.get("ragas_error"):
        lines.append(f"| RAGAS error | {str(agg['ragas_error'])[:120]} |")
    return "\n".join(lines) + "\n"


def _recommendations(aggregates: pd.DataFrame) -> str:
    workable = aggregates[aggregates["hit_rate"].notna()] if "hit_rate" in aggregates else aggregates
    bullets = []
    if not workable.empty:
        best_hit = workable.loc[workable["hit_rate"].idxmax(), "pipeline"]
        bullets.append(f"- **Best retrieval hit rate**: {best_hit}")
    lat = aggregates[aggregates["avg_e2e_latency_ms"].notna()]
    if not lat.empty:
        fastest = lat.loc[lat["avg_e2e_latency_ms"].idxmin(), "pipeline"]
        bullets.append(f"- **Fastest pipeline**: {fastest}")
    bullets.append(
        "- **Guidance**: use Pairing 3 for accuracy-critical questions (tables + graph); "
        "Pairing 2 for cost-effective structured queries; Pairing 1 for latency-sensitive use."
    )
    return "\n".join(bullets)


def _reproducibility() -> str:
    import importlib.metadata as md

    packages = [
        "langchain",
        "langchain-classic",
        "langchain-community",
        "langchain-chroma",
        "langchain-huggingface",
        "chromadb",
        "sentence-transformers",
        "transformers",
        "ragas",
        "torch",
        "docling",
    ]
    lines = ["| Package | Version |", "|---|---|"]
    for pkg in packages:
        try:
            lines.append(f"| {pkg} | {md.version(pkg)} |")
        except Exception:
            lines.append(f"| {pkg} | _not installed_ |")

    try:
        from config import EMBEDDING_MODEL, LLM_MODEL, LLM_PROVIDER, RERANKER_MODEL

        lines.append(f"\n- LLM provider: `{LLM_PROVIDER}` / model: `{LLM_MODEL}`")
        lines.append(f"- Embedding model: `{EMBEDDING_MODEL}`")
        lines.append(f"- Reranker model: `{RERANKER_MODEL}`")
    except Exception:
        pass
    return "\n".join(lines) + "\n"