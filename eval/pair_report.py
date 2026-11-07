"""Render a per-pairing evaluation report covering all four RAG_EVAL_METRICS categories."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from eval.metrics import recall_at_k


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _retrieval_table(retrieval: dict, section_retrieval: dict | None) -> str:
    rows = []
    ks = sorted({int(key.split("@")[1]) for key in retrieval if "@" in key})
    for k in ks:
        rows.append(
            {
                "k": k,
                "Recall@k": retrieval.get(f"recall@{k}"),
                "Precision@k": retrieval.get(f"precision@{k}"),
                "nDCG@k": retrieval.get(f"ndcg@{k}"),
                "Recall@k (section)": (section_retrieval or {}).get(f"recall@{k}"),
            }
        )
    table = pd.DataFrame(rows).to_markdown(index=False)
    return f"{table}\n\nMRR (answer-term): **{_fmt(retrieval.get('mrr'))}** | " \
           f"MRR (section): **{_fmt((section_retrieval or {}).get('mrr'))}**"


def _category_table(per_query: list[dict]) -> str:
    if not per_query:
        return "_No queries._"
    from collections import defaultdict

    buckets: dict[str, list[float]] = defaultdict(list)
    for row in per_query:
        buckets[row.get("category", "uncategorized")].append(recall_at_k(row["relevance_flags"], 5))
    data = [{"category": c, "Recall@5": sum(v) / len(v), "n": len(v)} for c, v in sorted(buckets.items())]
    return pd.DataFrame(data).round(4).to_markdown(index=False)


def generate_pair_report(
    result: dict,
    output_path: str,
    *,
    config_lines: dict[str, Any] | None = None,
) -> str:
    """Write a Markdown report for one pairing and return the output path."""
    lines: list[str] = []
    lines.append(f"# Evaluation Report — {result['pairing']}\n")
    lines.append(f"_Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}_\n")
    lines.append(f"Queries evaluated: **{result['num_queries']}**\n")

    if config_lines:
        lines.append("## Configuration\n")
        lines.append("| Setting | Value |")
        lines.append("|---|---|")
        for key, value in config_lines.items():
            lines.append(f"| {key} | {value} |")
        lines.append("")

    lines.append("## 1. Retrieval-Level Metrics\n")
    lines.append(_retrieval_table(result["retrieval"], result.get("retrieval_section")))
    lines.append("\n### Recall@5 by Query Category\n")
    lines.append(_category_table(result["per_query"]))

    lines.append("\n## 2. Chunking-Specific Metrics\n")
    lines.append(pd.DataFrame([result["chunking"]]).to_markdown(index=False))

    lines.append("\n## 3. End-to-End Generation Metrics\n")
    lines.append(pd.DataFrame([result["generation"]]).to_markdown(index=False))

    lines.append("\n## 4. Efficiency / Cost Metrics\n")
    lines.append(pd.DataFrame([result["efficiency"]]).to_markdown(index=False))

    lines.append("\n## Flagged / Unavailable Metrics\n")
    flags = result.get("flags") or []
    if flags:
        lines.append(pd.DataFrame(flags).to_markdown(index=False))
    else:
        lines.append("_None — all metrics computed._")

    lines.append("\n## Judge Coverage\n")
    lines.append(pd.DataFrame([result.get("judge_coverage", {})]).to_markdown(index=False))

    lines.append("\n## Per-Query Appendix\n")
    rows = []
    for r in result["per_query"]:
        rows.append(
            {
                "id": r["query_id"],
                "category": r.get("category"),
                "recall@5": recall_at_k(r["relevance_flags"], 5),
                "mrr": next((1 / i for i, f in enumerate(r["relevance_flags"], 1) if f), 0.0),
                "citation_acc": r.get("citation_accuracy"),
                "answer": (r.get("answer") or "")[:80].replace("\n", " "),
            }
        )
    lines.append(pd.DataFrame(rows).round(3).to_markdown(index=False))

    content = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    return output_path
