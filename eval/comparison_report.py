"""Generate the cross-pair comparison report required by RAG_EVAL_METRICS.md.

Reads ``eval/result_pair{1,2,3}.json`` (produced by ``eval.run_pairing``) and writes
``eval/RAG_EVAL_REPORT.md`` with the required outputs:

1. Summary table (one row per pair, metrics from all four categories).
2. Retrieval-quality ranking (Recall@k / MRR / nDCG).
3. Generation-quality ranking (faithfulness / correctness).
4. Efficiency tradeoff note.
5. Overall recommendation.
Plus a variance note and an explicit list of flagged/uncomputable metrics.

Run:  python -m eval.comparison_report
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_INPUTS = ["eval/result_pair1.json", "eval/result_pair2.json", "eval/result_pair3.json"]
DEFAULT_OUTPUT = "eval/RAG_EVAL_REPORT.md"


def load_results(paths: list[str]) -> list[dict]:
    results = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            logger.warning("Missing result file: %s (skipped)", path)
            continue
        results.append(json.loads(p.read_text(encoding="utf-8")))
    return results


def _b(value: Any, scale: float = 1.0, unit: str = "") -> Any:
    if value is None:
        return None
    return round(value / scale, 3)


def summary_dataframe(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        ret = r.get("retrieval", {})
        sec = r.get("retrieval_section", {}) or {}
        ch = r.get("chunking", {})
        gen = r.get("generation", {})
        eff = r.get("efficiency", {})
        rows.append(
            {
                "Pair": r["pairing"],
                "Queries": r["num_queries"],
                # Retrieval
                "Recall@5": ret.get("recall@5"),
                "Precision@5": ret.get("precision@5"),
                "nDCG@5": ret.get("ndcg@5"),
                "MRR": ret.get("mrr"),
                "Recall@5 (section)": sec.get("recall@5"),
                # Chunking
                "Chunks": ch.get("chunk_count"),
                "Avg chunk (chars)": _b(ch.get("avg_chunk_size_chars")),
                "Rel. density": ch.get("relevance_density"),
                "Boundary coh.": ch.get("boundary_coherence"),
                # Generation
                "Faithfulness": gen.get("faithfulness"),
                "Ans. relevance": gen.get("answer_relevance"),
                "Ans. correctness": gen.get("answer_correctness"),
                "Citation acc.": gen.get("citation_accuracy"),
                # Efficiency
                "Index (s)": _b(eff.get("indexing_time_ms"), 1000),
                "Retrieval (ms)": _b(eff.get("avg_retrieval_ms")),
                "E2E (ms)": _b(eff.get("avg_e2e_ms")),
                "Index (MiB)": _b(eff.get("index_size_bytes"), 1024 * 1024),
                "Tokens/q": _b(eff.get("tokens_per_query")),
            }
        )
    return pd.DataFrame(rows)


def _display(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of the summary table with rounded numbers and NaN rendered as 'N/A'."""
    out = df.copy()
    for col in out.columns:
        if col == "Pair":
            continue
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: "N/A" if pd.isna(v) else round(float(v), 4))
        elif out[col].isna().any():
            out[col] = out[col].map(lambda v: "N/A" if pd.isna(v) else v)
    return out


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def _best(df: pd.DataFrame, column: str) -> tuple[str, Any] | None:
    if column not in df or df[column].dropna().empty:
        return None
    idx = df[column].idxmax()
    return df.loc[idx, "Pair"], df.loc[idx, column]


def _rankings(df: pd.DataFrame) -> str:
    lines = ["| Aspect | Winner | Value | Runner-up |", "|---|---|---|---|"]
    for label, col in [
        ("Recall@5 (answer-term)", "Recall@5"),
        ("MRR (answer-term)", "MRR"),
        ("nDCG@5 (answer-term)", "nDCG@5"),
        ("Recall@5 (section)", "Recall@5 (section)"),
        ("Faithfulness", "Faithfulness"),
        ("Answer correctness", "Ans. correctness"),
        ("Citation accuracy", "Citation acc."),
    ]:
        ranked = df[[ "Pair", col]].dropna().sort_values(col, ascending=False).reset_index(drop=True)
        if ranked.empty:
            lines.append(f"| {label} | _unavailable_ | n/a | n/a |")
            continue
        winner, wv = ranked.loc[0, "Pair"], ranked.loc[0, col]
        runner = ranked.loc[1, "Pair"] if len(ranked) > 1 else "n/a"
        rv = ranked.loc[1, col] if len(ranked) > 1 else "n/a"
        lines.append(f"| {label} | {winner} | {_fmt(wv)} | {runner} ({_fmt(rv)}) |")
    return "\n".join(lines)


def _efficiency_note(df: pd.DataFrame) -> str:
    q = df.dropna(subset=["E2E (ms)"])
    if q.empty:
        return "_Efficiency data unavailable._"
    fastest = q.loc[q["E2E (ms)"].idxmin()]
    best_cite = df.dropna(subset=["Citation acc."])
    if best_cite.empty:
        return (
            f"Fastest end-to-end: **{fastest['Pair']}** ({fastest['E2E (ms)']} ms). "
            "Quality winner unavailable."
        )
    quality = best_cite.loc[best_cite["Citation acc."].idxmax()]
    if quality["Pair"] == fastest["Pair"]:
        return (
            f"**{quality['Pair']}** is both the quality winner (citation accuracy "
            f"{quality['Citation acc.']}) and the fastest ({fastest['E2E (ms)']} ms) — no tradeoff."
        )
    ratio = (
        quality["E2E (ms)"] / fastest["E2E (ms)"]
        if fastest["E2E (ms)"] and fastest["E2E (ms)"] > 0
        else float("nan")
    )
    ret_note = ""
    ret = df.dropna(subset=["Retrieval (ms)"])
    if not ret.empty:
        fastest_ret = ret.loc[ret["Retrieval (ms)"].idxmin()]
        ret_note = (
            f" Retrieval alone is far more asymmetric: {_fmt(fastest_ret['Retrieval (ms)'])} ms for "
            f"{fastest_ret['Pair']} vs {_fmt(quality['Retrieval (ms)'])} ms for {quality['Pair']}."
        )
    return (
        f"**{quality['Pair']}** has the best citation accuracy ({_fmt(quality['Citation acc.'])}) but "
        f"is ~{ratio:.0f}× slower end-to-end than **{fastest['Pair']}** "
        f"({_fmt(quality['E2E (ms)'])} ms vs {_fmt(fastest['E2E (ms)'])} ms).{ret_note} "
        "Choose quality when accuracy matters, and the faster pairing for interactive use."
    )


def _recommendation(df: pd.DataFrame) -> str:
    quality = _best(df, "Citation acc.")
    correctness = _best(df, "Ans. correctness")
    fastest = _best(df, "E2E (ms)")  # max is wrong; fix below
    q = df.dropna(subset=["E2E (ms)"])
    fast_pair = q.loc[q["E2E (ms)"].idxmin(), "Pair"] if not q.empty else "n/a"
    quality_pair = quality[0] if quality else "n/a"
    correctness_pair = correctness[0] if correctness else "n/a"
    return (
        f"For accuracy-critical ISO queries, **{quality_pair}** is recommended (best citation "
        f"accuracy{citation_note(quality)}), with **{correctness_pair}** strongest on answer "
        f"correctness. For latency-sensitive or interactive use, **{fast_pair}** is the most "
        "responsive. Balance these against index size and per-query token cost in the summary table."
    )


def citation_note(quality: tuple[str, Any] | None) -> str:
    return f" {quality[1]}" if quality else ""


def generate_report(results: list[dict], output_path: str = DEFAULT_OUTPUT) -> str:
    df = summary_dataframe(results)
    lines: list[str] = []
    lines.append("# RAG Evaluation: Chunking × Retrieval Pair Comparison\n")
    lines.append("_Generated by `eval/comparison_report.py` per `RAG_EVAL_METRICS.md`._\n")

    lines.append("## 1. Summary Table\n")
    lines.append(_display(df).to_markdown(index=False))

    lines.append("\n## 2. Retrieval & Generation Rankings\n")
    lines.append(_rankings(df))

    lines.append("\n## 3. Efficiency Tradeoff Note\n")
    lines.append(_efficiency_note(df))

    lines.append("\n## 4. Overall Recommendation\n")
    lines.append(_recommendation(df))

    lines.append("\n## 5. Variance\n")
    lines.append(
        "_Variance not reported: each pairing was evaluated with a single run (`--runs 1`). "
        "Re-run `python -m eval.run_pairing --pairing N --runs 3` to enable variance (mean ± std)._"
    )

    lines.append("\n## 6. Flagged / Uncomputable Metrics\n")
    flag_rows = []
    for r in results:
        for flag in r.get("flags", []):
            flag_rows.append({"Pair": r["pairing"], "Metric": flag.get("metric"), "Reason": flag.get("reason")})
    if flag_rows:
        lines.append(pd.DataFrame(flag_rows).to_markdown(index=False))
    else:
        lines.append("_None — all requested metrics were computed for every pairing._")

    lines.append("\n## 7. Methodology & Reproducibility\n")
    lines.append(
        "- Same gold query set and scoring criteria across all pairs.\n"
        "- Single judge model, temperature 0, identical prompts (see `eval/judge.py`).\n"
        "- Relevance: answer-term matching (primary) + section matching (secondary).\n"
        "- Uncomputable metrics are flagged, never omitted."
    )

    content = "\n".join(lines) + "\n"
    Path(output_path).write_text(content, encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build the cross-pair RAG comparison report.")
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    results = load_results(args.inputs)
    if not results:
        logger.error("No result files found; run eval.run_pairing first.")
        return 1
    path = generate_report(results, args.output)
    logger.info("Wrote %s", path)
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
