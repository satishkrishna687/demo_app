"""Evaluation & Comparison dashboard tab.

Reads the latest per-pairing results and renders charts/tables. Data generation commands
are shown when results are missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from ui import charts, eval_data

_GEN_COMMANDS = """```bash
python -m eval.run_pairing --pairing 1 --limit 3
python -m eval.run_pairing --pairing 2 --limit 3
python -m eval.run_pairing --pairing 3 --limit 3
python -m eval.comparison_report
```"""


def render_dashboard() -> None:
    """Render the Evaluation & Comparison dashboard (all four metric categories)."""
    st.subheader("📊 Evaluation & Comparison")
    st.caption(
        "Snapshot of the latest per-pairing evaluations. Methodology: "
        "`RAG_EVAL_METRICS.md`; raw data: `eval/result_pair*.json`."
    )

    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    results = eval_data.load_results()
    missing = eval_data.missing_result_numbers()

    if not results:
        st.info("No evaluation results found. Generate them first:")
        st.markdown(_GEN_COMMANDS)
        return
    if missing:
        st.warning(
            f"Missing results for pairing(s) {missing}. Generate with "
            "`python -m eval.run_pairing --pairing N`."
        )

    summary = eval_data.summary_df(results)

    st.markdown("### Summary")
    st.dataframe(summary, hide_index=True, width="stretch")

    st.markdown("### Retrieval quality")
    ret_df = eval_data.tidy_metric_df(results, eval_data.RETRIEVAL_METRICS).dropna(subset=["Value"])
    if not ret_df.empty:
        st.altair_chart(charts.grouped_bar(ret_df, "Retrieval metrics (answer-term)"), width="stretch")
    else:
        st.caption("Retrieval metrics unavailable.")

    st.markdown("### Generation quality")
    gen_df = eval_data.tidy_metric_df(results, eval_data.GENERATION_METRICS).dropna(subset=["Value"])
    if not gen_df.empty:
        st.altair_chart(
            charts.grouped_bar(gen_df, "Generation metrics (LLM judge + rule-based citation)"),
            width="stretch",
        )
    else:
        st.caption("Generation metrics unavailable.")

    st.markdown("### Quality vs latency")
    bubble_df = summary.dropna(subset=["E2E (ms)", "Citation accuracy"])
    if not bubble_df.empty:
        st.altair_chart(charts.bubble_quality_vs_latency(bubble_df), width="stretch")
    else:
        st.caption("Not enough data to plot quality vs latency.")

    st.markdown("### Recall@5 by query category")
    category_df = eval_data.category_recall_df(results, k=5)
    if not category_df.empty:
        st.altair_chart(charts.heatmap_recall(category_df), width="stretch")
    else:
        st.caption("Per-category data unavailable.")

    st.markdown("### Chunking characteristics")
    c_left, c_right = st.columns(2)
    with c_left:
        chunk_counts = summary.dropna(subset=["Chunks"])[["Pair", "Chunks"]]
        if not chunk_counts.empty:
            st.altair_chart(
                charts.simple_bar(chunk_counts, "Pair", "Chunks", "Chunks per pairing"),
                width="stretch",
            )
    with c_right:
        chunk_sizes = summary.dropna(subset=["Avg chunk (chars)"])[["Pair", "Avg chunk (chars)"]]
        if not chunk_sizes.empty:
            st.altair_chart(
                charts.simple_bar(chunk_sizes, "Pair", "Avg chunk (chars)", "Avg chunk size (chars)"),
                width="stretch",
            )

    st.markdown("### Efficiency")
    eff_cols = ["Pair", "Indexing (s)", "Retrieval (ms)", "E2E (ms)", "Index (MiB)", "Tokens/query"]
    st.dataframe(summary[[c for c in eff_cols if c in summary.columns]], hide_index=True, width="stretch")

    st.markdown("### Flagged / unavailable metrics")
    flags = eval_data.flags_df(results)
    if flags.empty:
        st.success("None — all requested metrics were computed.")
    else:
        st.dataframe(flags, hide_index=True, width="stretch")

    report = eval_data.report_text()
    if report:
        with st.expander("📄 Full comparison report (RAG_EVAL_REPORT.md)"):
            st.markdown(report)
    else:
        st.caption("Run `python -m eval.comparison_report` to generate the full report.")