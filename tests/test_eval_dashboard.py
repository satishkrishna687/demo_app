"""Tests for the evaluation dashboard data shaping and chart builders (no model loads)."""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def test_summary_and_tidy_frames():
    from ui import eval_data

    results = eval_data.load_results()
    if not results:
        return  # no evaluation artifacts in a clean checkout -> nothing to assert

    summary = eval_data.summary_df(results)
    assert not summary.empty
    assert "Pair" in summary.columns

    retrieval = eval_data.tidy_metric_df(results, eval_data.RETRIEVAL_METRICS)
    assert set(retrieval.columns) == {"Pair", "Metric", "Value"}

    generation = eval_data.tidy_metric_df(results, eval_data.GENERATION_METRICS)
    assert set(generation.columns) == {"Pair", "Metric", "Value"}


def test_category_frame_is_named_and_chartable():
    from ui import charts, eval_data

    results = eval_data.load_results()
    if not results:
        return

    category = eval_data.category_recall_df(results)
    if category.empty:
        return
    # Regression: the index must be named "category" so heatmap_recall can melt it.
    assert category.index.name == "category"
    charts.heatmap_recall(category)  # must not raise


def test_bar_and_bubble_charts_build():
    from ui import charts, eval_data

    results = eval_data.load_results()
    if not results:
        return

    summary = eval_data.summary_df(results)
    retrieval = eval_data.tidy_metric_df(results, eval_data.RETRIEVAL_METRICS).dropna(subset=["Value"])
    if not retrieval.empty:
        charts.grouped_bar(retrieval, "test")

    bubble = summary.dropna(subset=["E2E (ms)", "Citation accuracy"])
    if not bubble.empty:
        charts.bubble_quality_vs_latency(bubble)