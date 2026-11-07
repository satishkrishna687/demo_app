"""Altair chart builders for the evaluation dashboard."""

from __future__ import annotations

import altair as alt
import pandas as pd

_PALETTE = ["#4C78A8", "#F58518", "#54A24B"]


def grouped_bar(df: pd.DataFrame, title: str, y_title: str = "Score") -> alt.Chart:
    """Grouped bar chart of ``Value`` by ``Metric`` and ``Pair``."""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Metric:N", axis=alt.Axis(labelAngle=-20, title=None)),
            xOffset="Pair:N",
            y=alt.Y("Value:Q", axis=alt.Axis(title=y_title), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Pair:N", scale=alt.Scale(range=_PALETTE), title="Pairing"),
            tooltip=["Pair", "Metric", alt.Tooltip("Value:Q", format=".3f")],
        )
        .properties(title=title, height=360)
    )


def bubble_quality_vs_latency(summary: pd.DataFrame) -> alt.Chart:
    """Quality (citation accuracy) vs latency (E2E ms); bubble size = index size."""
    return (
        alt.Chart(summary)
        .mark_circle(opacity=0.7)
        .encode(
            x=alt.X("E2E (ms):Q", title="End-to-end latency (ms, lower is better)"),
            y=alt.Y("Citation accuracy:Q", title="Citation accuracy (higher is better)"),
            size=alt.Size("Index (MiB):Q", title="Index size (MiB)"),
            color=alt.Color("Pair:N", scale=alt.Scale(range=_PALETTE), title="Pairing"),
            tooltip=["Pair", "E2E (ms)", "Citation accuracy", "Index (MiB)", "Retrieval (ms)"],
        )
        .properties(title="Quality vs Latency (bubble size = index size)", height=360)
    )


def heatmap_recall(category_df: pd.DataFrame, title: str = "Recall@5 by query category") -> alt.Chart:
    """Heatmap of Recall@k across categories and pairings."""
    long = category_df.reset_index().melt(id_vars="category", var_name="Pair", value_name="Recall")
    base = alt.Chart(long).encode(
        x=alt.X("Pair:N", title=None),
        y=alt.Y("category:N", title=None),
    )
    rect = base.mark_rect().encode(
        color=alt.Color("Recall:Q", scale=alt.Scale(scheme="blues", domain=[0, 1])),
        tooltip=["Pair", "category", alt.Tooltip("Recall:Q", format=".3f")],
    )
    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("Recall:Q", format=".2f"),
        color=alt.condition(alt.datum.Recall > 0.6, alt.value("white"), alt.value("black")),
    )
    return (rect + text).properties(title=title, height=max(180, 40 * len(long["category"].unique())))


def simple_bar(df: pd.DataFrame, x: str, y: str, title: str, color: str = "Pair") -> alt.Chart:
    """Simple bar chart for chunk/efficiency style data."""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", axis=alt.Axis(labelAngle=-20, title=None)),
            y=alt.Y(f"{y}:Q", title=title),
            color=alt.Color(f"{color}:N", scale=alt.Scale(range=_PALETTE), title="Pairing"),
            tooltip=[color, alt.Tooltip(f"{y}:Q")],
        )
        .properties(title=title, height=320)
    )


def histogram(df: pd.DataFrame, value: str, title: str, maxbins: int = 40) -> alt.Chart:
    """Histogram of a numeric column (e.g. chunk size in characters)."""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", bin=alt.Bin(maxbins=maxbins), title=value),
            y=alt.Y("count():Q", title="chunks"),
            tooltip=[alt.Tooltip("count():Q", title="chunks")],
        )
        .properties(title=title, height=320)
    )


def stacked_bar(df: pd.DataFrame, x: str, y: str, color: str, title: str) -> alt.Chart:
    """Stacked bar chart (e.g. chunk counts by document, coloured by content type)."""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{x}:N", axis=alt.Axis(labelAngle=-20, title=None)),
            y=alt.Y(f"{y}:Q", title="chunks"),
            color=alt.Color(f"{color}:N", title=color),
            tooltip=[x, color, alt.Tooltip(f"{y}:Q")],
        )
        .properties(title=title, height=320)
    )


def percent_bar(df: pd.DataFrame, category: str, value: str, title: str) -> alt.Chart:
    """Horizontal bar of percentages (0-100) for metadata coverage."""
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", title="%", scale=alt.Scale(domain=[0, 100])),
            y=alt.Y(f"{category}:N", title=None, sort="-x"),
            tooltip=[category, alt.Tooltip(f"{value}:Q", format=".1f")],
        )
        .properties(title=title, height=260)
    )
