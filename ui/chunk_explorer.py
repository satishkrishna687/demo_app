"""Chunk Explorer tab: browse and visualise every chunk produced by each pairing.

Chunking runs on the cached parsed documents and needs **no embeddings**, so this tab is fast
and does not trigger the heavy pipeline builds. Large chunk sets are handled with filters and a
virtualised table rather than rendering everything at once.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from ui import charts

PAIRING_VARIANTS = {
    "Pairing 1: Recursive + Parent-Child": ["parents", "children"],
    "Pairing 2: Section + Hybrid": ["section chunks"],
    "Pairing 3: Advanced ISO + Graph": ["table + text chunks"],
}

SHORT_LABELS = {
    "Pairing 1: Recursive + Parent-Child": "P1 Recursive+Parent-Child",
    "Pairing 2: Section + Hybrid": "P2 Section+Hybrid",
    "Pairing 3: Advanced ISO + Graph": "P3 Advanced+Graph",
}


# --------------------------------------------------------------------------- loading
@st.cache_data(show_spinner="Chunking documents...")
def load_chunks(pairing: str, variant: str) -> list[dict]:
    """Return ``[{text, metadata}]`` for the selected pairing/variant (cached)."""
    from ui.pipeline_registry import load_structures

    documents = load_structures()

    if pairing.startswith("Pairing 1"):
        from ingestion.chunking_pair1 import create_parent_child_chunks

        parents, children = create_parent_child_chunks(documents)
        chunks = parents if variant == "parents" else children
    elif pairing.startswith("Pairing 2"):
        from ingestion.chunking_pair2 import create_section_based_chunks

        chunks = create_section_based_chunks(documents)
    else:
        from ingestion.chunking_pair3 import create_advanced_iso_chunks

        chunks = create_advanced_iso_chunks(documents)

    return [{"text": c.page_content, "metadata": dict(c.metadata)} for c in chunks]


# --------------------------------------------------------------------------- shaping
def chunks_to_df(records: list[dict]) -> pd.DataFrame:
    """Flatten chunk records into a metadata table (one row per chunk)."""
    rows = []
    for i, record in enumerate(records):
        meta = record.get("metadata", {}) or {}
        text = record.get("text", "") or ""
        rows.append(
            {
                "idx": i,
                "document_id": meta.get("document_id"),
                "section_number": meta.get("section_number"),
                "section_title": meta.get("section_title"),
                "content_type": meta.get("content_type", "text"),
                "page": meta.get("page_number"),
                "chars": len(text),
                "words": len(text.split()),
                "heading_path": meta.get("heading_path"),
                "parent_id": meta.get("parent_id"),
                "child_index": meta.get("child_index"),
                "cross_refs": len(meta.get("cross_references") or []),
                "preview": text[:120].replace("\n", " "),
            }
        )
    return pd.DataFrame(rows)


def _present(series: pd.Series) -> pd.Series:
    return series.notna() & (series.astype(str).str.strip() != "")


def coverage_df(df: pd.DataFrame) -> pd.DataFrame:
    """Percentage of chunks carrying each metadata field."""
    if df.empty:
        return pd.DataFrame(columns=["Field", "Percent"])
    n = len(df)
    fields = {
        "document_id": "document_id",
        "section_number": "section_number",
        "section_title": "section_title",
        "heading_path": "heading_path",
        "page_number": "page",
        "content_type": "content_type",
        "parent_id": "parent_id",
    }
    rows = []
    for label, column in fields.items():
        if column in df.columns:
            rows.append({"Field": label, "Percent": round(100.0 * _present(df[column]).sum() / n, 1)})
    return pd.DataFrame(rows)


def content_type_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["content_type", "chunks"])
    return df.groupby("content_type").size().reset_index(name="chunks")


def per_document_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["document_id", "content_type", "chunks"])
    return df.groupby(["document_id", "content_type"]).size().reset_index(name="chunks")


# --------------------------------------------------------------------------- render
def render_chunk_explorer() -> None:
    """Render the Chunk Explorer tab."""
    st.subheader("🧩 Chunk Explorer")
    st.caption(
        "Browse every chunk produced by each pairing. Chunking uses the cached parsed documents "
        "and needs no embeddings, so this view is fast."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        pairing = st.selectbox("Pairing", options=list(PAIRING_VARIANTS.keys()), index=1)
    with col_b:
        variant = st.selectbox("Variant", options=PAIRING_VARIANTS[pairing], index=0)

    records = load_chunks(pairing, variant)
    df = chunks_to_df(records)

    if df.empty:
        st.warning("No chunks produced for this selection.")
        return

    # ---- headline metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chunks", f"{len(df):,}")
    m2.metric("Avg size (chars)", f"{df['chars'].mean():.0f}")
    m3.metric("Min / Max (chars)", f"{df['chars'].min()} / {df['chars'].max()}")
    m4.metric("With section", f"{100.0 * _present(df['section_number']).sum() / len(df):.0f}%")

    # ---- filters
    st.markdown("### Filters")
    f1, f2, f3 = st.columns(3)
    with f1:
        docs_sel = st.multiselect(
            "Document", options=sorted(df["document_id"].dropna().unique().tolist())
        )
    with f2:
        types_sel = st.multiselect(
            "Content type", options=sorted(df["content_type"].dropna().unique().tolist())
        )
    with f3:
        section_q = st.text_input("Section contains", placeholder="e.g., 3.1").strip()

    f4, f5 = st.columns(2)
    with f4:
        lo, hi = int(df["chars"].min()), int(df["chars"].max())
        size_range = st.slider("Chunk size (chars)", min_value=lo, max_value=max(hi, lo + 1), value=(lo, max(hi, lo + 1)))
    with f5:
        text_q = st.text_input("Text contains", placeholder="e.g., SLAC").strip()

    mask = pd.Series(True, index=df.index)
    if docs_sel:
        mask &= df["document_id"].isin(docs_sel)
    if types_sel:
        mask &= df["content_type"].isin(types_sel)
    if section_q:
        mask &= df["section_number"].fillna("").str.contains(section_q, case=False, regex=False)
    mask &= df["chars"].between(size_range[0], size_range[1])
    if text_q:
        needle = text_q.lower()
        mask &= pd.Series(
            [needle in (records[i].get("text") or "").lower() for i in df.index], index=df.index
        )

    view = df[mask]
    st.caption(f"Showing **{len(view):,}** of **{len(df):,}** chunks.")

    # ---- charts
    st.markdown("### Distributions")
    c1, c2 = st.columns(2)
    with c1:
        if not view.empty:
            st.altair_chart(charts.histogram(view, "chars", "Chunk size distribution (chars)"), width="stretch")
    with c2:
        ct = content_type_df(view)
        if not ct.empty:
            st.altair_chart(
                charts.stacked_bar(ct, "content_type", "chunks", "content_type", "Chunks by content type"),
                width="stretch",
            )

    c3, c4 = st.columns(2)
    with c3:
        pd_df = per_document_df(view)
        if not pd_df.empty:
            st.altair_chart(
                charts.stacked_bar(pd_df, "document_id", "chunks", "content_type", "Chunks per document"),
                width="stretch",
            )
    with c4:
        cov = coverage_df(view)
        if not cov.empty:
            st.altair_chart(charts.percent_bar(cov, "Field", "Percent", "Metadata coverage (%)"), width="stretch")

    # ---- table + viewer
    st.markdown("### Chunks")
    table_cols = [
        "idx", "document_id", "section_number", "content_type", "page", "chars", "words",
        "cross_refs", "preview",
    ]
    event = st.dataframe(
        view[table_cols],
        hide_index=True,
        width="stretch",
        height=420,
        on_select="rerun",
        selection_mode="single-row",
        key="chunk_table",
    )

    selected_rows = getattr(getattr(event, "selection", None), "rows", []) or []
    if selected_rows:
        idx = int(view.iloc[selected_rows[0]]["idx"])
        record = records[idx]
        st.markdown(f"#### Chunk viewer — index {idx}")
        st.json(record.get("metadata", {}))
        st.code(record.get("text", ""), language="markdown")
    else:
        st.caption("Select a row in the table to view the full chunk text and metadata.")

    # ---- all-pairings overview (opt-in; chunks all three)
    with st.expander("📊 All-pairings overview"):
        if st.button("Compute overview (chunks all three pairings)"):
            rows = []
            for name, variants in PAIRING_VARIANTS.items():
                recs = load_chunks(name, variants[0])
                sizes = [len(r.get("text", "")) for r in recs]
                rows.append(
                    {
                        "Pair": SHORT_LABELS[name],
                        "Chunks": len(recs),
                        "Avg chunk (chars)": round(sum(sizes) / len(sizes), 1) if sizes else 0.0,
                    }
                )
            overview = pd.DataFrame(rows)
            st.dataframe(overview, hide_index=True, width="stretch")
            st.altair_chart(
                charts.simple_bar(overview, "Pair", "Chunks", "Chunks per pairing"), width="stretch"
            )
        else:
            st.caption("Click the button to chunk all three pairings and compare counts.")