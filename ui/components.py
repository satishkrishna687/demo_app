"""Reusable Streamlit UI components."""

from __future__ import annotations

from typing import Any

import streamlit as st


def _meta_str(meta: dict) -> str:
    parts = [
        f"Section {meta.get('section_number', 'N/A')}",
        f"{meta.get('section_title', '')}".strip(),
        f"[{meta.get('document_id', 'N/A')}]",
        f"(type: {meta.get('content_type', 'text')}, page: {meta.get('page_number', '?')})",
    ]
    return " — ".join(p for p in parts if p)


def render_source_panel(docs: list, latency_ms: float | None = None) -> None:
    """Render an expandable panel of retrieved chunks with their metadata."""
    with st.expander(f"📚 Sources ({len(docs)} chunks)", expanded=False):
        if latency_ms is not None:
            st.caption(f"Retrieval latency: {latency_ms:.0f} ms")
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            st.markdown(f"**{i}. {_meta_str(meta)}**")
            content = doc.page_content or ""
            if len(content) > 800:
                content = content[:800] + "\n\n... *(truncated)*"
            st.code(content, language="markdown")
            refs = meta.get("cross_references")
            if refs:
                st.caption(f"🔗 Cross-references: {', '.join(refs) if isinstance(refs, list) else refs}")
            st.divider()


def render_answer(answer: str, latency_ms: float | None = None) -> None:
    """Render an LLM answer with an optional latency badge."""
    st.markdown(answer or "_(no answer)_")
    if latency_ms is not None:
        st.caption(f"⏱️ Generated in {latency_ms:.0f} ms")


def render_chat_history(messages: list[dict[str, Any]]) -> None:
    """Render the full conversation, including source panels."""
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content", ""))
            if msg.get("sources"):
                render_source_panel(msg["sources"], msg.get("retrieval_ms"))


def export_chat_button(messages: list[dict[str, Any]]) -> None:
    """Provide a download button for the conversation as Markdown."""
    lines = ["# ISO 15118 RAG Chat Export\n"]
    for msg in messages:
        lines.append(f"## {msg['role'].capitalize()}\n")
        lines.append((msg.get("content") or "") + "\n")
        for src in msg.get("sources", []):
            meta = src.metadata
            lines.append(
                f"- {meta.get('document_id', 'N/A')} — "
                f"Section {meta.get('section_number', 'N/A')}: "
                f"{meta.get('section_title', '')}\n"
            )
    st.download_button(
        "📥 Export chat",
        data="\n".join(lines),
        file_name="chat_export.md",
        mime="text/markdown",
    )


def _pkg_version(name: str) -> str:
    """Return an installed package version, or a marker if it is missing."""
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return "not installed"


def langchain_stack_rows() -> list[tuple[str, str]]:
    """Return ``(component, implementation)`` rows proving the LangChain stack in use.

    Classes are imported but never instantiated, so this does not load any models.
    """
    from config import EMBEDDING_MODEL, LLM_MODEL, LLM_PROVIDER, RERANKER_MODEL

    rows: list[tuple[str, str]] = [
        (pkg, _pkg_version(pkg))
        for pkg in (
            "langchain",
            "langchain-core",
            "langchain-classic",
            "langchain-community",
            "langchain-chroma",
            "langchain-huggingface",
            "langchain-ollama",
            "langchain-text-splitters",
        )
    ]

    from langchain_chroma import Chroma
    from langchain_classic.retrievers import (
        EnsembleRetriever,
        ParentDocumentRetriever,
        SelfQueryRetriever,
    )
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.runnables import RunnableSequence
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    rows.append(("Chain (LCEL)", RunnableSequence.__name__))
    rows.append(("Retriever P1", ParentDocumentRetriever.__name__))
    rows.append(
        ("Retriever P2", f"{EnsembleRetriever.__name__} [{Chroma.__name__} + {BM25Retriever.__name__}]")
    )
    rows.append(
        (
            "Retriever P3",
            f"{EnsembleRetriever.__name__} [{SelfQueryRetriever.__name__} + {BM25Retriever.__name__}]",
        )
    )
    rows.append(("Vector store", Chroma.__name__))
    rows.append(("Embeddings", f"{HuggingFaceEmbeddings.__name__} ({EMBEDDING_MODEL})"))
    rows.append(("Splitter", RecursiveCharacterTextSplitter.__name__))

    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        rows.append(("LLM", f"{ChatOpenAI.__name__} ({LLM_MODEL})"))
    else:
        from langchain_ollama import ChatOllama

        rows.append(("LLM", f"{ChatOllama.__name__} ({LLM_MODEL})"))

    rows.append(("Reranker", f"CrossEncoder ({RERANKER_MODEL})"))
    return rows


def render_langchain_stack() -> None:
    """Render a diagnostics panel listing the LangChain components in use."""
    with st.expander("🔧 LangChain stack", expanded=False):
        st.caption(
            "Components used by the running pipelines. Ask a question in the Chat tab — the "
            "Sources panel shows LangChain `Document` objects with their metadata."
        )
        rows = langchain_stack_rows()
        st.dataframe(
            {"Component": [r[0] for r in rows], "Implementation": [r[1] for r in rows]},
            hide_index=True,
            width="stretch",
        )
