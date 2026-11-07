"""Streamlit Q&A interface for the ISO 15118 RAG pipelines.

Three tabs:
- **Ask (Chat)** — query a selected pipeline (or compare all three) with source citations.
- **Evaluation & Comparison** — charts and tables from the latest evaluation runs.
- **Chunks** — browse and visualise every chunk produced by each pairing.

Run with:  streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file as a script and puts the script's directory (ui/) on sys.path,
# not the repository root. Add the root so `import config`, `models`, etc. resolve.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import time

import streamlit as st

from config import LLM_MODEL, LLM_PROVIDER
from models.device import describe_device
from ui.chunk_explorer import render_chunk_explorer
from ui.components import (
    export_chat_button,
    render_answer,
    render_chat_history,
    render_langchain_stack,
    render_source_panel,
)
from ui.eval_dashboard import render_dashboard
from ui.pipeline_registry import PIPELINE_NAMES, compare_all, get_pipeline, run_query
from ui.settings import render_settings_sidebar, settings_signature

st.set_page_config(page_title="ISO 15118 RAG", page_icon="📘", layout="wide")


def _history_messages(messages: list[dict]) -> list:
    """Convert stored chat messages into LangChain chat history (user/assistant turns)."""
    from langchain_core.messages import AIMessage, HumanMessage

    history = []
    for msg in messages:
        content = (msg.get("content") or "").strip()
        if not content:
            continue
        if msg.get("role") == "user":
            history.append(HumanMessage(content=content))
        elif msg.get("role") == "assistant":
            history.append(AIMessage(content=content))
    return history

# ---------------------------------------------------------------- Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")

    selected_pipeline = st.selectbox(
        "Pipeline",
        options=PIPELINE_NAMES,
        index=len(PIPELINE_NAMES) - 1,
        help="Choose which chunking + retrieval strategy to use (Chat tab).",
    )

    mode = st.radio("Mode", options=["Single pipeline", "Compare all three"], index=0)

    st.divider()

    if selected_pipeline.startswith("Pairing 2"):
        section_filter = st.text_input(
            "Filter by section (optional)", placeholder="e.g., 3.1.1"
        ).strip() or None
    else:
        section_filter = None

    render_settings_sidebar()

    st.divider()
    st.caption(f"Device: `{describe_device()}`")
    st.caption(f"LLM: `{LLM_PROVIDER}` / `{LLM_MODEL}`")
    render_langchain_stack()

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if st.session_state.get("messages"):
        export_chat_button(st.session_state.messages)

# ---------------------------------------------------------------- Main
st.title("📘 ISO 15118 RAG")

tab_chat, tab_eval, tab_chunks = st.tabs(
    ["💬 Ask (Chat)", "📊 Evaluation & Comparison", "🧩 Chunks"]
)

# ---------------------------------------------------------------- Chat tab
with tab_chat:
    st.caption(f"Active pipeline: **{selected_pipeline}** | Mode: **{mode}**")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    render_chat_history(st.session_state.messages)

    user_input = st.chat_input("Ask a question about the ISO specification...")

    if user_input:
        history = _history_messages(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            if mode == "Single pipeline":
                try:
                    pipeline = get_pipeline(selected_pipeline, settings_key=settings_signature())
                    t0 = time.perf_counter()
                    answer, sources = run_query(pipeline, user_input, history, section_filter)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                except Exception as exc:  # surface errors in the UI rather than swallowing them
                    st.error(f"Pipeline error: {type(exc).__name__}: {exc}")
                    answer, sources, elapsed_ms = "", [], None

                if answer or sources:
                    render_answer(answer, elapsed_ms)
                    render_source_panel(sources)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "retrieval_ms": elapsed_ms,
                        }
                    )

            else:  # Compare all three
                st.markdown(f"### Comparing all three pipelines for: *{user_input}*")
                try:
                    results = compare_all(user_input, history, section_filter)
                except Exception as exc:
                    st.error(f"Comparison error: {type(exc).__name__}: {exc}")
                    results = {}

                for name, (answer, sources) in results.items():
                    st.markdown(f"---\n#### 🔹 {name}")
                    st.markdown(answer or "_(no answer)_")
                    render_source_panel(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Compared all 3 pipelines for: {user_input}",
                        "sources": [],
                    }
                )

# ---------------------------------------------------------------- Evaluation tab
with tab_eval:
    render_dashboard()

# ---------------------------------------------------------------- Chunks tab
with tab_chunks:
    render_chunk_explorer()
