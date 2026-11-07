"""User chat page — ask questions against your own indexed document.

Uses the Pairing 2 hybrid retriever built during upload and the OpenRouter
generator model. API keys are hidden from regular users (master key from admin
settings or environment defaults are used automatically).
"""

from __future__ import annotations

import streamlit as st

from webapp.auth import require_auth, current_user
from webapp.config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY_ENV, EMBEDDING_MODEL
from webapp.db import get_setting
from models.llm import get_llm
from webapp.pipeline import user_index_dir


def _format_sources(docs: list) -> str:
    """Render retrieved documents with document/section/page citation headers."""
    parts = []
    for d in docs:
        meta = d.metadata or {}
        header = (
            f"[{meta.get('document_id', 'N/A')} | "
            f"Section {meta.get('section_number', 'N/A')} | "
            f"page {meta.get('page_number', '?')}]"
        )
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_chain(retriever):
    """Build the RAG chain using the OpenRouter generator model."""
    from chain.rag_chain import build_rag_chain

    # Use the master LLM model from settings/env; API key resolved lazily from
    # env (OPENROUTER_API_KEY) — hidden from the regular user.
    llm_model = get_setting("master_llm_model", LLM_MODEL)
    return build_rag_chain(retriever, llm_model=llm_model, temperature=0.0)


def render_chat_page() -> None:
    """Render the chat page."""
    st.title("💬 Chat with your ISO document")
    st.caption("Ask questions about your uploaded and indexed document.")

    user = current_user()
    if user is None:
        st.error("Not logged in.")
        return

    user_id = int(user["id"])

    # Retrieve or build the user's retriever from their persisted index.
    retriever = _get_user_retriever(user_id)
    if retriever is None:
        st.info("No indexed document found. Please upload and index a PDF in the Upload Document tab first.")
        return

    # Build the chain (cached per session).
    if "chat_chain" not in st.session_state or st.session_state.get("chat_user_id") != user_id:
        st.session_state["chat_chain"] = _build_chain(retriever)
        st.session_state["chat_user_id"] = user_id
        st.session_state["chat_messages"] = []

    chain = st.session_state["chat_chain"]

    # Chat history.
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about your document...")

    if user_input:
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                answer = chain.invoke(user_input)
                st.markdown(answer)
                st.session_state["chat_messages"].append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ Error generating answer: {type(exc).__name__}: {exc}")


def _get_user_retriever(user_id: int):
    """Load the user's Pairing 2 retriever from their persisted index.

    Returns None if no index exists yet.
    """
    from langchain_chroma import Chroma
    from models.embeddings import get_embeddings
    from retrieval.pair2_retriever import build_metadata_filtered_hybrid_retriever

    index_dir = str(user_index_dir(user_id))
    collection_name = f"user_{user_id}_pair2"

    # Check the index directory has content.
    import os
    if not os.path.isdir(index_dir) or not any(os.scandir(index_dir)):
        return None

    try:
        embeddings = get_embeddings()
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=index_dir,
        )
        # Rebuild a lightweight retriever from the existing store.
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        return retriever
    except Exception:
        return None
