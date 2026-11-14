"""User document upload page with Pairing 2 pipeline build and progress bar."""

from __future__ import annotations

import shutil
from pathlib import Path

import streamlit as st

from webapp.auth import require_auth, current_user
from webapp.config import USERS_DIR
from webapp.db import add_document, list_documents, update_document_status
from webapp.pipeline import build_pairing2_pipeline, reset_user_index, user_pdf_dir
from webapp.secrets import require_openrouter_api_key


def _save_uploaded_file(user_id: int, uploaded_file) -> Path:
    """Save an uploaded Streamlit file to the user's PDF directory and return its path."""
    pdf_dir = user_pdf_dir(user_id)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = pdf_dir / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def render_upload_page() -> None:
    """Render the document upload + pipeline build page."""
    st.title("📤 Upload ISO Document")
    st.caption("Upload an ISO technical document. The pipeline will build a searchable index for your chat.")

    user = current_user()
    if user is None:
        st.error("Not logged in.")
        return

    # Check that the admin has set the OpenRouter API key.
    try:
        require_openrouter_api_key()
    except ValueError as exc:
        st.error(str(exc))
        st.info("Please ask the admin to set the OpenRouter API key in the Admin Dashboard.")
        return

    user_id = int(user["id"])

    # Show existing documents for this user.
    docs = list_documents(user_id)
    if docs:
        st.subheader("Your documents")
        for d in docs:
            status_icon = "✅" if d["status"] == "indexed" else ("⏳" if d["status"] == "indexing" else "📄")
            st.write(f"{status_icon} {d['filename']} — {d['status']}")

    st.divider()

    uploaded = st.file_uploader("Choose an ISO PDF", type=["pdf"], accept_multiple_files=False)

    if uploaded is not None:
        if st.button("Build pipeline", type="primary"):
            _run_build(user_id, uploaded)


def _run_build(user_id: int, uploaded_file) -> None:
    """Run the ingestion + indexing pipeline with a progress bar."""
    # Save the PDF.
    pdf_path = _save_uploaded_file(user_id, uploaded_file)
    doc_id = add_document(user_id, uploaded_file.name, str(pdf_path))
    update_document_status(doc_id, "indexing")

    progress = st.progress(0, text="Starting pipeline build...")
    status_text = st.empty()

    try:
        # Stage 1: parse (Docling)
        progress.progress(10, text="Parsing PDF with Docling...")
        status_text.info("Stage 1/4: Parsing PDF...")

        # Stage 2: chunk
        progress.progress(30, text="Chunking into sections...")
        status_text.info("Stage 2/4: Chunking (tables kept atomic)...")

        # Stage 3: embed + index (OpenRouter embeddings)
        progress.progress(60, text="Embedding chunks via OpenRouter...")
        status_text.info("Stage 3/4: Embedding and indexing...")

        retriever, vectorstore, chunks = build_pairing2_pipeline(
            user_id, pdf_path, document_id=uploaded_file.name
        )

        # Stage 4: done
        progress.progress(100, text="Pipeline ready.")
        status_text.success("✅ Pipeline built successfully! You can now chat in the Chat tab.")

        update_document_status(doc_id, "indexed")
        st.session_state["last_built_doc"] = doc_id
        st.session_state["last_built_retriever"] = retriever

    except Exception as exc:  # noqa: BLE001 - surface build errors in the UI
        progress.empty()
        status_text.error(f"❌ Pipeline build failed: {type(exc).__name__}: {exc}")
        update_document_status(doc_id, "failed")


def build_multiple_pipelines(user_id: int, uploaded_files: list) -> None:
    """Build Pairing 2 pipelines for multiple uploaded files with a progress bar."""
    if len(uploaded_files) > 5:
        st.error("Please upload at most 5 files at a time.")
        return

    total = len(uploaded_files)
    progress = st.progress(0, text="Starting pipeline build...")
    status_text = st.empty()

    for idx, uploaded_file in enumerate(uploaded_files, start=1):
        # Save the PDF.
        pdf_path = _save_uploaded_file(user_id, uploaded_file)
        doc_id = add_document(user_id, uploaded_file.name, str(pdf_path))
        update_document_status(doc_id, "indexing")

        stage_pct = int((idx - 1) / total * 100)
        progress.progress(stage_pct, text=f"Processing {uploaded_file.name} ({idx}/{total})...")
        status_text.info(f"File {idx}/{total}: {uploaded_file.name}")

        try:
            retriever, vectorstore, chunks = build_pairing2_pipeline(
                user_id, pdf_path, document_id=uploaded_file.name
            )
            update_document_status(doc_id, "indexed")
            st.session_state["last_built_doc"] = doc_id
            st.session_state["last_built_retriever"] = retriever
        except Exception as exc:  # noqa: BLE001
            update_document_status(doc_id, "failed")
            st.error(f"❌ Failed to index {uploaded_file.name}: {type(exc).__name__}: {exc}")

    progress.progress(100, text="All files processed.")
    status_text.success(f"✅ Processed {total} file(s). You can now chat in the Chat tab.")
