"""Multi-user ISO 15118 RAG web application — entry point.

Run with:  streamlit run webapp/app.py

This application lives entirely under ``webapp/`` and does not disturb the
existing ``ui/`` application. It provides:
- User login / registration (Phase 1)
- Per-user document upload + Pairing 2 pipeline (Phase 2)
- User chat against their own index (Phase 3)
- Admin dashboard for users and master API settings (Phase 4)
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is importable (Streamlit puts webapp/ on sys.path,
# not the root). Needed to import config, models, ingestion, retrieval, etc.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from webapp.config import ROOT, USERS_DIR, ADMIN_USERNAME
from webapp.auth import (
    init_auth,
    is_authenticated,
    current_user,
    require_auth,
    require_admin,
    render_login_form,
    render_logout_button,
)
from webapp.db import list_users, list_documents, get_setting, set_setting

st.set_page_config(
    page_title="ISO 15118 RAG — Multi-User",
    page_icon="📘",
    layout="wide",
)

# Initialize database + auth (idempotent).
init_auth()

# --------------------------------------------------------------------------- Sidebar
with st.sidebar:
    st.title("📘 ISO 15118 RAG")
    st.caption("Multi-user edition")

    if is_authenticated():
        user = current_user()
        st.success(f"Logged in as **{user['username']}** ({user['role']})")
        render_logout_button()
        st.divider()

        if user["role"] == "admin":
            st.subheader("🛡️ Admin Dashboard")
            st.caption("Manage users and API settings.")
            if st.button("Open Admin Dashboard", type="primary"):
                st.session_state["admin_view"] = True
                st.rerun()
        else:
            # User document upload section (sidebar, always available for users)
            st.subheader("📤 Upload Document")
            uploaded_files = st.file_uploader(
                "Choose ISO PDF(s) (up to 5)",
                type=["pdf"],
                accept_multiple_files=True,
                key="sidebar_uploader",
            )
            if uploaded_files:
                if len(uploaded_files) > 5:
                    st.error("Please upload at most 5 files at a time.")
                else:
                    if st.button("Build pipeline for all files", type="primary"):
                        from webapp.user_upload import build_multiple_pipelines
                        build_multiple_pipelines(int(user["id"]), uploaded_files)

            st.divider()
            st.subheader("Your documents")
            docs = list_documents(int(user["id"]))
            if not docs:
                st.caption("No documents yet.")
            else:
                for d in docs:
                    status_icon = "✅" if d["status"] == "indexed" else ("⏳" if d["status"] == "indexing" else "📄")
                    st.write(f"{status_icon} {d['filename']} — {d['status']}")
    else:
        page = None

# --------------------------------------------------------------------------- Public (no auth)
if not is_authenticated():
    render_login_form()
    st.stop()

# --------------------------------------------------------------------------- Authenticated main area
user = current_user()

if user["role"] == "admin" and st.session_state.get("admin_view", False):
    from webapp.admin_dashboard import render_admin_page

    render_admin_page()
    if st.button("← Back to Chat"):
        st.session_state["admin_view"] = False
        st.rerun()
else:
    # User chat in the center by default
    from webapp.user_chat import render_chat_page

    render_chat_page()
