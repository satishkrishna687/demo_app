"""Admin dashboard — user controls and master API settings."""

from __future__ import annotations

import streamlit as st

from webapp.auth import require_admin, current_user
from webapp.config import (
    EMBEDDING_MODEL,
    EMBEDDING_BASE_URL,
    EMBEDDING_API_KEY_ENV,
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_API_KEY_ENV,
)
from webapp.db import (
    get_setting,
    set_setting,
    list_users,
    set_user_active,
    delete_user,
    list_documents,
)


def render_admin_page() -> None:
    """Render the admin dashboard."""
    st.title("🛡️ Admin Dashboard")
    st.caption("Manage users and master API settings for the ISO 15118 RAG web app.")

    admin = current_user()
    st.write(f"Logged in as admin: **{admin['username']}**")

    tab_users, tab_api, tab_docs = st.tabs(["Users", "API Settings", "Documents"])

    # ------------------------------------------------------------------ Users tab
    with tab_users:
        st.subheader("User management")
        users = list_users()
        for u in users:
            cols = st.columns([3, 2, 2, 2, 2])
            with cols[0]:
                st.write(f"**{u['username']}** ({u['role']})")
            with cols[1]:
                active = st.toggle(
                    "Active",
                    value=bool(u["is_active"]),
                    key=f"active_{u['id']}",
                )
                if active != bool(u["is_active"]):
                    set_user_active(u["id"], active)
                    st.rerun()
            with cols[2]:
                st.write("Active" if u["is_active"] else "Disabled")
            with cols[3]:
                st.write(f"Joined {u['created_at'][:10]}")
            with cols[4]:
                if u["username"] != "admin" and st.button(
                    "Delete", key=f"del_{u['id']}", type="secondary"
                ):
                    delete_user(u["id"])
                    st.rerun()

        st.divider()
        st.subheader("Create new user")
        with st.form("create_user_form"):
            new_user = st.text_input("Username")
            new_pass = st.text_input("Password", type="password")
            role = st.selectbox("Role", options=["user", "admin"], index=0)
            submitted = st.form_submit_button("Create user")
            if submitted:
                from auth import register

                if not new_user or not new_pass:
                    st.error("Username and password are required.")
                else:
                    uid = register(new_user, new_pass, role)
                    if uid is None:
                        st.error("Username already exists.")
                    else:
                        st.success(f"User '{new_user}' created.")
                        st.rerun()

    # ------------------------------------------------------------------ API Settings tab
    with tab_api:
        st.subheader("Master API settings (OpenRouter)")
        st.caption(
            "These values are used as defaults for all regular users. "
            "API keys are stored in the database (not in logs)."
        )

        with st.form("api_settings_form"):
            master_key = st.text_input(
                "OpenRouter API key",
                value=get_setting("master_api_key", ""),
                type="password",
                help="Shared by all users for embedding + generation.",
            )
            embedding_model = st.text_input(
                "Embedding model",
                value=get_setting("master_embedding_model", EMBEDDING_MODEL),
                help="e.g. qwen/qwen3-embedding-8b",
            )
            llm_model = st.text_input(
                "Generator model",
                value=get_setting("master_llm_model", LLM_MODEL),
                help="e.g. qwen/qwen3.7-flash",
            )
            submitted = st.form_submit_button("Save settings")
            if submitted:
                set_setting("master_api_key", master_key)
                set_setting("master_embedding_model", embedding_model)
                set_setting("master_llm_model", llm_model)
                st.success("API settings saved.")

        st.divider()
        st.write("Current endpoint configuration:")
        st.code(
            f"Embedding base URL: {EMBEDDING_BASE_URL}\n"
            f"LLM base URL: {LLM_BASE_URL}\n"
            f"Embedding API key env: {EMBEDDING_API_KEY_ENV}\n"
            f"LLM API key env: {LLM_API_KEY_ENV}"
        )

    # ------------------------------------------------------------------ Documents tab
    with tab_docs:
        st.subheader("All documents (system-wide)")
        docs = list_documents(user_id=None)
        if not docs:
            st.info("No documents uploaded yet.")
        else:
            for d in docs:
                cols = st.columns([3, 2, 2, 2, 2])
                with cols[0]:
                    st.write(f"**{d['filename']}**")
                with cols[1]:
                    st.write(f"User: {d['username']}")
                with cols[2]:
                    status_icon = "✅" if d["status"] == "indexed" else ("⏳" if d["status"] == "indexing" else "📄")
                    st.write(f"{status_icon} {d['status']}")
                with cols[3]:
                    st.write(d["created_at"][:16])
                with cols[4]:
                    st.write(f"Doc #{d['id']}")
