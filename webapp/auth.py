"""Authentication helpers for the multi-user web application.

Passwords are hashed via :func:`webapp.db.hash_password` / verified via
:func:`webapp.db.verify_password` (bcrypt when available, salted SHA-256
fallback). Session state is managed entirely in ``st.session_state`` —
nothing is written to disk except the password hashes in SQLite.
"""

from __future__ import annotations

import streamlit as st

from webapp.db import (
    create_user,
    get_user_by_username,
    init_db,
    list_users,
    set_user_active,
    delete_user,
    get_user,
    hash_password,
    verify_password,
)


def init_auth() -> None:
    """Initialize the database (idempotent). Call once at app start."""
    init_db()


def login(username: str, password: str) -> dict | None:
    """Validate credentials and return the user dict, or None on failure."""
    user = get_user_by_username(username)
    if user is None:
        return None
    if not user["is_active"]:
        return None
    if not verify_password(password, user["password_hash"]):
        return None
    return user


def register(username: str, password: str, role: str = "user") -> int | None:
    """Register a new user. Returns the new user id, or None if username exists."""
    if get_user_by_username(username) is not None:
        return None
    return create_user(username, password, role)


def logout() -> None:
    """Clear the authentication session."""
    for key in ("auth_user", "auth_role", "auth_authenticated"):
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    """Return True if a user is logged in."""
    return bool(st.session_state.get("auth_authenticated"))


def current_user() -> dict | None:
    """Return the logged-in user dict, or None."""
    if not is_authenticated():
        return None
    return st.session_state.get("auth_user")


def require_auth() -> None:
    """Redirect to login if not authenticated (call at top of protected pages)."""
    if not is_authenticated():
        st.warning("Please log in to continue.")
        st.stop()


def require_admin() -> None:
    """Redirect to login if not an active admin."""
    require_auth()
    if current_user().get("role") != "admin":
        st.error("Admin access required.")
        st.stop()


def render_login_form() -> None:
    """Render the login / register form in the main area."""
    st.title("🔐 ISO 15118 RAG — Sign In")

    tab_login, tab_register = st.tabs(["Login", "Register"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                user = login(username, password)
                if user is None:
                    st.error("Invalid username or password.")
                else:
                    st.session_state["auth_user"] = user
                    st.session_state["auth_role"] = user["role"]
                    st.session_state["auth_authenticated"] = True
                    st.success(f"Welcome, {user['username']}!")
                    st.rerun()

    with tab_register:
        with st.form("register_form"):
            new_user = st.text_input("Choose username")
            new_pass = st.text_input("Choose password", type="password")
            submitted = st.form_submit_button("Register")
            if submitted:
                if not new_user or not new_pass:
                    st.error("Username and password are required.")
                else:
                    uid = register(new_user, new_pass, "user")
                    if uid is None:
                        st.error("Username already exists.")
                    else:
                        st.success("Account created. Please log in.")


def render_logout_button() -> None:
    """Render a logout button in the sidebar."""
    if st.sidebar.button("🚪 Logout"):
        logout()
        st.rerun()
