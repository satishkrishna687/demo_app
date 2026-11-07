"""Sidebar settings for runtime model selection and API keys.

Model names and API keys entered here are stored in ``st.session_state`` only and
applied through :mod:`models.runtime` — they are never written to disk or logs.
"""

from __future__ import annotations

import streamlit as st

from models import runtime

_EMBEDDING_MODEL_KEY = "settings_embedding_model"
_EMBEDDING_KEY_KEY = "settings_embedding_api_key"
_LLM_MODEL_KEY = "settings_llm_model"
_LLM_KEY_KEY = "settings_llm_api_key"


def _apply() -> None:
    """Push the current session settings into the runtime override store."""
    runtime.set_overrides(
        embedding_model=st.session_state.get(_EMBEDDING_MODEL_KEY) or None,
        embedding_api_key=st.session_state.get(_EMBEDDING_KEY_KEY) or None,
        llm_model=st.session_state.get(_LLM_MODEL_KEY) or None,
        llm_api_key=st.session_state.get(_LLM_KEY_KEY) or None,
    )


def render_settings_sidebar() -> None:
    """Render the model/API-key settings section in the Streamlit sidebar."""
    from config import EMBEDDING_MODEL, LLM_MODEL, LLM_PROVIDER

    # Initialise widget state once (widgets with `key=` must not also get `value=`).
    defaults = {
        _EMBEDDING_MODEL_KEY: "",
        _EMBEDDING_KEY_KEY: "",
        _LLM_MODEL_KEY: "",
        _LLM_KEY_KEY: "",
    }
    for key, default in defaults.items():
        st.session_state.setdefault(key, default)

    st.sidebar.divider()
    st.sidebar.subheader("🤖 Models & API keys")
    st.sidebar.caption(
        "Keys are kept in this browser session only (never stored or logged). "
        "Leave blank to use the environment / .env defaults."
    )

    st.sidebar.text_input(
        f"Embedding model (default: {EMBEDDING_MODEL})",
        key=_EMBEDDING_MODEL_KEY,
        help="OpenRouter model slug, e.g. qwen/qwen3-embedding-8b",
    )
    st.sidebar.text_input(
        "Embedding API key",
        key=_EMBEDDING_KEY_KEY,
        type="password",
        help="Leave blank to use the OPENROUTER_API_KEY environment variable.",
    )

    st.sidebar.text_input(
        f"Generator model (default: {LLM_MODEL})",
        key=_LLM_MODEL_KEY,
        help=f"Model slug for the {LLM_PROVIDER} provider, e.g. z-ai/glm-5.3-flash",
    )
    st.sidebar.text_input(
        "Generator API key",
        key=_LLM_KEY_KEY,
        type="password",
        help="Leave blank to use the environment / .env default.",
    )

    _apply()


def settings_signature() -> tuple:
    """Cache-key tuple: changes when the user alters model settings (forces rebuild)."""
    return runtime.signature()
