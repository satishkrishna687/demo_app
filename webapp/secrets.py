"""Secret configuration for the multi-user web application.

The OpenRouter API key is set by the admin through the admin dashboard and stored
in the SQLite settings table. It is shared by all users.

Resolution order (highest priority first):
1. SQLite settings table (``master_api_key``) — set by the admin via the Admin
   Dashboard. This ALWAYS wins so the admin's key reflects for everyone.
2. Streamlit secrets (``st.secrets["OPENROUTER_API_KEY"]``) — for Cloud deploy
   before the admin has used the dashboard.
3. Environment variable ``OPENROUTER_API_KEY`` — for local dev.
4. None — the admin has not set the key yet; callers should show an error.
"""

from __future__ import annotations

import os

try:  # Streamlit provides st.secrets when running via `streamlit run`
    import streamlit as st

    _HAS_STREAMLIT = True
except Exception:  # pragma: no cover - streamlit may not be installed in some contexts
    _HAS_STREAMLIT = False


def get_openrouter_api_key() -> str | None:
    """Return the OpenRouter API key if set, else None."""
    # 1. Admin-set key from the SQLite settings table (highest priority).
    try:
        from webapp.db import get_setting

        key = get_setting("master_api_key", "").strip()
        if key:
            return key
    except Exception:
        pass

    # 2. Streamlit secrets (Cloud deploy).
    if _HAS_STREAMLIT:
        try:
            key = st.secrets.get("OPENROUTER_API_KEY", "")
            if key:
                return str(key).strip()
        except Exception:
            pass

    # 3. Environment variable (local dev).
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key

    return None


def require_openrouter_api_key() -> str:
    """Return the OpenRouter API key, raising ``ValueError`` if not set."""
    key = get_openrouter_api_key()
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Ask the admin to configure it in the Admin Dashboard > API Settings."
        )
    return key


# ---------------------------------------------------------------------------
# Hardcoded configuration (no secrets here)
# ---------------------------------------------------------------------------

from pathlib import Path

# Repository root (parent of this webapp package).
ROOT: Path = Path(__file__).resolve().parents[1]

# --- Directories ---
DATA_DIR: Path = ROOT / "data"                    # per-user uploads + indexes
USERS_DIR: Path = DATA_DIR / "users"
DB_PATH: Path = DATA_DIR / "users.db"             # SQLite auth + settings store

# --- Pairing 2 defaults (reuses existing ingestion/retrieval code) ---
PAIRING: str = "2"
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50
TOP_K: int = 5

# --- OpenRouter / OpenAI-compatible API defaults (hardcoded) ---
EMBEDDING_PROVIDER: str = "openai"
EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"
EMBEDDING_BASE_URL: str = "https://openrouter.ai/api/v1"
EMBEDDING_API_KEY_ENV: str = "OPENROUTER_API_KEY"

LLM_PROVIDER: str = "openai"
LLM_MODEL: str = "qwen/qwen3.7-flash"
LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
LLM_API_KEY_ENV: str = "OPENROUTER_API_KEY"

# --- Auth ---
SESSION_TIMEOUT_MINUTES: int = 120
ADMIN_USERNAME: str = "admin"

# Ensure directories exist at import time (safe, idempotent).
DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_DIR.mkdir(parents=True, exist_ok=True)
