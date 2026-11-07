"""Configuration for the multi-user web application.

Reads environment variables (see ``.env.example``) with sensible defaults.
Keeps heavy imports out so the module stays importable everywhere.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (parent of this webapp package).
ROOT = Path(__file__).resolve().parents[1]

# --- Directories ---
DATA_DIR: Path = ROOT / "data"
USERS_DIR: Path = DATA_DIR / "users"          # per-user uploads + indexes
DB_PATH: Path = DATA_DIR / "users.db"         # SQLite auth + settings store

# --- Pairing 2 defaults (reuses existing ingestion/retrieval code) ---
PAIRING: str = "2"
CHUNK_SIZE: int = int(os.getenv("WEBAPP_CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("WEBAPP_CHUNK_OVERLAP", "50"))
TOP_K: int = int(os.getenv("WEBAPP_TOP_K", "5"))

# --- OpenRouter / OpenAI-compatible API defaults ---
EMBEDDING_PROVIDER: str = os.getenv("WEBAPP_EMBEDDING_PROVIDER", "openai").strip().lower()
EMBEDDING_MODEL: str = os.getenv(
    "WEBAPP_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b"
)
EMBEDDING_BASE_URL: str = os.getenv(
    "WEBAPP_EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1"
).strip()
EMBEDDING_API_KEY_ENV: str = os.getenv("WEBAPP_EMBEDDING_API_KEY_ENV", "OPENROUTER_API_KEY")

LLM_PROVIDER: str = os.getenv("WEBAPP_LLM_PROVIDER", "openai").strip().lower()
LLM_MODEL: str = os.getenv("WEBAPP_LLM_MODEL", "qwen/qwen3.7-flash")
LLM_BASE_URL: str = os.getenv(
    "WEBAPP_LLM_BASE_URL", "https://openrouter.ai/api/v1"
).strip()
LLM_API_KEY_ENV: str = os.getenv("WEBAPP_LLM_API_KEY_ENV", "OPENROUTER_API_KEY")

# --- Auth ---
SESSION_TIMEOUT_MINUTES: int = int(os.getenv("WEBAPP_SESSION_TIMEOUT_MINUTES", "120"))
ADMIN_USERNAME: str = os.getenv("WEBAPP_ADMIN_USERNAME", "admin")

# Ensure directories exist at import time (safe, idempotent).
DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_DIR.mkdir(parents=True, exist_ok=True)
