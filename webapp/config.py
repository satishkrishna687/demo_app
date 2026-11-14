"""Configuration for the multi-user web application.

All values are hardcoded except OPENROUTER_API_KEY, which must be set in the
environment (or in .env) and is never committed to the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root (parent of this webapp package).
ROOT = Path(__file__).resolve().parents[1]

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

# OPENROUTER_API_KEY is read from the environment at runtime (never hardcoded).
# It is resolved lazily by the model factories.

# Ensure directories exist at import time (safe, idempotent).
DATA_DIR.mkdir(parents=True, exist_ok=True)
USERS_DIR.mkdir(parents=True, exist_ok=True)
