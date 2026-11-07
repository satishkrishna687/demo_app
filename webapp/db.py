"""SQLite data store for the multi-user web application.

Tables:
- users: id, username, password_hash, role (user|admin), is_active, created_at
- documents: id, user_id, filename, file_path, status, created_at
- settings: key, value (master API settings, admin-only)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from webapp.config import DB_PATH

try:  # bcrypt is preferred but not required
    import bcrypt  # type: ignore

    _HAS_BCRYPT = True
except Exception:  # pragma: no cover - bcrypt optional
    _HAS_BCRYPT = False

_SALT = os.environ.get("WEBAPP_AUTH_SALT", "iso15118-webapp-salt-change-me")


def hash_password(password: str) -> str:
    """Hash a password for storage."""
    if _HAS_BCRYPT:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    return hashlib.sha256((_SALT + password).encode("utf-8")).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    if _HAS_BCRYPT:
        return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    return hmac.compare_digest(hash_password(password), stored_hash)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection, committing on success and rolling back on error."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables and seed the default admin user if the DB is new."""
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploaded',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )

        # Seed default admin if none exists.
        row = conn.execute(
            "SELECT id FROM users WHERE role = 'admin' LIMIT 1"
        ).fetchone()
        if row is None:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO users (username, password_hash, role, is_active, created_at) "
                "VALUES (?, ?, 'admin', 1, ?)",
                ("admin", hash_password("admin123"), now),
            )
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                ("master_api_key", "", now),
            )
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                ("master_embedding_model", "qwen/qwen3-embedding-8b", now),
            )
            conn.execute(
                "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                ("master_llm_model", "qwen/qwen3.7-flash", now),
            )


# --------------------------------------------------------------------------- users
def create_user(username: str, password: str, role: str = "user") -> int:
    """Create a user and return the new user id."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, password_hash, role, is_active, created_at) "
            "VALUES (?, ?, ?, 1, ?)",
            (username, hash_password(password), role, now),
        )
        return int(cur.lastrowid)


def get_user_by_username(username: str) -> dict | None:
    """Return a user row dict by username, or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, is_active, created_at "
            "FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def get_user(user_id: int) -> dict | None:
    """Return a user row dict by id, or None."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, is_active, created_at "
            "FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict]:
    """Return all users (id, username, role, is_active, created_at)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, username, role, is_active, created_at FROM users ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def set_user_active(user_id: int, is_active: bool) -> None:
    """Enable or disable a user account."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET is_active = ? WHERE id = ?",
            (1 if is_active else 0, user_id),
        )


def delete_user(user_id: int) -> None:
    """Delete a user and their documents."""
    with get_conn() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


# --------------------------------------------------------------------------- documents
def add_document(user_id: int, filename: str, file_path: str) -> int:
    """Record an uploaded document and return its id."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO documents (user_id, filename, file_path, status, created_at) "
            "VALUES (?, ?, ?, 'uploaded', ?)",
            (user_id, filename, file_path, now),
        )
        return int(cur.lastrowid)


def list_documents(user_id: int | None = None) -> list[dict]:
    """List documents, optionally filtered by user."""
    with get_conn() as conn:
        if user_id is None:
            rows = conn.execute(
                "SELECT d.id, d.user_id, u.username, d.filename, d.status, d.created_at "
                "FROM documents d JOIN users u ON d.user_id = u.id "
                "ORDER BY d.created_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, user_id, filename, file_path, status, created_at "
                "FROM documents WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
    return [dict(r) for r in rows]


def update_document_status(doc_id: int, status: str) -> None:
    """Update a document's processing status."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE documents SET status = ? WHERE id = ?", (status, doc_id)
        )


# --------------------------------------------------------------------------- settings
def get_setting(key: str, default: str = "") -> str:
    """Return a setting value, or default if not set."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    """Set (upsert) a setting value."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
            (key, value, now),
        )
