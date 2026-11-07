"""Multi-user ISO 15118 RAG web application.

Separate Streamlit application (entry point: ``webapp/app.py``) that does not
disturb the existing ``ui/`` application. Provides user login/registration,
per-user document upload + Pairing 2 pipeline, user chat, and an admin
dashboard for user and API-key management.
"""

from __future__ import annotations

__version__ = "0.1.0"
