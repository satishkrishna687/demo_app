"""Runtime model overrides (set from the UI at request time).

The factories in :mod:`models.llm` and :mod:`models.embeddings` read configuration
from ``config.py`` at import time. The Streamlit UI lets the user pick model names
and supply API keys at runtime, so those factories consult this module first:
explicit call arguments > runtime overrides > config defaults.

Overrides live only in process memory — API keys are never written to disk or logs.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_OVERRIDES: dict[str, str] = {}

# Recognised override keys.
LLM_MODEL = "llm_model"
LLM_API_KEY = "llm_api_key"
LLM_BASE_URL = "llm_base_url"
EMBEDDING_MODEL = "embedding_model"
EMBEDDING_API_KEY = "embedding_api_key"
EMBEDDING_PROVIDER = "embedding_provider"

_ALL_KEYS = frozenset(
    {LLM_MODEL, LLM_API_KEY, LLM_BASE_URL, EMBEDDING_MODEL, EMBEDDING_API_KEY, EMBEDDING_PROVIDER}
)


def set_overrides(**values: str | None) -> None:
    """Set runtime overrides. ``None`` values clear the corresponding key."""
    with _LOCK:
        for name, value in values.items():
            if name not in _ALL_KEYS:
                raise ValueError(f"Unknown override key: {name!r}")
            if value is None or (isinstance(value, str) and not value.strip()):
                _OVERRIDES.pop(name, None)
            else:
                _OVERRIDES[name] = value.strip()


def get_override(name: str) -> str | None:
    """Return the current override for ``name`` (or ``None``)."""
    with _LOCK:
        return _OVERRIDES.get(name)


def clear_overrides() -> None:
    """Drop all runtime overrides (useful for tests)."""
    with _LOCK:
        _OVERRIDES.clear()


def signature() -> tuple:
    """Stable tuple of current overrides, usable as a cache key."""
    with _LOCK:
        return tuple(sorted(_OVERRIDES.items()))
