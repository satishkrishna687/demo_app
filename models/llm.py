"""Chat LLM factory.

Switches between a fully local model served by Ollama and the OpenAI API, driven by
``LLM_PROVIDER`` in the environment (``ollama`` by default, ``openai`` optional). This is
the single hybrid switch in the stack: embeddings and the reranker always run locally.

Example:
    from models.llm import get_llm
    llm = get_llm()                      # honours LLM_PROVIDER / LLM_MODEL
    llm = get_llm(provider="openai")      # explicit override
"""

from __future__ import annotations

import urllib.request

from typing import Any

from config import LLM_API_KEY_ENV, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER, OLLAMA_BASE_URL
from langchain_core.language_models.chat_models import BaseChatModel
from models import runtime

_CACHE: dict[tuple[str, str, float], BaseChatModel] = {}

_SUPPORTED_PROVIDERS = ("ollama", "openai")


def _resolve_api_key(env_var: str) -> str:
    """Resolve an API key from the environment, raising a clear error if missing."""
    import os

    key = os.environ.get(env_var, "").strip()
    if not key:
        raise ValueError(
            f"API key environment variable {env_var!r} is not set. "
            "Set it in your environment or .env file."
        )
    return key


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    base_url: str | None = None,
    api_key: str | None = None,
) -> BaseChatModel:
    """Return a cached chat model for the configured or overridden provider.

    Args:
        provider: ``"ollama"`` or ``"openai"``. Defaults to ``LLM_PROVIDER``.
        model: Model name. Defaults to ``LLM_MODEL``.
        temperature: Sampling temperature (0 for deterministic RAG answers).
        base_url: Override the OpenAI-compatible endpoint (e.g.
            ``https://openrouter.ai/api/v1``). Defaults to ``LLM_BASE_URL``.
        api_key: Explicit API key for the ``openai`` provider; defaults to the
            env var named by ``LLM_API_KEY_ENV``.

    Raises:
        ValueError: If the provider is not ``ollama`` or ``openai``.
    """
    prov = (provider or LLM_PROVIDER).lower()
    name = model or runtime.get_override(runtime.LLM_MODEL) or LLM_MODEL
    url = (
        base_url
        if base_url is not None
        else (runtime.get_override(runtime.LLM_BASE_URL) or LLM_BASE_URL)
    )
    api_key = api_key or runtime.get_override(runtime.LLM_API_KEY)
    key = (prov, name, temperature, url, api_key is not None)

    if key in _CACHE:
        return _CACHE[key]

    if prov == "openai":
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {"model": name, "temperature": temperature}
        if url:
            kwargs["base_url"] = url
        if api_key:
            kwargs["api_key"] = api_key
        elif LLM_API_KEY_ENV != "OPENAI_API_KEY":
            kwargs["api_key"] = _resolve_api_key(LLM_API_KEY_ENV)
        llm: BaseChatModel = ChatOpenAI(**kwargs)
    elif prov == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(model=name, temperature=temperature, base_url=OLLAMA_BASE_URL)
    else:
        raise ValueError(
            f"Unsupported LLM provider {prov!r}; expected one of {_SUPPORTED_PROVIDERS}."
        )

    _CACHE[key] = llm
    return llm


def ollama_reachable(timeout: float = 2.0) -> bool:
    """Return True if the Ollama server responds at ``OLLAMA_BASE_URL``.

    Used for preflight checks so the UI/CLI can fail fast with a clear message instead
    of raising a connection error deep inside a RAG call.
    """
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def clear_llm_cache() -> None:
    """Drop cached LLM clients (useful for tests)."""
    _CACHE.clear()
