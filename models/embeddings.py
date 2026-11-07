"""Embedding model factory.

Uses a local sentence-transformers model by default so the corpus stays private and no
API key is required. The compute device is resolved automatically for the host platform
(cuda on Ubuntu + NVIDIA, mps on Apple Silicon, else cpu) via :func:`models.device.get_device`.

Two quality measures are applied on top of the raw model:

- **Asymmetric prompts**: Qwen3-Embedding is instruction-aware and ships built-in prompts
  (``query`` = a retrieval instruction, ``document`` = empty). ``HuggingFaceEmbeddings``
  applies ``query_encode_kwargs`` to queries and ``encode_kwargs`` to documents, so the
  instruction is added to queries only. Controlled by ``EMBEDDING_USE_PROMPTS``.
- **Table flattening**: Markdown table syntax (pipes/separators) is stripped from the text
  sent to the encoder, which reduces noise tokens. The *stored* document text is untouched
  (Chroma stores it separately), so the LLM still sees the full Markdown table.

Example:
    from models.embeddings import get_embeddings
    embeddings = get_embeddings()
    vector = embeddings.embed_query("maximum charging voltage")

To swap providers (e.g. OpenAI ``text-embedding-3-small``), change :func:`get_embeddings`
to construct a ``langchain_openai.OpenAIEmbeddings`` instead — every consumer of this
factory is provider-agnostic.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.embeddings import Embeddings

from config import (
    EMBEDDING_API_KEY_ENV,
    EMBEDDING_BASE_URL,
    EMBEDDING_DOC_PROMPT_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    EMBEDDING_QUERY_PROMPT_NAME,
    EMBEDDING_STRIP_TABLES,
    EMBEDDING_USE_PROMPTS,
)
from models import runtime
from models.device import get_device

# Cache keyed by (model_name, device) so repeated pipeline builds do not reload weights.
_CACHE: dict[tuple[str, str], Any] = {}

_MULTISPACE_RE = re.compile(r"[ \t]{2,}")


def _is_separator_row(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= set("|-: ")


def strip_markdown_tables(text: str) -> str:
    """Flatten Markdown tables to plain text for embedding.

    Separator rows (``|---|---|``) are dropped and cell pipes are replaced with spaces, so
    the encoder sees the cell contents without pipe/dash noise. Non-table lines are kept
    verbatim.
    """
    out: list[str] = []
    for line in text.splitlines():
        if _is_separator_row(line):
            continue
        if line.lstrip().startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            out.append(" ".join(cell for cell in cells if cell))
        else:
            out.append(line)
    return _MULTISPACE_RE.sub(" ", "\n".join(out)).strip()


def _use_prompts(model_name: str) -> bool:
    """Decide whether to apply the model's built-in query/document prompts."""
    setting = EMBEDDING_USE_PROMPTS
    if setting in ("1", "true", "yes", "on"):
        return True
    if setting in ("0", "false", "no", "off"):
        return False
    return "qwen3-embedding" in model_name.lower()


class _TableStrippingEmbeddings(Embeddings):
    """Wrap an embeddings backend, flattening Markdown tables before embedding documents.

    Only the text sent to the encoder is cleaned; the stored document text is unchanged.
    Queries are passed through untouched (their prompt is handled by the backend).
    """

    def __init__(self, base: Embeddings, strip_tables: bool = True) -> None:
        self._base = base
        self._strip = strip_tables

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._strip:
            texts = [strip_markdown_tables(text) for text in texts]
        # OpenAI-compatible endpoints reject empty input strings; the local
        # sentence-transformers path tolerated them. Substitute a space for
        # encoder input only — stored document text is untouched.
        texts = [t if t.strip() else " " for t in texts]
        return self._base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._base.embed_query(text if text.strip() else " ")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._strip:
            texts = [strip_markdown_tables(text) for text in texts]
        return await self._base.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._base.aembed_query(text)


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


def get_embeddings(
    model_name: str | None = None,
    device: str | None = None,
    provider: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Return a cached embeddings object.

    Args:
        model_name: Override the configured ``EMBEDDING_MODEL``.
        device: Override the auto-detected device (``cuda`` / ``mps`` / ``cpu``).
            Ignored for the API provider.
        provider: Override ``EMBEDDING_PROVIDER`` (``local`` or ``openai``).
        api_key: Explicit API key for the ``openai`` provider; defaults to the
            env var named by ``EMBEDDING_API_KEY_ENV``.

    Returns:
        A LangChain embeddings object exposing ``embed_documents`` / ``embed_query``.
    """
    name = model_name or runtime.get_override(runtime.EMBEDDING_MODEL) or EMBEDDING_MODEL
    prov = (provider or runtime.get_override(runtime.EMBEDDING_PROVIDER) or EMBEDDING_PROVIDER).lower()
    dev = device or get_device()
    api_key = api_key or runtime.get_override(runtime.EMBEDDING_API_KEY)
    key = (prov, name, dev, api_key is not None)

    if key not in _CACHE:
        if prov == "openai":
            from langchain_openai import OpenAIEmbeddings

            base = _RetryEmbeddings(
                OpenAIEmbeddings(
                    model=name,
                    base_url=EMBEDDING_BASE_URL,
                    api_key=api_key or _resolve_api_key(EMBEDDING_API_KEY_ENV),
                    # Small batches: free-tier embedding endpoints (e.g. Nebius via
                    # OpenRouter) enforce tight per-minute token quotas; large
                    # batches trigger 429s that would abort a whole index build.
                    chunk_size=64,
                    # OpenAI-compatible endpoints may return unnormalized vectors;
                    # normalize explicitly so cosine similarity behaves like the
                    # local sentence-transformers path.
                    check_embedding_ctx_length=False,
                )
            )
            _CACHE[key] = _TableStrippingEmbeddings(
                _NormalizedEmbeddings(base), strip_tables=EMBEDDING_STRIP_TABLES
            )
        elif prov == "local":
            from langchain_huggingface import HuggingFaceEmbeddings

            encode_kwargs: dict[str, Any] = {"normalize_embeddings": True}
            query_encode_kwargs: dict[str, Any] = {"normalize_embeddings": True}
            if _use_prompts(name):
                encode_kwargs["prompt_name"] = EMBEDDING_DOC_PROMPT_NAME
                query_encode_kwargs["prompt_name"] = EMBEDDING_QUERY_PROMPT_NAME

            base = HuggingFaceEmbeddings(
                model_name=name,
                model_kwargs={"device": dev},
                encode_kwargs=encode_kwargs,
                query_encode_kwargs=query_encode_kwargs,
            )
            _CACHE[key] = _TableStrippingEmbeddings(base, strip_tables=EMBEDDING_STRIP_TABLES)
        else:
            raise ValueError(
                f"Unsupported embedding provider {prov!r}; expected 'local' or 'openai'."
            )

    return _CACHE[key]


class _RetryEmbeddings(Embeddings):
    """Retry wrapper for rate-limited embedding endpoints.

    Free/quoted-tier endpoints (e.g. Nebius-served models via OpenRouter) return
    HTTP 429 with "retry after ~1s" messages. The OpenAI SDK's built-in retries
    (2 attempts, sub-second backoff) give up too early during bulk indexing, so
    this wrapper retries with exponential backoff.
    """

    # Backoff schedule in seconds; ~2 minutes total worst case per call.
    _BACKOFF = (2, 5, 15, 45, 90)

    def __init__(self, base: Embeddings) -> None:
        self._base = base

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        text = str(exc).lower()
        return "429" in text or "rate limit" in text or "rate_limit" in text

    def _with_retry(self, fn, *args):
        import time

        last: Exception | None = None
        for wait in (0, *self._BACKOFF):
            if wait:
                time.sleep(wait)
            try:
                return fn(*args)
            except Exception as exc:  # noqa: BLE001 - retry any transport error
                last = exc
                if not self._is_rate_limit(exc):
                    raise
        raise last  # type: ignore[misc]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._with_retry(self._base.embed_documents, texts)

    def embed_query(self, text: str) -> list[float]:
        return self._with_retry(self._base.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._base.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._base.aembed_query(text)


class _NormalizedEmbeddings(Embeddings):
    """L2-normalize vectors from an OpenAI-compatible embeddings endpoint.

    The local sentence-transformers path normalizes via ``normalize_embeddings``;
    API endpoints do not guarantee it, so we normalize here to keep cosine
    similarity semantics identical across providers.
    """

    def __init__(self, base: Embeddings) -> None:
        self._base = base

    @staticmethod
    def _normalize(vectors: list[list[float]]) -> list[list[float]]:
        import math

        out: list[list[float]] = []
        for vec in vectors:
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / norm for x in vec])
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._normalize(self._base.embed_documents(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._normalize([self._base.embed_query(text)])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._normalize(await self._base.aembed_documents(texts))

    async def aembed_query(self, text: str) -> list[float]:
        return self._normalize([await self._base.aembed_query(text)])[0]


def clear_embeddings_cache() -> None:
    """Drop cached embedding models (useful for tests)."""
    _CACHE.clear()
