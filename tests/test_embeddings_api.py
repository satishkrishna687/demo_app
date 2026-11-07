"""Deterministic tests for the API (OpenAI-compatible) embeddings path.

No network calls: the OpenAI backend is replaced by a fake via monkeypatching, so
these tests exercise provider selection, runtime overrides, normalisation, and
table stripping only.
"""

from __future__ import annotations

import math

import pytest
from langchain_core.embeddings import Embeddings

from models import embeddings as emb_mod
from models import runtime
from models.embeddings import _NormalizedEmbeddings, get_embeddings


class _FakeOpenAIEmbeddings(Embeddings):
    """Stand-in for ``OpenAIEmbeddings`` returning unnormalised constant vectors."""

    def __init__(self, **kwargs) -> None:  # accept any OpenAIEmbeddings kwargs
        self.model = kwargs.get("model", "")
        self.base_url = kwargs.get("base_url", "")
        self.api_key = kwargs.get("api_key", "")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[3.0, 4.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [3.0, 4.0]


@pytest.fixture(autouse=True)
def _clean_runtime_and_cache():
    runtime.clear_overrides()
    emb_mod.clear_embeddings_cache()
    yield
    runtime.clear_overrides()
    emb_mod.clear_embeddings_cache()


def _patch_openai_backend(monkeypatch):
    """Make the 'openai' provider branch construct the fake backend."""

    class _FakeModule:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self):  # pragma: no cover - not used
            raise AssertionError

    def _factory(**kwargs):
        return _FakeOpenAIEmbeddings(**kwargs)

    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings", _factory, raising=False
    )


# --------------------------------------------------------------------------- provider switch
def test_openai_provider_builds_api_backend(monkeypatch):
    _patch_openai_backend(monkeypatch)
    emb = get_embeddings(model_name="qwen/qwen3-embedding-8b", provider="openai", api_key="k")
    # The wrapper chain is TableStripping(Normalized(Fake)); verify normalisation.
    vec = emb.embed_query("hello")
    assert math.isclose(math.sqrt(vec[0] ** 2 + vec[1] ** 2), 1.0, rel_tol=1e-9)


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embeddings(provider="bogus")


def test_runtime_override_selects_provider_and_model(monkeypatch):
    _patch_openai_backend(monkeypatch)
    runtime.set_overrides(embedding_provider="openai", embedding_model="qwen/qwen3-embedding-8b")
    emb = get_embeddings(api_key="k")
    vec = emb.embed_query("hello")
    assert len(vec) == 2  # came from the fake API backend, not a local model


def test_normalization_wrapper_normalizes_documents_and_queries():
    base = _FakeOpenAIEmbeddings()
    norm = _NormalizedEmbeddings(base)
    docs = norm.embed_documents(["a", "b"])
    for vec in docs:
        assert math.isclose(math.hypot(*vec), 1.0, rel_tol=1e-9)
    q = norm.embed_query("q")
    assert math.isclose(math.hypot(*q), 1.0, rel_tol=1e-9)


def test_empty_chunks_substituted_for_encoder_input(monkeypatch):
    """OpenAI-compatible endpoints reject empty strings; they must be substituted."""
    _patch_openai_backend(monkeypatch)

    captured: dict = {}

    class _Capturing(_FakeOpenAIEmbeddings):
        def embed_documents(self, texts):
            captured["texts"] = list(texts)
            return super().embed_documents(texts)

    def _factory(**kwargs):
        return _Capturing(**kwargs)

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", _factory, raising=False)

    emb = get_embeddings(provider="openai", api_key="k")
    emb.embed_documents(["", "   ", "| a | b |\n|---|---|", "real text"])
    assert captured["texts"] == [" ", " ", "a b", "real text"]
    # Query path is guarded too.
    emb.embed_query("")
    assert True  # no exception raised


def test_retry_wrapper_retries_rate_limits_then_succeeds():
    from models.embeddings import _RetryEmbeddings

    calls = {"n": 0}

    class _Flaky(Embeddings):
        def embed_documents(self, texts):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("Error code: 429 - rate_limit_error, retry after 1 sec")
            return [[1.0]]

        def embed_query(self, text):
            return [1.0]

    wrapper = _RetryEmbeddings(_Flaky())
    assert wrapper.embed_documents(["x"]) == [[1.0]]
    assert calls["n"] == 3


def test_retry_wrapper_reraises_non_rate_limit_errors():
    from models.embeddings import _RetryEmbeddings

    class _Broken(Embeddings):
        def embed_documents(self, texts):
            raise RuntimeError("connection refused")

        def embed_query(self, text):
            return [1.0]

    with pytest.raises(RuntimeError, match="connection refused"):
        _RetryEmbeddings(_Broken()).embed_documents(["x"])


def test_api_path_still_strips_tables(monkeypatch):
    _patch_openai_backend(monkeypatch)

    captured: dict = {}

    class _Capturing(_FakeOpenAIEmbeddings):
        def embed_documents(self, texts):
            captured["texts"] = list(texts)
            return super().embed_documents(texts)

    def _factory(**kwargs):
        return _Capturing(**kwargs)

    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", _factory, raising=False)

    emb = get_embeddings(provider="openai", api_key="k")
    emb.embed_documents(["| a | b |\n|---|---|\n| 1 | 2 |"])
    assert captured["texts"] == ["a b\n1 2"]  # stored text cleaning still applies
