"""Deterministic tests for embedding text handling (no model loads)."""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from models import embeddings as emb_mod
from models.embeddings import _TableStrippingEmbeddings, _use_prompts, strip_markdown_tables


class _FakeBase(Embeddings):
    def __init__(self) -> None:
        self.docs: list[str] | None = None
        self.query: str | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.docs = list(texts)
        return [[0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query = text
        return [0.0]


# --------------------------------------------------------------------------- strip_markdown_tables
def test_strip_markdown_tables_flattens_rows_and_drops_separators():
    md = "**Table 1 - Demo**\n\n| id | name |\n|---|---|\n| 1 | alpha |\n| 2 | beta |"
    out = strip_markdown_tables(md)
    assert "|---|---|" not in out
    assert "|" not in out
    assert "id name" in out
    assert "1 alpha" in out
    assert "**Table 1 - Demo**" in out  # caption preserved


def test_strip_markdown_tables_leaves_plain_text_unchanged():
    text = "This is a normal paragraph.\n\nWith two lines."
    assert strip_markdown_tables(text) == text


# --------------------------------------------------------------------------- wrapper
def test_wrapper_strips_tables_for_documents_only():
    base = _FakeBase()
    wrapper = _TableStrippingEmbeddings(base, strip_tables=True)

    wrapper.embed_documents(["| a | b |\n|---|---|\n| 1 | 2 |"])
    assert base.docs == ["a b\n1 2"]

    wrapper.embed_query("| a | b |")
    assert base.query == "| a | b |"  # queries are passed through untouched


def test_wrapper_can_disable_stripping():
    base = _FakeBase()
    wrapper = _TableStrippingEmbeddings(base, strip_tables=False)
    wrapper.embed_documents(["| a | b |"])
    assert base.docs == ["| a | b |"]


# --------------------------------------------------------------------------- prompt selection
def test_use_prompts_auto_detects_qwen3(monkeypatch):
    monkeypatch.setattr(emb_mod, "EMBEDDING_USE_PROMPTS", "auto")
    assert _use_prompts("Qwen/Qwen3-Embedding-0.6B") is True
    assert _use_prompts("BAAI/bge-m3") is False


def test_use_prompts_forced(monkeypatch):
    monkeypatch.setattr(emb_mod, "EMBEDDING_USE_PROMPTS", "0")
    assert _use_prompts("Qwen/Qwen3-Embedding-0.6B") is False
    monkeypatch.setattr(emb_mod, "EMBEDDING_USE_PROMPTS", "1")
    assert _use_prompts("BAAI/bge-m3") is True
