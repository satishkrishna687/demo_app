# Milestone 19 — Embedding & Reranker Quality Fixes

**Status**: ✅ Complete
**Date**: 2026-09-26
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, Python 3.12.13
**Plan reference**: [`OPEN_SOURCE_MODELS.md`](../OPEN_SOURCE_MODELS.md), [`RAG_SPECIFICATION.md`](../RAG_SPECIFICATION.md) §2

---

## 1. Objective

Improve retrieval quality without changing the model stack, by using the embedding model
correctly and removing two known weaknesses in how text is fed to the encoder/reranker.

---

## 2. Changes

### 2.1 Asymmetric query instruction (Qwen3-Embedding)

Qwen3-Embedding is instruction-aware and ships built-in prompts in its
`config_sentence_transformers.json`:

```json
{ "prompts": { "query": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:", "document": "" } }
```

The factory previously encoded queries and documents identically, so the instruction was
never applied. [`get_embeddings()`](../models/embeddings.py:118) now passes
`query_encode_kwargs={"prompt_name": "query"}` and
`encode_kwargs={"prompt_name": "document"}`; `HuggingFaceEmbeddings` applies the former to
queries and the latter to documents, giving the intended asymmetry.

- Controlled by `EMBEDDING_USE_PROMPTS` (`auto` default → enabled only for Qwen3-Embedding;
  `1`/`0` force on/off). Set `0` when switching to a model without built-in prompts.

### 2.2 Markdown-table flattening before embedding

Markdown table syntax (pipes, separator rows) adds noise tokens and dilutes the embedding.
[`strip_markdown_tables()`](../models/embeddings.py:52) drops separator rows and replaces
cell pipes with spaces. It is applied inside a thin
[`_TableStrippingEmbeddings`](../models/embeddings.py:83) wrapper around the backend, so:

- the **stored** document text (what the LLM sees) is unchanged — Chroma stores it separately;
- only the text sent to the encoder is cleaned;
- queries are passed through untouched.

Controlled by `EMBEDDING_STRIP_TABLES` (default on).

### 2.3 Reranker max length

The cross-encoder was capped at `max_length=512`, which truncated the (now larger) table
chunks. [`get_reranker()`](../models/reranker.py:22) now uses `RERANKER_MAX_LENGTH`
(default **1024**; `bge-reranker-v2-m3` supports up to 8192).

### 2.4 A/B configuration

All knobs are environment-driven, so models can be compared without code changes:

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Swap embedding model |
| `EMBEDDING_USE_PROMPTS` | `auto` | Apply built-in query/document prompts |
| `EMBEDDING_QUERY_PROMPT_NAME` | `query` | Prompt name for queries |
| `EMBEDDING_DOC_PROMPT_NAME` | `document` | Prompt name for documents |
| `EMBEDDING_STRIP_TABLES` | `1` | Flatten Markdown tables before embedding |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Swap reranker |
| `RERANKER_MAX_LENGTH` | `1024` | Cross-encoder max sequence length |

---

## 3. Verification

- `python -m pytest tests/ -q` → **39 passed** (6 new in
  [`tests/test_embeddings.py`](../tests/test_embeddings.py)).
- Real-model check: `embed_query` and `embed_documents` both return 1024-dim vectors and
  differ for the same text, confirming the query instruction is applied to queries only.
- `strip_markdown_tables("| a | b |\n|---|---|\n| 1 | 2 |")` → `"a b\n1 2"`.

---

## 4. Notes / next steps

- Existing Chroma indexes were built with the old (symmetric, un-stripped) embeddings and
  must be rebuilt for the new behaviour to take effect. Delete `data/indexes/pairN/` and
  `chroma_db/`, or run the eval/UI which rebuilds them.
- To A/B a larger embedding (e.g. `Qwen/Qwen3-Embedding-4B`), set `EMBEDDING_MODEL` and
  re-run `python -m eval.run_pairing --pairing N`; compare Recall@5 / nDCG@5. On an 8 GB GPU
  the reranker may need `DEVICE=cpu` to fit alongside a 4B embedding.
