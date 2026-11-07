# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Commands
- Tests: `python -m pytest tests/ -q` (fast, deterministic; no model loads). No pytest config file exists.
- Run UI from the repo root only: `streamlit run ui/app.py` (ui modules inject the repo root into `sys.path`; running from inside `ui/` breaks `import config`).
- Eval CLIs: `python -m eval.build_gold_set`, `python -m eval.run_pairing --pairing N --limit 3`, `python -m eval.comparison_report`.
- There is no `main.py` yet (planned Milestone 14); use the eval/UI entry points above.

## Non-obvious constraints
- `config.py` sets `TORCH_DISABLE_NATIVE_JIT=1` and `PYTORCH_ENABLE_MPS_FALLBACK=1` at import time; these MUST be set before torch is imported. `models/device.py` and `ingestion/loader.py` re-set them defensively. Never import torch before `config`.
- ALL vector-store writes must go through [`add_documents_in_batches()`](vectorstore/store.py:36) — it enforces Chroma's ~5461 upsert cap and strips `None`/empty-list metadata that Chroma rejects. Do not call `vectorstore.add_documents()` directly.
- Chunkers consume `StructuredDocument` objects from [`load_iso_structures()`](ingestion/loader.py:198), NOT raw Markdown. Section boundaries, tables, and page numbers come from Docling's structured labels via [`ingestion/structure.py`](ingestion/structure.py:1). Do NOT reintroduce regex-over-Markdown heading detection — it misses annexes and non-numeric headings and collapses the document into giant sections.
- Tables must stay atomic: use [`pack_units()`](ingestion/chunk_utils.py:1) and [`split_table_markdown()`](ingestion/chunk_utils.py:20) rather than a generic text splitter, or tables get fragmented mid-row.
- `ingestion/loader.py` deliberately drives Docling's `DocumentConverter` directly (not `ExportType.MARKDOWN`) so each PDF stays ONE `Document` with page boundaries derived from a `\f` placeholder. Section-based chunking depends on sections not being split across pages.
- Device resolution is `cuda > mps > cpu`, overridable via `DEVICE` env var (see [`get_device()`](models/device.py:27)).
- Reranking (Pairing 3) is `try/except`-guarded and degrades to unranked results instead of crashing on GPU OOM.
- LLM clients are cached in [`models/llm.py`](models/llm.py:20); call `clear_llm_cache()` in tests. Embeddings/reranker are always local; only the chat LLM switches via `LLM_PROVIDER` (`ollama` default, `openai` optional).
- Embeddings apply Qwen3's built-in query prompt to queries only and flatten Markdown tables before encoding (stored chunk text is unchanged) — see [`models/embeddings.py`](models/embeddings.py:1). Changing `EMBEDDING_MODEL` / `EMBEDDING_USE_PROMPTS` / `EMBEDDING_STRIP_TABLES` changes the vectors, so existing Chroma indexes must be rebuilt.
- RAGAS returns `nan` with the small local judge; the custom judge in [`eval/judge.py`](eval/judge.py) is the default. Uncomputable metrics are recorded as `None` and surfaced in `flags`, never dropped.
- Index freshness is a count heuristic in [`ensure_indexed()`](vectorstore/store.py:36): if the collection count equals the document count, the index is reused without re-embedding. Delete the persist directory to force a rebuild after changing chunking/embedding settings.
- Graph node ids embed `document_id` + content hash (`{doc}:{section}_{md5[:8]}`) because identical clause numbers exist across the three ISO parts; cross-reference edges are scoped to one `document_id` and fan-out is capped at 20 targets per ref (see [`graph/builder.py`](graph/builder.py:28)).
- Pipelines are built one at a time, not all three — building all three was OOM-killed on 8 GB (see [`ui/pipeline_registry.py`](ui/pipeline_registry.py:1) and docs/MILESTONE_7.md).
- The judge prompt demands a bare number; the first numeric token is clamped to [0,1], one retry, then `None` (flagged). Citation accuracy is deterministic regex matching of "Section/Clause N" against retrieved sections — no judge LLM.
- Model/API-key overrides from the UI flow through [`models/runtime.py`](models/runtime.py) (in-memory only, never persisted); factories resolve explicit args > runtime overrides > config defaults. `get_pipeline()` caches on the runtime signature, so changed settings rebuild the pipeline.
- `CLOUD_MODE=1` skips Docling entirely: chunks are recovered from the shipped Chroma index via `_load_chunks_from_store()` in [`ui/pipeline_registry.py`](ui/pipeline_registry.py). Pairing 1 is unsupported in cloud mode (no persisted vector store). Cloud deps live in `requirements-cloud.txt` (chromadb pinned for index compatibility).
- Embeddings support an OpenAI-compatible provider (`EMBEDDING_PROVIDER=openai`, e.g. OpenRouter `qwen/qwen3-embedding-8b`); API vectors are L2-normalized by `_NormalizedEmbeddings` to match the local path's cosine semantics. Switching provider/model requires a full index rebuild.

## Conventions
- Python 3.12; every module starts with `from __future__ import annotations`.
- PyTorch is installed separately per-OS (`requirements-macos.txt` / `requirements-linux-cuda.txt`), not via `requirements.txt`.
- Runtime artifacts are git-ignored: `data/parsed/` (Docling JSON cache), `data/indexes/pairN/`, `chroma_db/`.
- Three "pairings" are the core abstraction: chunking in `ingestion/chunking_pairN.py` (all take `list[StructuredDocument]`), retrieval in `retrieval/pairN_retriever.py`, registered in [`ui/pipeline_registry.py`](ui/pipeline_registry.py:24).
- `scripts/inspect_chunks.py` is the diagnostic for chunk quality (counts, section coverage, oversized sections, table/graph stats); run it after any chunking change.
