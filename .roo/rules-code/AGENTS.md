# Project Coding Rules (Non-Obvious Only)

- Never import `torch` before `config` — [`config.py`](config.py:14) sets `TORCH_DISABLE_NATIVE_JIT` / `PYTORCH_ENABLE_MPS_FALLBACK` at import; torch caches this state. `models/device.py` and `ingestion/loader.py` re-set them defensively.
- ALL vector-store writes go through [`add_documents_in_batches()`](vectorstore/store.py:36). It enforces Chroma's ~5461 upsert cap and strips `None`/empty-list metadata Chroma rejects. Never call `vectorstore.add_documents()` directly.
- Chunkers take `list[StructuredDocument]` from [`load_iso_structures()`](ingestion/loader.py:198). Structure (headings, annexes, tables, page numbers) comes from Docling labels via [`ingestion/structure.py`](ingestion/structure.py:1) — never parse Markdown headings with a regex (it misses annexes/non-numeric headings and creates giant sections).
- Keep tables atomic: use [`pack_units()`](ingestion/chunk_utils.py:1) + [`split_table_markdown()`](ingestion/chunk_utils.py:20); a generic splitter fragments tables mid-row.
- Adding a pairing means three coordinated edits: `ingestion/chunking_pairN.py`, `retrieval/pairN_retriever.py`, and a branch in [`build_pipeline()`](ui/pipeline_registry.py:39) plus a name in `PIPELINE_NAMES`.
- Every module starts with `from __future__ import annotations` (Python 3.12).
- Any module under `ui/` that imports repo-root packages must inject the repo root into `sys.path` first (see [`ui/app.py`](ui/app.py:17)); Streamlit puts `ui/` on `sys.path`, not the root.
- Do not switch [`ingestion/loader.py`](ingestion/loader.py:1) to Docling's `ExportType.MARKDOWN` — it returns one doc per PDF with no page metadata. The loader drives `DocumentConverter` directly to keep each PDF as ONE `Document` with `\f` page boundaries; section chunking depends on this.
- LLM clients are cached in [`models/llm.py`](models/llm.py:20); call `clear_llm_cache()` in tests that change provider/model.
- Embeddings are wrapped in [`_TableStrippingEmbeddings`](models/embeddings.py:83): queries get Qwen3's built-in prompt, documents get Markdown tables flattened. Never strip tables from the stored `page_content` — only the encoder input is cleaned. Changing embedding settings invalidates existing Chroma indexes.
- Tests must stay fast and deterministic — no model loads, no PDF parsing (see `tests/`). Use fake Docling items (see [`tests/test_structure.py`](tests/test_structure.py:1)) and fake embeddings to test chunking/retrieval.
