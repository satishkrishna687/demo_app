# Milestone 3 — Multi-PDF Ingestion

**Status**: ✅ Complete
**Date**: 2026-09-23
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §6.1 (Loader), §7 (Milestone 3), §13 (Deviations)

---

## 1. Objective

Implement [`ingestion/loader.py`](../ingestion/loader.py) to ingest **all three** ISO 15118 PDFs
in [`Iso/`](../Iso) into LangChain `Document` objects, with:

- Layout-aware parsing and tables preserved as Markdown (Docling).
- A `document_id` per PDF so downstream chunkers/retrievers can scope by document.
- ISO header/footer/page-number/copyright stripping.
- Page-boundary metadata for citations.

---

## 2. Files Created / Modified

| Path | Action | Purpose |
|---|---|---|
| [`ingestion/loader.py`](../ingestion/loader.py) | Created | Multi-PDF loader: convert, clean, cache, attach metadata |
| [`config.py`](../config.py) | Modified | Added `DOCLING_CACHE_DIR` and `DOCLING_BATCH_SIZE` |
| [`.env.example`](../.env.example) | Modified | Documented the two new settings |
| [`.gitignore`](../.gitignore) | Modified | Ignore generated `data/` (Docling JSON cache) |
| [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) | Modified | Added §13.7–§13.8 documenting the deviations |

---

## 3. Design

### 3.1 Conversion
- Drives `docling.document_converter.DocumentConverter` **directly** (not `DoclingLoader`).
- `PdfPipelineOptions`: `do_table_structure=True`, `do_ocr` configurable (default **False**),
  batch sizes set to `DOCLING_BATCH_SIZE` (default 1).
- Accelerator device mapped from [`models.device.get_device()`](../models/device.py):
  `cuda` → `AcceleratorDevice.CUDA`, `mps` → `MPS`, else `CPU`.

### 3.2 Caching
- Each converted `DoclingDocument` is serialized to `data/parsed/{document_id}.json`
  (`DOCLING_CACHE_DIR`). Re-runs load the JSON and skip conversion.
- Corrupt/incompatible cache is detected and triggers re-conversion.

### 3.3 One Document per PDF (section integrity)
- The whole PDF becomes a **single** `Document`. This is deliberate: section-based chunking
  (Pairings 2 and 3) must see sections intact, so splitting per page would break clause
  boundaries.
- Page boundaries are recorded as character offsets (`page_breaks`) derived from Docling's
  `page_break_placeholder`. [`page_for_offset()`](../ingestion/loader.py) maps any chunk offset
  back to a 1-based page number.

### 3.4 Metadata attached
| Key | Meaning |
|---|---|
| `document_id` | e.g. `ISO-15118-2` (from filename stem) |
| `source_file` | e.g. `ISO-15118-2.pdf` |
| `page_number` | `None` at document level; resolve per chunk via `page_for_offset()` |
| `num_pages` | Page count |
| `page_breaks` | Character offsets where each page starts |

### 3.5 Cleaning
`strip_iso_headers_footers()` removes `Page X of Y`, standalone page numbers, and
`© ISO YYYY ... All rights reserved` lines, then collapses excess blank lines.

---

## 4. Commands Executed

```bash
# API discovery (installed versions differ from the plan)
.venv/bin/python -c "import inspect; from langchain_docling import DoclingLoader; ..."
.venv/bin/python -c "from docling.datamodel.pipeline_options import PdfPipelineOptions; ..."
.venv/bin/python -c "from docling_core.types.doc.document import DoclingDocument as DD; inspect.signature(DD.export_to_markdown)"

# Single-file smoke test
.venv/bin/python -c "from ingestion.loader import load_pdf; load_pdf('Iso/ISO-15118-1.pdf')"

# Full corpus
.venv/bin/python -c "from ingestion.loader import load_iso_documents; load_iso_documents('Iso')"
```

---

## 5. Verification Results

### 5.1 Full corpus
```
[53.0s] total documents: 3
  ISO-15118-1    pages= 15 chars= 190987 table_rows=  89 page_breaks=14
  ISO-15118-2    pages=348 chars=1100291 table_rows=1997 page_breaks=347
  ISO-15118-3    pages= 86 chars= 262844 table_rows= 471 page_breaks=85
distinct document_ids: True ['ISO-15118-1', 'ISO-15118-2', 'ISO-15118-3']
page_for_offset sample: 165
```

### 5.2 Cache
```
data/parsed/ISO-15118-1.json   379K
data/parsed/ISO-15118-2.json    15M
data/parsed/ISO-15118-3.json   2.0M
```
Second run loaded all three from cache (ISO-15118-1 and -2 were instant; only ISO-15118-3
needed conversion on the first full run).

### 5.3 Device usage
Docling logged `Accelerator device: 'cuda:0'` — layout and table-structure models ran on the
RTX 3050.

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **OOM kill on large PDF** | Process `Killed` while converting ISO-15118-2 (5.4 MB, 350 pages) | Docling with OCR + table structure + default batch sizes exceeded the 7.7 GB RAM limit | Disabled OCR by default and set `ocr/layout/table_batch_size=1` | ✅ Resolved — converted in 259 s |
| 2 | **`DoclingLoader` API mismatch** | Plan's `pipeline_options=` kwarg does not exist | langchain-docling 3.0.0 exposes `converter`/`convert_kwargs`/`export_type` instead | Used `DocumentConverter` directly for full control | ✅ Resolved |
| 3 | **No page metadata from MARKDOWN export** | `DoclingLoader(export_type=MARKDOWN)` returned 1 doc with only `source`; `dl_meta` was `None` | MARKDOWN export does not emit provenance | Derived page boundaries via `export_to_markdown(page_break_placeholder=...)` + `page_for_offset()` | ✅ Resolved |
| 4 | RapidOCR CUDA unavailable | `CUDAExecutionProvider is not in available providers` | `onnxruntime` (CPU) installed, not `onnxruntime-gpu` | OCR disabled by default (ISO PDFs have a text layer); documented for scanned docs | ⚠️ Non-blocking |
| 5 | HF Hub rate-limit warning | `unauthenticated requests to the HF Hub` | No `HF_TOKEN` | Documented (plan §13.6) | ⚠️ Non-blocking |
| 6 | Slow first conversion | ISO-15118-2 took 259 s | Large PDF, CPU-bound table structure | JSON cache makes subsequent runs fast | ✅ Resolved (cached) |

### 6.1 Failure #1 in detail (the significant one)

The first full-corpus run was killed by the Linux OOM killer during ISO-15118-2:

```
INFO Processing document ISO-15118-2.pdf
Killed
```

The machine has **7.7 GiB RAM** (4.8 GiB available) and 2 GiB swap. Docling's default pipeline
with OCR enabled and default batch sizes (4) peaked above available memory. Two changes fixed it:

1. `do_ocr = False` — the ISO PDFs have an embedded text layer, so OCR is unnecessary and is the
   dominant memory consumer.
2. `ocr_batch_size = layout_batch_size = table_batch_size = 1` — bounds peak memory.

With these, ISO-15118-2 converted successfully (350 pages, 1.1 M chars) and ISO-15118-3 in 42 s.

**Cross-platform note**: Apple Silicon M1 machines typically have 8–16 GB unified memory, so the
same low-memory defaults are appropriate there. On a machine with more RAM, `DOCLING_BATCH_SIZE`
can be raised via `.env`.

---

## 7. Deviations from the Plan

1. **`load_iso_document(pdf_path, use_ocr=True)` → `load_iso_documents(pdf_dir, use_ocr=False)`** —
   multi-PDF directory input, and OCR default flipped to `False` (memory). Documented in plan §13.7.
2. **`DoclingLoader` → `DocumentConverter`** — the installed langchain-docling API differs; direct
   converter use also enables page-boundary extraction.
3. **One Document per PDF, not per page** — required to keep sections intact for Pairings 2/3.
   Page numbers are resolved per chunk via `page_for_offset()` instead of a document-level
   `page_number`. Documented in plan §13.8.
4. **New config**: `DOCLING_CACHE_DIR`, `DOCLING_BATCH_SIZE`.
5. **`data/` added to `.gitignore`** — the Docling JSON cache is a generated artifact.

---

## 8. Acceptance Check (Milestone 3)

- [x] `ingestion/loader.py` implemented for all three PDFs.
- [x] Document count > 0 per file (exactly 1 per PDF).
- [x] `document_id` present and distinct across the three parts.
- [x] Tables preserved as Markdown (2,557 table rows total).
- [x] ISO headers/footers/page numbers stripped.
- [x] Page boundaries captured (`page_breaks`) and resolvable (`page_for_offset`).
- [x] Conversion cache written and reused.
- [x] Failures captured with mitigations (this document).
- [ ] MPS (Apple Silicon) verification — pending M1 hardware.

---

## 9. Next Step

**Milestone 4 — Pairing 1**: implement [`ingestion/chunking_pair1.py`](../ingestion/chunking_pair1.py)
(recursive parent/child chunks), [`retrieval/pair1_retriever.py`](../retrieval/pair1_retriever.py)
(`ParentDocumentRetriever` from `langchain_classic`), and
[`chain/rag_chain.py`](../chain/rag_chain.py) (LCEL chain using `get_llm()`). Test: every child
has a `parent_id` present in the parent set, and answers cite sections.
