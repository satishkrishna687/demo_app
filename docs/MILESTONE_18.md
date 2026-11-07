# Milestone 18 — Structure-Aware Chunking Fix

**Status**: ✅ Complete
**Date**: 2026-09-26
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, Python 3.12.13
**Plan reference**: [`RAG_SPECIFICATION.md`](../RAG_SPECIFICATION.md) §5.2–5.4, §9

---

## 1. Objective

The three pairings' chunks looked incomplete and inconsistent. This milestone diagnoses the
cause against the intended design and fixes it.

---

## 2. Diagnosis

The chunkers parsed Docling's **Markdown** with a numeric-only heading regex
(`^(#{1,4}\s*)?(\d+(?:\.\d+)*)\s+(.+)$`). Docling's real structure lives in the
`section_header` labels and `TableItem` objects, so the regex could not see:

- **Annexes** (`Annex A`, `A.1`, …) — not numeric, so all annex content was absorbed into the
  last numeric section.
- **Non-numeric headings** (`Key`, `Figure 1 - …`, term names) — content absorbed.
- **Table boundaries** — giant tables were recursively split into hundreds of fragments.
- **Page furniture** (`© ISO 2019`, `iTeh STANDARD PREVIEW`, `COPYRIGHT PROTECTED DOCUMENT`)
  rendered as `##` headings and leaked into chunks.

### Evidence (before)

| Pairing | Chunks | Distinct sections | Worst section | Max chunk |
|---|---|---|---|---|
| 2 | 3827 | 407 | `8.9.2.2` = **1193** (annex content) | 899 |
| 3 | 1970 | 407 | `8.9.2.2` = 481 | **21,482** |
| 1 | 8495 children | 0 (by design) | — | min chunk = **1 char** |

Pairing 3 graph: 1970 nodes, 955 edges, **1603 isolated (81%)**.

---

## 3. Fix (hybrid approach)

Structure comes from Docling; Markdown is still used for chunk *content*.

- **New** [`ingestion/structure.py`](../ingestion/structure.py): builds an ordered
  `Section`/`Block` outline from `DoclingDocument.iterate_items()` — `section_header` labels,
  `TableItem` objects, and provenance page numbers. Merges Docling's split number/title
  headings (`"3.1.1"` + `"actor"`), normalises annexes (`"Annex A"` → `"A"`), and skips page
  furniture and the TOC (`document_index`).
- **New** [`ingestion/chunk_utils.py`](../ingestion/chunk_utils.py): `pack_units()` (block-aware
  packing, tables atomic) and `split_table_markdown()` (splits oversized tables on row
  boundaries, repeating the header).
- [`ingestion/loader.py`](../ingestion/loader.py): added `load_iso_structures()` returning
  `StructuredDocument` (Markdown `Document` + `sections`); `load_iso_documents()` kept as a
  wrapper.
- All three chunkers now take `list[StructuredDocument]`:
  - Pairing 1: recursive baseline, but drops degenerate chunks and resolves per-child page
    numbers.
  - Pairing 2: section/annex boundaries, per-chunk page numbers, tables kept atomic.
  - Pairing 3: table-first standalone chunks (bounded by `TABLE_MAX_CHARS`), clause chunks,
    broader cross-reference patterns.
- [`graph/builder.py`](../graph/builder.py): added `hierarchy` edges (section → parent section)
  alongside `cross_reference` edges.
- [`config.py`](../config.py): added `TABLE_MAX_CHARS` (default 4000).

---

## 4. Results (after)

| Pairing | Chunks | Distinct sections | Worst section | Max chunk |
|---|---|---|---|---|
| 2 | 2128 | 495 | `8.6.2.1` = 63 | 4018 |
| 3 | 2007 | 495 | `8.6.2.1` = 62 | 3989 |
| 1 | 7825 children | 0 (by design) | — | min chunk = 5 chars |

- Annexes A–K are now separate sections; `8.9.2.2` no longer dominates.
- No section exceeds 100 chunks; no chunk exceeds ~4 KB.
- Pairing 3: 345 table chunks (182 with captions); graph 2007 nodes, **4929 edges**,
  isolated down from 1603 → **700**.
- Page numbers present on 100% of chunks in all pairings.

---

## 5. Verification

- `python -m pytest tests/ -q` → **33 passed** (11 new in
  [`tests/test_structure.py`](../tests/test_structure.py)).
- `python scripts/inspect_chunks.py` → the metrics above.
- Full ingestion + retrieval path validated with deterministic fake embeddings (Chroma +
  BM25 + ensemble + graph expansion), no model loads.

---

## 6. Deviations / notes

- Pairing 2 keeps `content_type="text"` for all chunks (per the spec); tables are still kept
  atomic and bounded.
- Block-based packing does not apply character overlap between packed blocks (overlap is only
  used when splitting an oversized single block). This avoids duplicated content across chunks.
- Front-matter content (Foreword/Introduction) is retained under an `unknown` section; it is
  the only source of the ~2% of chunks without a real `section_number`.
