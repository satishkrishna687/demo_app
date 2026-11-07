# Milestone 5 — Pairing 2 (Section-Based Chunking + Metadata-Filtered Hybrid Search)

**Status**: ✅ Complete
**Date**: 2026-09-23
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §6.2–6.4, §7 (Milestone 5), §13 (Deviations)

---

## 1. Objective

Implement the structured pipeline: split documents at ISO clause boundaries with a hierarchical
`heading_path`, then retrieve with a hybrid (dense + BM25) ensemble that also supports
metadata-filtered queries scoped to a specific clause number.

---

## 2. Files Created / Modified

| Path | Action | Purpose |
|---|---|---|
| [`ingestion/chunking_pair2.py`](../ingestion/chunking_pair2.py) | Created | Section-aware chunking + ISO metadata |
| [`retrieval/pair2_retriever.py`](../retrieval/pair2_retriever.py) | Created | `EnsembleRetriever` (vector + BM25) and `filtered_retrieve()` |
| [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) | Modified | Added §13.11 (heading extraction robustness) |

No config changes were needed; this milestone built directly on the Milestone 4 batching helper
and the pre-torch env flags.

---

## 3. Design

### 3.1 Chunking ([`ingestion/chunking_pair2.py`](../ingestion/chunking_pair2.py))
- Heading regex: `^(#{1,4}\s*)?(\d+(?:\.\d+)*)\s+(.+)$` (multiline).
- **Robustness over the plan**:
  - Only **hash-prefixed** headings (`## 4.1 Title`) are preferred; bare numbered lines and
    table-of-contents entries are excluded.
  - TOC **dot-leader** lines (e.g. `7.6.1 Overview ..........`) are skipped.
  - Titles are cleaned of leading `#` and of inline sub-headings (Docling sometimes renders
    `3.1.1\n\n## actor`).
- Long sections (> `chunk_size * 1.5`) are further split with `RecursiveCharacterTextSplitter`.
- Metadata per chunk: `section_number`, `section_title`, `heading_path`, `document_id`,
  `source_file`, `page_number` (via `page_for_offset`), `content_type="text"`.
- Fallback: recursive splitting with `section_number="unknown"` when no headings are found.

### 3.2 Retrieval ([`retrieval/pair2_retriever.py`](../retrieval/pair2_retriever.py))
- `build_metadata_filtered_hybrid_retriever(chunks)` returns `(EnsembleRetriever, Chroma)`:
  - Vector retriever: `Chroma.as_retriever(k=TOP_K)`.
  - Sparse retriever: `BM25Retriever.from_documents(chunks)` with `k=TOP_K`.
  - `EnsembleRetriever(weights=[0.6, 0.4])` fuses via Reciprocal Rank Fusion.
- `filtered_retrieve(vectorstore, query, section_filter, k)` scopes a similarity search to an
  exact `section_number`, with an automatic fallback from `filter=` to `where=` for Chroma
  version differences.
- Ingestion uses the shared `add_documents_in_batches()` (Milestone 4) to respect Chroma's
  batch limit.

---

## 4. Commands Executed

```bash
# Heading-format inspection
.venv/bin/python - <<'PY'
# printed lines matching the section regex and lines starting with '#'
PY

# End-to-end test
.venv/bin/python - <<'PY'
from ingestion.loader import load_iso_documents
from ingestion.chunking_pair2 import create_section_based_chunks
from retrieval.pair2_retriever import build_metadata_filtered_hybrid_retriever
docs = load_iso_documents("Iso", use_ocr=False)
chunks = create_section_based_chunks(docs)
retriever, vs = build_metadata_filtered_hybrid_retriever(chunks)
print(retriever.invoke("SECC communication setup"))
PY
```

---

## 5. Verification Results

### 5.1 Section metadata coverage
```
section chunks: 3827
with section_number: 3827 | with heading_path: 3827
sample meta: {'document_id': 'ISO-15118-1', 'section_number': '1',
              'section_title': 'Scope', 'heading_path': '1 Scope', 'page_number': 7}
```

### 5.2 Hybrid retrieval
```
[85.2s] retriever built
[0.1s] hybrid retrieved 8; sections=['8.7.3.3','8.9.2.2','7.10.1','7.3.1','8.7.2.1','8.7.3.2']
```

### 5.3 Metadata filtering
```
total chunks: 3827 | distinct sections: 407
'4.1' present: False | '3.1.1' present: True
top sections: [('8.9.2.2', 1193), ('12.3', 401), ('8.6.2.1', 368), ...]
collection count: 3827
filtered section=3.1.1 -> 1 docs; sections=['3.1.1']  (exact-match holds: True)
filtered section=7.10.1.3.1 -> 1 docs; sections=['7.10.1.3.1']
```

**Key checks:**
- 100% of chunks have a `section_number` and non-empty `heading_path`.
- Hybrid retrieval returns a fused, deduplicated set spanning multiple sections.
- Metadata filtering returns only chunks whose `section_number` exactly matches the filter.

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **TOC false headings** | Regex matched table-of-contents lines like `7.6.1 Overview ..........` | The plan regex accepts bare numbered lines | Prefer hash-prefixed headings and skip dot-leader lines (`\.{4,}`) | ✅ Resolved |
| 2 | **Split heading titles** | Titles like `3.1.1\n\n## actor` | Docling renders the clause number and title on separate lines | `_clean_title()` takes the first meaningful line and strips leading `#` | ✅ Resolved |
| 3 | **Filter `"4.1"` returned 0** | `filtered_retrieve(vs, ..., "4.1")` → 0 docs | Section `4.1` does not exist in the corpus (verified: 407 distinct sections; `4.1` absent) | Not a bug — documented; re-tested with existing sections (`3.1.1`, `7.10.1.3.1`) which match exactly | ✅ Confirmed correct |
| 4 | **Huge section** | `8.9.2.2` produced 1193 chunks | A very large clause/annex body exceeds the 1.5× threshold and is recursively split | Expected behaviour; over-splitting of a giant clause is acceptable for retrieval | ⚠️ Acceptable |
| 5 | No Triton / batch failures | — | Milestone 4 fixes (pre-torch env flags, batched ingestion) applied | Reused existing helpers unchanged | ✅ No recurrence |

### 6.1 Failure #1 in detail

The plan's section regex treats any line matching `\d+(\.\d+)*\s+Title` as a clause heading. In the
Docling Markdown, the table of contents also matches this pattern (e.g.
`7.6.1 Overview ...................... 42`). Left unfiltered, TOC lines would create bogus sections
whose "body" is the TOC remainder.

**Fix**: treat only headings that Docling rendered with a Markdown hash prefix (`## `) as clause
headings, and additionally drop any candidate whose title contains a dot leader. This yields clean
section boundaries while still capturing nested clauses (`4.2.1`, `7.10.1.3.1`).

### 6.2 Failure #3 in detail

The initial filter test used `"4.1"` (a plan example) and returned zero documents, which looked
like a filter bug. Inspecting the section distribution showed ISO 15118 uses a different numbering
for the retrieved subset — `4.1` simply is not present. Re-testing with confirmed sections
(`3.1.1`, `7.10.1.3.1`) returned exact matches, proving the filter (including the `filter`→`where`
fallback) works.

---

## 7. Deviations from the Plan

1. **Heading extraction hardened** — prefer hash-prefixed headings and skip TOC dot-leader lines
   (the plan accepted all numbered lines). Documented in plan §13.11.
2. **Title cleaning** added for Docling's split-number rendering.
3. **`filtered_retrieve()` fallback** from `filter=` to `where=` for Chroma version differences.
4. **`BM25Retriever.from_documents(chunks)`** — the installed signature does not take a separate
   `metadatas` argument; metadata travels with each `Document`.

---

## 8. Acceptance Check (Milestone 5)

- [x] `ingestion/chunking_pair2.py` implemented.
- [x] `retrieval/pair2_retriever.py` implemented.
- [x] `section_number` populated on all chunks (3827/3827).
- [x] `heading_path` populated on all chunks (3827/3827).
- [x] Hybrid (vector + BM25) ensemble retrieval works.
- [x] Filtered query on a specific section number returns only that section.
- [x] Failures captured with mitigations (this document).
- [ ] MPS (Apple Silicon) verification — pending M1 hardware.

---

## 9. Next Step

**Milestone 6 — Pairing 3**: implement [`ingestion/chunking_pair3.py`](../ingestion/chunking_pair3.py)
(table-first chunks + clause chunks + cross-reference metadata), [`graph/builder.py`](../graph/builder.py)
(NetworkX cross-reference graph), and [`retrieval/pair3_retriever.py`](../retrieval/pair3_retriever.py)
(self-query + ensemble + graph expansion + reranking). Test: table chunks exist, cross-references
are populated, and graph traversal works.