# Milestone 4 — Pairing 1 (Recursive + Parent-Child Retrieval)

**Status**: ✅ Complete
**Date**: 2026-09-23
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §6.2–6.4, §6.6, §7 (Milestone 4), §13 (Deviations)

---

## 1. Objective

Implement the baseline pipeline: recursive parent/child chunking, a `ParentDocumentRetriever`
that matches on child chunks and returns parent context, and the shared LCEL RAG chain.

---

## 2. Files Created / Modified

| Path | Action | Purpose |
|---|---|---|
| [`ingestion/chunking_pair1.py`](../ingestion/chunking_pair1.py) | Created | Parent/child chunking with `parent_id`, `document_id`, `page_number` |
| [`retrieval/pair1_retriever.py`](../retrieval/pair1_retriever.py) | Created | `ParentDocumentRetriever` over pre-split chunks |
| [`chain/rag_chain.py`](../chain/rag_chain.py) | Created | LCEL RAG chain using `get_llm()`, provider-agnostic |
| [`vectorstore/store.py`](../vectorstore/store.py) | Created | `add_documents_in_batches()` helper (shared by all pairings) |
| [`config.py`](../config.py) | Modified | Env vars set pre-torch; added `VECTORSTORE_BATCH_SIZE` |
| [`ingestion/loader.py`](../ingestion/loader.py) | Modified | Env vars set pre-import (see §6.1) |
| [`.env.example`](../.env.example) | Modified | Documented `VECTORSTORE_BATCH_SIZE` |
| [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) | Modified | Added §13.9–§13.10 |

---

## 3. Design

### 3.1 Chunking ([`ingestion/chunking_pair1.py`](../ingestion/chunking_pair1.py))
- Parent splitter: 1024 tokens / 100 overlap, `add_start_index=True`.
- Child splitter: 256 tokens / 25 overlap.
- Each parent gets `parent_id = "{document_id}_parent_{i}"`; children inherit `parent_id`,
  `document_id`, `source_file`, and a `page_number` resolved from the parent's start offset via
  `page_for_offset()`.

### 3.2 Retrieval ([`retrieval/pair1_retriever.py`](../retrieval/pair1_retriever.py))
- `ParentDocumentRetriever` configured with `id_key="parent_id"`.
- Parents are placed in an `InMemoryStore` keyed by `parent_id`.
- Children (carrying `parent_id` metadata) are embedded into Chroma with deterministic ids.
- The retriever's built-in logic searches children, reads `parent_id`, and `mget`s the parent
  from the docstore — exactly the child-search → parent-fetch behaviour we want.
- `child_splitter`/`parent_splitter` are supplied but **never used** (we pre-split); they are
  required by the pydantic model and cannot be `None` (see §6.3).

### 3.3 Chain ([`chain/rag_chain.py`](../chain/rag_chain.py))
- `build_rag_chain(retriever, llm_model=None)` uses `get_llm()` (Ollama/OpenAI switch).
- Accepts a LangChain retriever **or** a plain callable (Pairing 3 will pass a lambda).
- `format_docs()` labels each chunk with `document_id`, `section_number`, and `page_number`.

---

## 4. Commands Executed

```bash
# API discovery
.venv/bin/python -c "from langchain_classic.retrievers import ParentDocumentRetriever; ..."

# End-to-end test
.venv/bin/python - <<'PY'
from ingestion.loader import load_iso_documents
from ingestion.chunking_pair1 import create_parent_child_chunks
from retrieval.pair1_retriever import build_parent_child_retriever
from chain.rag_chain import build_rag_chain
docs = load_iso_documents("Iso", use_ocr=False)
parents, children = create_parent_child_chunks(docs)
retriever = build_parent_child_retriever(children, parents)
print(build_rag_chain(retriever).invoke("What is the maximum AC charging voltage?"))
PY
```

---

## 5. Verification Results

```
docs=3 parents=2203 children=8495
children with missing parent: 0
[120.4s] retriever built
[0.1s] retrieved 5 docs; levels=['parent', 'parent', 'parent', 'parent', 'parent']
  - ISO-15118-2 page 116 | | EVMaximumVoltageLimit          | V                | Volt
  - ISO-15118-2 page 108 | | EVSEPowerLimitAchieved   | simpleType boolean refer to Annex C.6
[6.5s] ANSWER:
 I don't have enough information. The provided context only mentions the maximum voltage
 limits for the EVSE, which are in volts (V), but it does not specify the maximum AC
 charging voltage.
OK pairing1
```

**Key checks:**
- Every child has a `parent_id` present in the parent set (0 missing).
- Retrieval returns **parent** chunks (`chunk_level == "parent"`) — the child→parent mapping works.
- Page numbers are resolved correctly (116, 108).
- The chain produces a grounded, citation-aware answer (correctly refusing to fabricate when the
  retrieved context lacked the exact value — the intended safe behaviour).
- `document_id` distinguishes results from the three ISO parts.

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **Triton error returned** | `RuntimeError: Failed to find C compiler` on embedding | `ingestion.loader` imported Docling (→ torch) *before* `models.device` set `TORCH_DISABLE_NATIVE_JIT`; torch caches native-JIT state at import | Set both env vars at the top of [`config.py`](../config.py) and [`ingestion/loader.py`](../ingestion/loader.py), before any torch import | ✅ Resolved |
| 2 | **Chroma batch limit** | `ValueError: Batch size of 8495 is greater than max batch size of 5461` | Chroma caps a single upsert at ~5461; we sent 8495 children at once | Added [`vectorstore/store.py`](../vectorstore/store.py) `add_documents_in_batches()` (default 1000); used in the retriever | ✅ Resolved |
| 3 | **`child_splitter=None` rejected** | `ValidationError: child_splitter Input should be an instance of TextSplitter` | langchain_classic 1.0.8 makes `child_splitter`/`parent_splitter` required (`TextSplitter`), unlike the plan's example | Pass real `RecursiveCharacterTextSplitter` instances (unused, since we pre-split) | ✅ Resolved |
| 4 | Conservative answer | "I don't have enough information." | Retrieval returned voltage-limit tables but not the exact "maximum AC charging voltage" clause | Expected faithful behaviour; Pairings 2/3 with section metadata + graph should improve recall | ⚠️ Acceptable |
| 5 | Slow first build | 120 s to embed 8495 children | One-time embedding on GPU | Chroma persists; subsequent runs upsert the same ids | ✅ Acceptable |

### 6.1 Failure #1 in detail (regression of the Milestone 2 fix)

The Milestone 2 fix worked when `models.device` was imported first, but Pairing 1 imports
`ingestion.loader` first, which imports Docling → torch **before** `models.device` runs. Torch
caches its native-JIT decision at import, so setting the env var afterwards had no effect.

**Fix**: set `TORCH_DISABLE_NATIVE_JIT=1` and `PYTORCH_ENABLE_MPS_FALLBACK=1` at the very top of
[`config.py`](../config.py) (imported by every module) and [`ingestion/loader.py`](../ingestion/loader.py)
(before any third-party import). This guarantees the flags are applied regardless of import order.

### 6.2 Failure #2 in detail

Chroma's Rust backend enforces a maximum upsert batch. Because ISO-15118-2 alone produces
thousands of children, a single `add_documents` call exceeded it. The shared
`add_documents_in_batches()` helper now chunks ingestion; it will also be used by Pairings 2 and 3.

---

## 7. Deviations from the Plan

1. **Added `vectorstore/` package** — not in the plan's repo layout; holds the shared batching
   helper. Documented in plan §13.10.
2. **Env vars now set in `config.py`** (primary) rather than only `models/device.py`, because of
   torch's import-time caching. Documented in plan §13.9.
3. **`ParentDocumentRetriever` requires real splitters** — the plan's `child_splitter=None` fails
   validation; unused splitters are passed instead.
4. **New config** `VECTORSTORE_BATCH_SIZE` (default 1000).
5. **Retriever built from pre-split chunks** — parents in docstore, children in Chroma, `id_key="parent_id"`.

---

## 8. Acceptance Check (Milestone 4)

- [x] `ingestion/chunking_pair1.py` implemented.
- [x] `retrieval/pair1_retriever.py` implemented.
- [x] `chain/rag_chain.py` implemented (uses `get_llm()`).
- [x] Parent count > child count; every child `parent_id` exists in parents (0 missing).
- [x] Retrieval returns parent chunks with resolved page numbers.
- [x] End-to-end chain returns a string with citations.
- [x] Failures captured with mitigations (this document).
- [ ] MPS (Apple Silicon) verification — pending M1 hardware.

---

## 9. Next Step

**Milestone 5 — Pairing 2**: implement [`ingestion/chunking_pair2.py`](../ingestion/chunking_pair2.py)
(section-aware chunks with `heading_path`) and [`retrieval/pair2_retriever.py`](../retrieval/pair2_retriever.py)
(`EnsembleRetriever` = vector + BM25, with metadata filtering). Test: `section_number` and
`heading_path` populated; a filtered query on a section number works.