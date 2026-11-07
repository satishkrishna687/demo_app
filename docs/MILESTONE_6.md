# Milestone 6 — Pairing 3 (Advanced ISO Chunking + Graph-Enhanced Hybrid Retrieval + Reranking)

**Status**: ✅ Complete
**Date**: 2026-09-23
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §6.2–6.5, §7 (Milestone 6), §13 (Deviations)

---

## 1. Objective

Implement the most sophisticated pipeline: table-first + clause chunking with cross-reference
metadata, a NetworkX cross-reference graph, and a four-stage retriever (metadata-aware candidates
→ hybrid fusion → graph expansion → cross-encoder reranking).

---

## 2. Files Created / Modified

| Path | Action | Purpose |
|---|---|---|
| [`ingestion/chunking_pair3.py`](../ingestion/chunking_pair3.py) | Created | Table + clause chunks, cross-references, dedupe |
| [`graph/builder.py`](../graph/builder.py) | Created | NetworkX cross-reference graph (clause refs, bounded fan-out) |
| [`retrieval/pair3_retriever.py`](../retrieval/pair3_retriever.py) | Created | 4-stage graph-enhanced retriever with reranking |
| [`vectorstore/store.py`](../vectorstore/store.py) | Modified | Chroma-safe metadata sanitisation |
| [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) | Modified | Added §13.12 |

---

## 3. Design

### 3.1 Chunking ([`ingestion/chunking_pair3.py`](../ingestion/chunking_pair3.py))
- **Tables first**: Markdown tables become standalone chunks (`content_type="table"`) with a
  `table_caption`; the rest of the section is split into `content_type="text"` chunks.
- **Cross-references**: clause/annex refs (`see 4.2`, `refer to Annex A`, ...) stored in
  `section_references`; all refs (incl. `Table N`/`Figure N`) stored in `cross_references`.
- **Dedupe**: exact-duplicate chunks are removed so vector ids stay unique.
- Reuses the heading regex/helpers from Pairing 2 for consistent section boundaries.

### 3.2 Graph ([`graph/builder.py`](../graph/builder.py))
- Node id: `{document_id}:{section_number}_{md5(content)[:8]}` (no cross-part collisions).
- Edges from **clause** references only, matched within the same `document_id`; exact matches are
  preferred over descendants, and fan-out is capped at 20 targets per reference.

### 3.3 Retrieval ([`retrieval/pair3_retriever.py`](../retrieval/pair3_retriever.py))
1. Metadata-aware candidates via `SelfQueryRetriever`, with a **vector-retriever fallback**.
2. `EnsembleRetriever` (vector + BM25, weights 0.6/0.4).
3. BFS graph expansion up to `max_depth`.
4. Cross-encoder reranking (`BAAI/bge-reranker-v2-m3`), gracefully skipped on failure.

---

## 4. Commands Executed

```bash
# End-to-end
.venv/bin/python - <<'PY'
# load docs -> create_advanced_iso_chunks -> build_cross_reference_graph
# -> build_graph_enhanced_retriever -> graph_expanded_retrieve -> build_rag_chain
PY

# Focused graph / self-query checks (reusing the persisted store)
.venv/bin/python - <<'PY'
# stub retriever to prove BFS expansion; SelfQueryRetriever.from_llm probe
PY
```

---

## 5. Verification Results

### 5.1 Chunking + graph
```
chunks=1970 types=Counter({'text': 1715, 'table': 255})
table chunks: 255
chunks with section_references: 851
graph nodes=1970 edges=955
```

### 5.2 Retriever + chain
```
[102.6s] retriever=EnsembleRetriever
collection count: 1970
[16.3s] expanded 5; sections=['8.9.2.2','8.9.2.2','8.9.2.2','8.9.2.2','8.9.2.2']
ANSWER: ... the nominal AC charging voltage, which is 230 V ...
OK pairing3
```

### 5.3 Focused graph-expansion proof
```
node ISO-15118-2:1_9c6060e9 out_degree=5
stub retrieval -> expanded 10 docs (expected >= 6)
expansion added graph neighbours: True
```

**Key checks:**
- Table chunks exist (255) alongside text chunks.
- Cross-references populated (851 chunks with clause refs).
- Graph node count == chunk count (1970); edges > 0 (955).
- Graph expansion demonstrably adds neighbours (1 seed → 10 docs).
- End-to-end chain returns a grounded, cited answer.

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **Duplicate chunk ids** | `DuplicateIDError: Expected IDs to be unique` | Identical chunk content in the same section produced identical md5-based ids | Added `_dedupe()` in the chunker (key: document, section, content_type, content) | ✅ Resolved (2095 → 1970) |
| 2 | **Empty-list metadata** | `ValueError: Expected metadata list value for key 'cross_references' to be non-empty` | Chroma rejects empty lists (and `None`) as metadata values | Centralised `_chroma_safe_metadata()` in [`vectorstore/store.py`](../vectorstore/store.py); drops `None`/empty lists on copies sent to Chroma | ✅ Resolved |
| 3 | **Graph edge explosion** | 72,765 edges | `Table N`/`Figure N` refs were matched against clause numbers, and short refs like `1` matched all `1.x` | Graph uses clause-only `section_references`; exact match preferred; fan-out capped at 20 | ✅ Resolved (→ 955 edges) |
| 4 | **`SelfQueryRetriever.from_llm` fails** | `ImportError: cannot import name 'DatabricksVectorSearch'` | langchain-community 0.4.2's self-query translators import a missing symbol | Pairing 3 falls back to a plain vector retriever; failure is now logged via `logger.warning` | ⚠️ Degraded — documented |
| 5 | Results clustered in one section | All 5 results from `8.9.2.2` | That section dominates the corpus (1,193 chunks in Pairing 2) | Acceptable; evaluation (Milestone 7) will quantify retrieval quality | ⚠️ Observation |

### 6.1 Failure #4 in detail

`SelfQueryRetriever.from_llm` triggers an internal import of self-query translators that reference
`DatabricksVectorSearch`, which is not exported by the installed `langchain-community 0.4.2`. The
import fails regardless of the LLM backend. Because the retriever builder wraps this in a
`try/except`, Pairing 3 still constructs successfully using a **plain vector retriever** for
Stage 1; the automatic natural-language→metadata-filter translation is therefore inactive.

**Options** (documented, not applied): pin a compatible `langchain-community`, or implement a
lightweight query→section-filter parser. The fallback keeps the pipeline functional meanwhile.

---

## 7. Deviations from the Plan

1. **Added `section_references`** (clause-only) distinct from `cross_references` (all refs), so the
   graph is not polluted by table/figure numbers. Documented in plan §13.12.
2. **Chunk dedupe** added to guarantee unique vector ids.
3. **Chroma metadata sanitisation** centralised in the batching helper.
4. **Graph fan-out cap** (20) and exact-match preference added.
5. **`SelfQueryRetriever` unavailable** → vector-retriever fallback (pipeline still works).

---

## 8. Acceptance Check (Milestone 6)

- [x] `ingestion/chunking_pair3.py` implemented.
- [x] `graph/builder.py` implemented.
- [x] `retrieval/pair3_retriever.py` implemented (incl. reranking stage).
- [x] Table chunks exist (`content_type="table"`).
- [x] Cross-references populated.
- [x] Graph nodes == chunk count; edges > 0.
- [x] Graph expansion adds neighbours (proven with a stub retriever).
- [x] End-to-end chain returns a grounded answer.
- [x] Failures captured with mitigations (this document).
- [ ] MPS (Apple Silicon) verification — pending M1 hardware.

---

## 9. Next Step

**Milestone 7 — Evaluation Harness**: create `eval/test_queries.json` and implement
[`eval/evaluator.py`](../eval/evaluator.py) + [`eval/report_generator.py`](../eval/report_generator.py)
to compare all three pipelines with RAGAS and custom metrics, producing `eval/report.md`.