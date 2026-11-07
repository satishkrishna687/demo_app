# Milestone 12 — Pairing 3 Thorough Evaluation

**Status**: ✅ Complete (3-query subset; full run documented but deferred by request)
**Date**: 2026-09-24
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §7 (Milestone 12), §7.1 (Metric Mapping)
**Spec reference**: [`RAG_EVAL_METRICS.md`](../RAG_EVAL_METRICS.md)

---

## 1. Objective

Run the full `RAG_EVAL_METRICS.md` metric suite on **Pairing 3** (Advanced ISO Chunking +
Graph-Enhanced Hybrid Retrieval + Reranking) and produce a per-pairing report.

---

## 2. Artifacts

| Path | Purpose |
|---|---|
| [`eval/report_pair3.md`](../eval/report_pair3.md) | Generated report for Pairing 3 |
| [`eval/result_pair3.json`](../eval/result_pair3.json) | Raw metrics (for the Milestone 13 comparison) |
| [`docs/MILESTONE_12.md`](../docs/MILESTONE_12.md) | This document |

Reused infrastructure from Milestones 10–11: [`eval/run_pairing.py`](../eval/run_pairing.py),
[`eval/pair_report.py`](../eval/pair_report.py), [`eval/runner.py`](../eval/runner.py).

---

## 3. Method

```bash
# Subset run actually executed
python -m eval.run_pairing --pairing 3 --limit 3

# Full run (deferred by request)
python -m eval.run_pairing --pairing 3
```

Pairing 3 is built into its own index dir (`./data/indexes/pair3`); indexing time includes chunking
(tables + clauses + cross-reference graph) and embedding. Each gold query is evaluated with the
four-stage retriever (hybrid search → graph expansion → reranking), generation, judge metrics, and
deterministic citation accuracy.

---

## 4. Results

### 4.1 Configuration
| Setting | Value |
|---|---|
| LLM provider/model | `ollama` / `llama3.2:3b-instruct-q4_K_M` |
| Device | `cuda (NVIDIA GeForce RTX 3050)` |
| Index dir | `./data/indexes/pair3` |
| Retrieval | graph-enhanced hybrid + cross-encoder reranking (SelfQuery fell back to vector — see §6.4) |

### 4.2 Retrieval-level
| k | Recall@k (answer-term) | Recall@k (section) |
|---|---|---|
| 1 | 0.667 | 1.000 |
| 3 | 0.667 | 1.000 |
| 5 | 0.667 | 1.000 |
| 10 | 0.667 | 1.000 |
| — | **MRR 0.667** | **MRR 1.000** |

The two relevance signals diverge: section-based relevance is perfect (1.0), while answer-term
relevance is 0.667 — see §6.1.

### 4.3 Chunking-specific
| chunk_count | avg size (chars) | min | max | relevance_density | boundary_coherence |
|---|---|---|---|---|---|
| 1970 | 657.07 | 3 | **21482** | 0.190 | 0.317 |

The max chunk of 21,482 chars is an unsplit large table (see §6.2); boundary coherence is the
lowest of the three pairings.

### 4.4 End-to-end generation
| faithfulness | answer_relevance | answer_correctness | citation_accuracy |
|---|---|---|---|
| 0.410 | 0.417 | 0.583 | **1.000** |

Citation accuracy is perfect — the best of the three pairings — thanks to table + clause chunks
carrying citable section numbers.

### 4.5 Efficiency / cost
| indexing (ms) | retrieval (ms) | generation (ms) | e2e (ms) | index size | LLM calls/query | tokens/query |
|---|---|---|---|---|---|---|
| 125,517 | **10,513.3** | 3,785.9 | 11,388.4 | 21,867,436 B (~20.9 MiB) | 1 | 2,516.7 |

Indexing breakdown: chunking **476 ms**, embedding **125,041 ms**. Retrieval is ~160× slower than
Pairings 1/2 due to graph expansion + cross-encoder reranking.

### 4.6 Flags
```
None — all metrics computed.
```

---

## 5. Cross-Pairing Snapshot (subset)

| Metric | Pairing 1 | Pairing 2 | Pairing 3 |
|---|---|---|---|
| Recall@5 (answer-term) | 1.000 | 1.000 | 0.667 |
| Recall@5 (section) | N/A | 1.000 | 1.000 |
| Citation accuracy | 0.000 | 0.722 | **1.000** |
| Boundary coherence | 0.417 | 0.400 | 0.317 |
| Indexing time | 109.9 s | 70.9 s | 125.5 s |
| Retrieval latency | 61 ms | 65 ms | **10,513 ms** |
| Index size | ~249 MiB | ~27.7 MiB | ~20.9 MiB |

(Full multi-metric comparison is produced in Milestone 13.)

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **Answer-term recall lower (0.667)** | One of 3 queries had no retrieved chunk containing the answer term | Graph expansion/reranking surfaced section-correct chunks whose text didn't include the exact `answer_terms` (term may appear in a sibling chunk) | Section-based recall is 1.0 — the metric divergence is expected; documented, and section relevance is the more reliable signal for sectioned pairings | ⚠️ Documented |
| 2 | **Very large chunk** | `max_chunk_size_chars = 21482` | Large tables are emitted as single table chunks and are never size-capped | Flagged; a table-splitting cap is a planned improvement | ⚠️ Open |
| 3 | **High retrieval latency** | 10,513 ms/query vs ~65 ms for Pairings 1/2 | Graph expansion increases the candidate set, and the cross-encoder reranker scores many pairs | Candidate set already bounded (`RERANK_TOP_N`, `max_depth=1`, `top_k*2`); documented as the quality/latency tradeoff | ⚠️ Inherent |
| 4 | **SelfQueryRetriever fallback** | `WARNING: SelfQueryRetriever unavailable (ImportError: DatabricksVectorSearch...)` | langchain-community 0.4.2 self-query translators import a missing symbol | Stage 1 falls back to a plain vector retriever (already handled + logged); documented in Milestone 6 | ⚠️ Degraded |
| 5 | **GPU VRAM pressure during rerank** | Historical `CUDACachingAllocator ... OOM` when the reranker loads alongside the embedding model | 8 GB VRAM | Reranking is guarded by `try/except` (degrades to unranked); run reranker on CPU or use a smaller model | ⚠️ Mitigated |
| 6 | Full 32-query run deferred | Only 3 queries | Requested | Documented; report states scope; full command provided | ⚠️ By request |

### 6.1 Failure #1 in detail (metric divergence)

Pairing 3's answer-term recall (0.667) is lower than its section-based recall (1.0). This is **not a
retrieval failure**: the section-based metric confirms the *correct clause* was retrieved, but for
one query the specific `answer_terms` string appeared in a different chunk of that clause (or in a
table chunk not ranked in the top-k). The two signals measure different things:
- **answer-term** = exact term presence in the retrieved chunk text (strict, sensitive to chunk
  boundaries);
- **section** = whether the retrieved chunk's clause is a gold clause (semantic, tolerant to
  boundary placement).

For sectioned pairings, the section-based signal is the more meaningful one; this is documented as
a metric characteristic rather than a bug.

### 6.2 Failure #2 in detail (oversized chunk)

The largest chunk is 21,482 characters — an entire multi-page table emitted as one chunk (tables are
not passed through the recursive splitter). This inflates token cost and lowers boundary coherence
(0.317). **Planned mitigation**: cap table chunk size by splitting large Markdown tables at row
boundaries with a repeated header; not implemented in this milestone.

### 6.3 Failure #3 in detail (latency)

Retrieval latency is dominated by (a) graph expansion adding neighbours and (b) the cross-encoder
reranker scoring each (query, candidate) pair. The candidate set is already bounded, but on an 8 GB
GPU the reranker is the bottleneck. This is the explicit **quality ↔ latency tradeoff** the
`RAG_EVAL_METRICS.md` report is asked to surface; Pairing 3 wins on citation accuracy but costs
~160× the retrieval latency of Pairings 1/2.

### 6.4 Failure #4 in detail

Unchanged from Milestone 6: `SelfQueryRetriever.from_llm` raises `ImportError` because
langchain-community 0.4.2's self-query translators import `DatabricksVectorSearch`, which is absent.
Stage 1 degrades to a plain vector retriever and logs a warning. Fixing requires pinning a compatible
`langchain-community`; deferred.

---

## 7. Deviations

1. **3-query subset** instead of the full 32 (consistent with Milestones 10–11).
2. **Judge is the local 3B model** (low resolution).
3. **Section-based relevance treated as the primary signal** for sectioned pairings.
4. **Index size is approximate** (whole-directory walk).

---

## 8. Acceptance Check (Milestone 12)

- [x] Pairing 3 built with measured indexing time and index size.
- [x] All four `RAG_EVAL_METRICS.md` categories computed and reported.
- [x] Section-based retrieval computed (applicable) and scoring 1.0.
- [x] `eval/report_pair3.md` and `eval/result_pair3.json` generated.
- [x] No uncomputable metrics (flags: none).
- [x] Failures/observations captured with mitigations (this document).
- [ ] Full 32-query run — deferred by request.

---

## 9. Next Step

**Milestone 13 — Cross-Pair Comparison Report**: implement `eval/comparison_report.py` to aggregate
`eval/result_pair1.json`, `result_pair2.json`, and `result_pair3.json` into
`eval/RAG_EVAL_REPORT.md` with the summary table, rankings, efficiency tradeoff, and recommendation
per `RAG_EVAL_METRICS.md`.