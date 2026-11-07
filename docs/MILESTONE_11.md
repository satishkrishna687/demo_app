# Milestone 11 — Pairing 2 Thorough Evaluation

**Status**: ✅ Complete (3-query subset; full run documented but deferred by request)
**Date**: 2026-09-24
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §7 (Milestone 11), §7.1 (Metric Mapping)
**Spec reference**: [`RAG_EVAL_METRICS.md`](../RAG_EVAL_METRICS.md)

---

## 1. Objective

Run the full `RAG_EVAL_METRICS.md` metric suite on **Pairing 2** (Section-Based Chunking +
Metadata-Filtered Hybrid Retrieval) and produce a per-pairing report.

---

## 2. Artifacts

| Path | Purpose |
|---|---|
| [`eval/report_pair2.md`](../eval/report_pair2.md) | Generated report for Pairing 2 |
| [`eval/result_pair2.json`](../eval/result_pair2.json) | Raw metrics (for the Milestone 13 comparison) |
| [`docs/MILESTONE_11.md`](../docs/MILESTONE_11.md) | This document |

Reused infrastructure from Milestone 10: [`eval/run_pairing.py`](../eval/run_pairing.py),
[`eval/pair_report.py`](../eval/pair_report.py), [`eval/runner.py`](../eval/runner.py).

---

## 3. Method

```bash
# Subset run actually executed
python -m eval.run_pairing --pairing 2 --limit 3

# Full run (deferred by request)
python -m eval.run_pairing --pairing 2
```

Pairing 2 is built into its own index dir (`./data/indexes/pair2`); indexing time (chunking +
embedding) and index size are measured, then each gold query is evaluated for retrieval (hybrid
vector + BM25), generation, judge metrics, and deterministic citation accuracy.

---

## 4. Results

### 4.1 Configuration
| Setting | Value |
|---|---|
| LLM provider/model | `ollama` / `llama3.2:3b-instruct-q4_K_M` |
| Device | `cuda (NVIDIA GeForce RTX 3050)` |
| Index dir | `./data/indexes/pair2` |
| Retrieval | hybrid (Chroma vector + BM25, weights 0.6/0.4) |

### 4.2 Retrieval-level
| k | Recall@k (answer-term) | Recall@k (section) |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 3 | 1.000 | 1.000 |
| 5 | 1.000 | 1.000 |
| 10 | 1.000 | 1.000 |
| — | **MRR 1.000** | **MRR 1.000** |

Unlike Pairing 1, the **section-based metric is applicable here** (section chunks carry
`section_number`), and it also scores 1.0.

### 4.3 Chunking-specific
| chunk_count | avg size (chars) | min | max | relevance_density | boundary_coherence |
|---|---|---|---|---|---|
| 3827 | 333.93 | 1 | 899 | 0.192 | 0.400 |

### 4.4 End-to-end generation
| faithfulness | answer_relevance | answer_correctness | citation_accuracy |
|---|---|---|---|
| 0.410 | 0.483 | 0.547 | **0.722** |

Citation accuracy is markedly higher than Pairing 1 (0.000) because section chunks carry
citable clause numbers.

### 4.5 Efficiency / cost
| indexing (ms) | retrieval (ms) | generation (ms) | e2e (ms) | index size | LLM calls/query | tokens/query |
|---|---|---|---|---|---|---|
| 70,858 | 64.7 | 3,155.9 | 3,220.6 | 29,024,248 B (~27.7 MiB) | 1 | 2,167.3 |

Indexing breakdown: chunking **83 ms**, embedding **70,775 ms**.

### 4.6 Flags
```
None — all metrics computed.
```

---

## 5. Observations

- Both answer-term and section-based retrieval are perfect on the subset (Recall@k = MRR = 1.0).
- Citation accuracy (0.72) is the highest so far — section metadata enables correct citations.
- Chunks are smaller on average (334 vs 671 chars for Pairing 1), consistent with clause-level
  splitting; relevance density is lower (0.19), likely because short clause chunks include
  headings/boilerplate relative to their size.
- Embedding dominates indexing time again (70.8 s of 70.9 s).
- Index size (~27.7 MiB) is much smaller than Pairing 1's measured ~249 MiB (see §6.4).

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | Full 32-query run deferred | Only 3 queries evaluated | Requested not to run the full set | Documented; report header states "Queries evaluated: 3"; full command provided | ⚠️ By request |
| 2 | Coarse judge resolution | Scores cluster at 0.0/0.5 | Local 3B judge | Documented; recommend a stronger judge LLM for finer scores | ⚠️ Documented |
| 3 | Tiny fragments | `min_chunk_size_chars = 1` | Recursive splitter emits short tails within long clauses | Candidate min-length filter (post-MVP) | ⚠️ Observation |
| 4 | **Index-size measurement conflates collections** | Pairing 1 reported ~249 MiB vs Pairing 2 ~27.7 MiB (much more than the ~2.2× chunk-count ratio) | `dir_size_bytes` walks the entire per-pairing Chroma dir, which includes Chroma overhead/artifacts beyond vectors | Each pairing already uses a **separate** dir (pair1/pair2), so values are per-pairing, but they still include Chroma overhead; a per-collection size metric is a future improvement | ⚠️ Documented |
| 5 | Re-embedding on every run | ~71 s each invocation | Builders always upsert; no index-reuse short-circuit | Future optimization (cache-skip when the collection exists) | ⚠️ Open |
| 6 | No functional failures | Build + evaluate completed cleanly | — | The Milestone 10 section-metric fix applied cleanly here (section metrics computed) | ✅ Clean |

### 6.1 No hard failures this milestone

Pairing 2's build and evaluation completed on the first attempt. The Milestone 10 fix for
section-metric applicability worked as intended: because Pairing 2's chunks carry
`section_number`, `retrieval_section` was computed (Recall@k = MRR = 1.0) instead of being flagged
N/A.

### 6.2 Failure #4 in detail

Index size is measured by walking the per-pairing Chroma directory. Each pairing already writes to
its own directory (`./data/indexes/pairN`), so the values are not shared — but they include Chroma's
internal files, not just vectors. The Pairing 1 number (~249 MiB) is disproportionately larger than
expected from its vector count (8495 children), suggesting stale/duplicated artifacts in that
directory from earlier runs. **Mitigation**: treat index size as approximate; a precise per-collection
byte metric (e.g. summing stored embeddings × dimension × 4 bytes, or `collection.count()`-based
estimates) is planned as an improvement. Left as-is for now.

---

## 7. Deviations

1. **3-query subset** instead of the full 32 (consistent with the Milestone 10 decision).
2. **Judge is the local 3B model** (low resolution).
3. **Index size is approximate** (whole-directory walk).

---

## 8. Acceptance Check (Milestone 11)

- [x] Pairing 2 built with measured indexing time and index size.
- [x] All four `RAG_EVAL_METRICS.md` categories computed and reported.
- [x] Section-based retrieval metric computed (applicable) and scoring 1.0.
- [x] `eval/report_pair2.md` and `eval/result_pair2.json` generated.
- [x] No uncomputable metrics (flags: none).
- [x] Failures/observations captured with mitigations (this document).
- [ ] Full 32-query run — deferred by request.

---

## 9. Next Step

**Milestone 12 — Pairing 3 Thorough Evaluation**: run `python -m eval.run_pairing --pairing 3`,
produce `eval/report_pair3.md`, and document in `docs/MILESTONE_12.md`.