# Milestone 10 — Pairing 1 Thorough Evaluation

**Status**: ✅ Complete (3-query subset; full 32-query run documented but deferred by request)
**Date**: 2026-09-24
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §7 (Milestone 10), §7.1 (Metric Mapping)
**Spec reference**: [`RAG_EVAL_METRICS.md`](../RAG_EVAL_METRICS.md)

---

## 1. Objective

Run the full `RAG_EVAL_METRICS.md` metric suite on **Pairing 1** (Recursive + Parent-Child
retrieval) — retrieval, chunking, generation, and efficiency — and produce a per-pairing report.

---

## 2. Artifacts

| Path | Purpose |
|---|---|
| [`eval/run_pairing.py`](../eval/run_pairing.py) | CLI: build one pairing, measure indexing, evaluate, write report + raw JSON |
| [`eval/pair_report.py`](../eval/pair_report.py) | Renders the four-category Markdown report |
| [`eval/report_pair1.md`](../eval/report_pair1.md) | Generated report for Pairing 1 |
| [`eval/result_pair1.json`](../eval/result_pair1.json) | Raw metrics (consumed by the Milestone 13 comparison) |
| [`eval/runner.py`](../eval/runner.py) | Modified: section-metric N/A handling (see §6.1) |

---

## 3. Method

```bash
# Subset run actually executed
python -m eval.run_pairing --pairing 1 --limit 3

# Full run (deferred by request)
python -m eval.run_pairing --pairing 1
```

The runner builds Pairing 1 into its own index dir (`./data/indexes/pair1`), measures indexing
time (chunking + embedding separately) and index size, then evaluates every gold query: retrieval
(via the children→parents retriever), generation (via the RAG chain), and judge-based metrics
(relevance density, boundary coherence, faithfulness, answer relevance, answer correctness) plus
deterministic citation accuracy.

---

## 4. Results

### 4.1 Configuration
| Setting | Value |
|---|---|
| LLM provider/model | `ollama` / `llama3.2:3b-instruct-q4_K_M` |
| Device | `cuda (NVIDIA GeForce RTX 3050)` |
| Index dir | `./data/indexes/pair1` |
| Chunk stats source | parent chunks (children are internal) |

### 4.2 Retrieval-level (answer-term relevance)
| k | Recall@k | nDCG@k |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 3 | 1.000 | 1.000 |
| 5 | 1.000 | 1.000 |
| 10 | 1.000 | 1.000 |
| — | **MRR 1.000** | |

Section-based retrieval: **not applicable** (flagged — see §6.1).

### 4.3 Chunking-specific
| chunk_count | avg size (chars) | min | max | relevance_density | boundary_coherence |
|---|---|---|---|---|---|
| 2203 | 671.28 | 1 | 1024 | 0.295 | 0.417 |

### 4.4 End-to-end generation
| faithfulness | answer_relevance | answer_correctness | citation_accuracy |
|---|---|---|---|
| 0.467 | 0.420 | 0.497 | 0.000 |

### 4.5 Efficiency / cost
| indexing (ms) | retrieval (ms) | generation (ms) | e2e (ms) | index size | LLM calls/query | tokens/query |
|---|---|---|---|---|---|---|
| 109,885 | 61.1 | 3,752.7 | 3,813.8 | 261,121,478 B (~249 MiB) | 1 | 2,302.7 |

Indexing breakdown: chunking **1,312 ms**, embedding **108,573 ms** (embedding dominates).

### 4.6 Flags
```
retrieval_section: not applicable: retrieved chunks carry no section_number (recursive chunking)
```

---

## 5. Observations

- Retrieval hit on the subset is perfect (Recall@k = MRR = nDCG = 1.0), but this is only 3 queries.
- Judge scores are coarse (mostly 0.0/0.5) because the local 3B judge is being used; treat them as
  directional, not precise.
- Pairing 1's answers frequently cite "Section N/A" (its chunks have no section metadata), which
  structurally caps citation accuracy.
- Embedding dominates indexing time; the 249 MiB index is the largest of the three pairings.

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **Section metric wrong for Pairing 1** | `Recall@k (section)`/`MRR (section)` reported `0.0` | Pairing 1 recursive chunks have no `section_number`, so section relevance is *undefined*, not zero | Runner now detects absence of section metadata and reports section metrics as `None` + an explicit `retrieval_section` flag | ✅ Fixed |
| 2 | Full 32-query run deferred | Only 3 queries evaluated | Requested to skip the full run for now | Documented; full command provided; report header states "Queries evaluated: 3" | ⚠️ By request |
| 3 | Coarse judge resolution | Scores cluster at 0.0/0.5 | 3B local judge returns simple numbers | Documented; recommend a stronger judge LLM (e.g. `LLM_PROVIDER=openai`) for finer resolution | ⚠️ Documented |
| 4 | Re-embedding on every run | Indexing ≈ 110 s each invocation | Builders always upsert; no "reuse existing index" short-circuit | Noted as a future optimization (cache-skip when the collection exists) | ⚠️ Open |
| 5 | Tiny fragments | `min_chunk_size_chars = 1` | Recursive splitter emits very short tails | Observation; a min-length filter is a candidate post-MVP improvement | ⚠️ Observation |
| 6 | Limited citation accuracy | Aggregate `citation_accuracy = 0.0` | Pairing 1 answers cite section numbers that its chunks cannot supply | Inherent to recursive chunking; documented | ⚠️ Inherent |

### 6.1 Failure #1 in detail (real bug)

The first run displayed section-based `Recall@k`/`MRR` as **0.0** for Pairing 1. This is misleading:
the metric is *undefined* because recursive chunks carry no `section_number`. The runner was updated
to track whether any retrieved chunk had section metadata; when none do (as for Pairing 1), it sets
`retrieval_section` to `None` values and appends a flag. Re-running confirmed:

```
retrieval_section.mrr: None | recall@5: None
flags: [{'metric': 'retrieval_section', 'reason': 'not applicable: retrieved chunks carry no section_number (recursive chunking)'}]
```

This honours `RAG_EVAL_METRICS.md`'s rule to flag metrics that cannot be computed rather than
reporting a misleading value.

---

## 7. Deviations

1. **3-query subset** instead of the full 32 (explicitly requested).
2. **Section-based retrieval flagged N/A** for Pairing 1.
3. **Citation accuracy is 0** for Pairing 1 by construction (no section metadata).
4. **Judge is the local 3B model** (low resolution).

---

## 8. Acceptance Check (Milestone 10)

- [x] `eval/run_pairing.py` builds Pairing 1 and measures indexing time/size.
- [x] All four `RAG_EVAL_METRICS.md` categories computed and reported.
- [x] `eval/report_pair1.md` and `eval/result_pair1.json` generated.
- [x] Uncomputable metrics flagged, not omitted.
- [x] Test suite still green (14 passed).
- [x] Failures captured with mitigations (this document).
- [ ] Full 32-query run — deferred by request.

---

## 9. Next Step

**Milestone 11 — Pairing 2 Thorough Evaluation**: run `python -m eval.run_pairing --pairing 2`,
produce `eval/report_pair2.md`, and document in `docs/MILESTONE_11.md`.