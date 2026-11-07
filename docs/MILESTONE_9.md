# Milestone 9 — Evaluation Data, Metric Library, and Judge

**Status**: ✅ Complete
**Date**: 2026-09-24
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §7 (Milestone 9), §7.1 (Metric Mapping)

---

## 1. Objective

Build the foundation for thorough per-pairing evaluation: a gold query set, a deterministic metric
library, an LLM judge, a per-pairing runner, and a pytest suite — implementing the four metric
categories in [`RAG_EVAL_METRICS.md`](../RAG_EVAL_METRICS.md).

---

## 2. Files Created / Modified

| Path | Action | Purpose |
|---|---|---|
| [`eval/gold_seeds.json`](../eval/gold_seeds.json) | Created | 32 hand-authored seeds (query, category, `answer_terms`) |
| [`eval/build_gold_set.py`](../eval/build_gold_set.py) | Created | Derives `gold_relevant_sections` + `gold_answer` from corpus text |
| [`eval/gold_set.json`](../eval/gold_set.json) | Generated | Gold set (32/32 seeds matched to sections) |
| [`eval/metrics.py`](../eval/metrics.py) | Created | Recall@k, Precision@k, MRR, nDCG@k, chunk stats, index size |
| [`eval/judge.py`](../eval/judge.py) | Created | Deterministic LLM judge + rule-based citation accuracy |
| [`eval/runner.py`](../eval/runner.py) | Created | Per-pairing runner producing all four metric categories |
| [`tests/test_metrics.py`](../tests/test_metrics.py) | Created | 12 deterministic metric unit tests |
| [`tests/smoke_test.py`](../tests/smoke_test.py) | Created | 4 fast smoke tests (config, device, gold seeds, metrics) |

---

## 3. Design

### 3.1 Gold set construction (`build_gold_set.py`)
Chunks are generated at runtime, so fixed passage IDs are unavailable. Instead, each seed query
carries `answer_terms`; the builder locates those terms in the **section chunks** (Pairing 2's
chunker used purely as a search index) and records:
- `gold_relevant_sections` — `document_id:section_number` keys containing the terms, and
- `gold_answer` — a short reference snippet around the first match.

This is **weak supervision anchored in corpus text** (not in the retriever), so it is reproducible
and non-circular. Unmatched seeds are flagged, not dropped.

### 3.2 Metric library (`metrics.py`)
- Primary relevance: **answer-term matching** — a retrieved chunk is relevant if its text contains
  any of the query's terms (or the gold answer). This works uniformly across pairings because
  Pairing 1 chunks have no section metadata.
- Secondary relevance: **section-based** (`document_id:section_number` ∈ gold sections), reported
  separately for Pairings 2 and 3.
- Implements Recall@k, Precision@k, MRR, nDCG@k for k ∈ {1,3,5,10}, plus chunk statistics
  (count/min/mean/max chars) and recursive index size.

### 3.3 Judge (`judge.py`)
- One `LLMJudge` using the configured model at `temperature=0`.
- Strict prompt: *"Respond with ONLY a single number between 0 and 1."*
- Robust parsing: first numeric token, clamped to [0,1]; one retry; then `None` (flagged).
- Scores: relevance density, boundary coherence, faithfulness, answer relevance, answer
  correctness.
- Citation accuracy is deterministic (regex-cited sections vs. retrieved sections), returning
  `None` when no sections are cited.
- Results cached per instruction string to avoid recomputation.

### 3.4 Runner (`runner.py`)
`evaluate_pairing(...)` returns a structured dict with the four categories
(`retrieval`, `retrieval_section`, `chunking`, `generation`, `efficiency`), `per_query` rows,
`judge_coverage`, and `flags` for every metric that could not be computed.

---

## 4. Commands Executed

```bash
.venv/bin/python -m eval.build_gold_set        # -> eval/gold_set.json
.venv/bin/python -m pytest tests/ -q           # -> 14 passed
.venv/bin/python -c "from eval.judge import LLMJudge; ..."      # judge smoke
.venv/bin/python -c "from eval.runner import evaluate_pairing; ..."  # runner smoke (fake pipeline)
```

---

## 5. Verification Results

### 5.1 Gold set
```
INFO Wrote eval/gold_set.json (32/32 seeds matched to sections)
gold entries: 32 | matched to >=1 section: 32
```

### 5.2 Tests
```
14 passed in 1.34s
```

### 5.3 Judge with the local model
```
relevance_density: 0.5
boundary_coherence: 0.0
faithfulness: 0.5
citation_accuracy: 1.0
citation_accuracy(none): None
```
The strict prompt makes the local `llama3.2:3b` model return parseable numeric scores — a notable
improvement over the serialized-JSON RAGAS prompts that produced `nan` in Milestone 7.

### 5.4 Runner aggregation (fake pipeline)
```
retrieval: recall@1=1.0, precision@1=1.0, ndcg@1=1.0, mrr=1.0, ...
generation: citation_accuracy=1.0, others None (judge disabled)
chunking: chunk_count=1, avg_chunk_size_chars=10.0
flags: [generation.faithfulness (judge disabled), ..., efficiency.index_size_bytes, tokens_per_query]
```
Unavailable metrics are **flagged, not omitted**, as required by `RAG_EVAL_METRICS.md`.

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **pytest failures (2)** | `test_ndcg_at_k`, `test_aggregate_retrieval` failed | Incorrect expected values in the *tests*, not the metrics (recall@1 over two queries where one has a rank-1 hit is 0.5; nDCG ideal uses the relevant count = 1.0) | Corrected the test expectations | ✅ Resolved (14 passed) |
| 2 | **Section relevance undefined for Pairing 1** | Pairing 1 recursive chunks carry no `section_number` | Recursive splitting has no clause metadata | Primary relevance switched to answer-term matching (works for all pairings); section relevance retained as a secondary signal for Pairings 2/3 | ✅ Resolved |
| 3 | **Local judge reliability** | Milestone 7 RAGAS metrics were `nan` due to non-JSON judge output | 3B model does not emit strict JSON | Strict "only a number 0–1" prompt + first-number regex parsing + retry + cache | ✅ Resolved (parseable scores) |
| 4 | **Token-usage instrumentation uncertain** | `UsageMetadataCallbackHandler` availability/return shape varies by version | LangChain 1.x callback API | Defensive `try/except`; tokens reported as `None` and flagged when unavailable | ⚠️ Flagged |
| 5 | **Gold answers are weak labels** | `gold_answer` is a corpus snippet, not a curated answer | No human-authored gold answers for ISO clauses | Documented as weak supervision; correctness judging uses the snippet as a reference; seeds are flagged if unmatched | ⚠️ By design |
| 6 | **Judge cost/latency** | Each judged metric is an LLM call | Local CPU/GPU inference per call | Per-instruction cache + `max_judge_queries` cap | ✅ Mitigated |

### 6.1 Failure #1 in detail

Two unit tests failed on first run. Both were **test errors**: the metric implementations were
correct. `test_aggregate_retrieval` expected `recall@1 == 0.0` for a two-query set where the first
query has a relevant doc at rank 1 (correct value: 0.5), and `test_ndcg_at_k` used an incorrect
ideal-DCG denominator (2 relevant assumed instead of the actual 1). Expectations were corrected;
the suite now passes.

### 6.2 Failure #2 in detail

`RAG_EVAL_METRICS.md` derives Recall@k/Precision@k from *gold relevant-chunk labels*. Our chunks
are generated at runtime and Pairing 1's recursive chunks have **no section metadata**, so
section-based labelling cannot cover all pairings. The primary relevance signal was therefore
defined as **answer-term presence in the retrieved chunk text**, which is deterministic and applies
to all three pairings; section-based relevance is computed additionally where metadata exists.

---

## 7. Deviations from the Plan / `RAG_EVAL_METRICS.md`

1. **Relevance = answer-term matching** (primary) rather than gold chunk IDs (chunks are not
   stable across pairings). Section-based relevance is a secondary metric.
2. **Gold set is built by weak supervision** (`build_gold_set.py`) from corpus text.
3. **Citation accuracy is rule-based** (deterministic) rather than LLM-judged.
4. **nDCG uses binary relevance** by default (graded relevance optional).
5. **Tokens per query** may be flagged unavailable depending on callback support.

---

## 8. Acceptance Check (Milestone 9)

- [x] `eval/gold_seeds.json` (32 seeds) and generated `eval/gold_set.json` (32/32 matched).
- [x] `eval/metrics.py` implements Recall@k, Precision@k, MRR, nDCG@k, chunk stats, index size.
- [x] `eval/judge.py` implements the five judge metrics + deterministic citation accuracy.
- [x] `eval/runner.py` produces all four metric categories with flags.
- [x] Deterministic metric unit tests pass (`pytest`: 14 passed).
- [x] Failures captured with mitigations (this document).
- [ ] Pairing-level evaluation runs — Milestones 10–12.

---

## 9. Next Step

**Milestone 10 — Pairing 1 Thorough Evaluation**: build Pairing 1, run the full metric suite over
the gold set, produce `eval/report_pair1.md`, and document process/failures/mitigations in
`docs/MILESTONE_10.md`.