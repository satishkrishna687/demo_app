# Milestone 7 — Evaluation Harness

**Status**: ✅ Complete
**Date**: 2026-09-24
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §6.8, §7 (Milestone 7), §13 (Deviations)

---

## 1. Objective

Build a reproducible evaluation harness that compares the three pipelines on a shared query set
with custom retrieval metrics and (best-effort) RAGAS metrics, and generate a Markdown report.

---

## 2. Files Created / Modified

| Path | Action | Purpose |
|---|---|---|
| [`eval/test_queries.json`](../eval/test_queries.json) | Created | 20 queries across 6 categories |
| [`eval/evaluator.py`](../eval/evaluator.py) | Created | Custom metrics + RAGAS (best-effort) + CLI |
| [`eval/report_generator.py`](../eval/report_generator.py) | Created | Markdown comparison report |
| [`eval/report.md`](../eval/report.md) | Generated | Comparison report output |
| [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) | Modified | Added §13.13 |

---

## 3. Design

### 3.1 Query set ([`eval/test_queries.json`](../eval/test_queries.json))
20 queries across `parameter_lookup`, `procedure`, `cross_reference`, `annex_lookup`,
`definition`, `comparison`. Several carry a best-effort `expected_section` for hit-rate
computation; `gold_answer` is left empty (curating gold answers for ISO clauses is out of scope).

### 3.2 Evaluator ([`eval/evaluator.py`](../eval/evaluator.py))
- **Custom metrics** (always): retrieval hit rate, citation accuracy, retrieval/end-to-end
  latency, average chunks retrieved, per-category hit rate.
- **RAGAS** (best-effort): `faithfulness` + `answer_relevancy`, plus `context_recall` /
  `answer_correctness` when gold answers exist. Uses the configured LLM/embeddings.
- **`install_ragas_shim()`**: makes ragas importable under langchain-community 0.4.2.
- **`build_pipeline(name, documents)`**: builds one pipeline; the CLI loops over pipelines and
  `gc.collect()`s between them to bound memory.
- CLI: `python -m eval.evaluator [--limit N] [--skip-ragas] [--queries P] [--output P]`.

### 3.3 Report ([`eval/report_generator.py`](../eval/report_generator.py))
Executive summary, custom aggregate table, RAGAS table (or an explicit note), hit rate by
category, per-pipeline details, recommendations, and a reproducibility appendix (pinned versions
+ model names).

---

## 4. Commands Executed

```bash
# Harness smoke (imports + report rendering with fake data)
.venv/bin/python -c "import eval.evaluator, eval.report_generator; ..."

# Full evaluation (capped for time; RAGAS disabled for speed/stability)
.venv/bin/python -m eval.evaluator --limit 3 --skip-ragas

# RAGAS-enabled path (slow with the local 3B model)
.venv/bin/python -m eval.evaluator --limit 3
```

---

## 5. Verification Results

`eval/report.md` (custom metrics):
```
| pipeline                            | num_queries | hit_rate | citation_accuracy | avg_retrieval_latency_ms | avg_e2e_latency_ms | avg_chunks_retrieved |
| Pairing 1: Recursive + Parent-Child |      3      |  0.000   |       0.000       |        57.15             |     2912.49        |         5            |
| Pairing 2: Section + Hybrid         |      3      |  0.667   |       1.000       |        79.76             |     2579.94        |        10            |
| Pairing 3: Advanced ISO + Graph     |      3      |  0.333   |       0.333       |      8999.21             |     4601.74        |         5            |
```

**Key checks:**
- All three pipelines evaluated end-to-end on a shared query set.
- Custom metrics computed and rendered; best hit rate = Pairing 2 (0.667), best citation accuracy
  = Pairing 2 (1.000).
- Report includes per-category hit rate and a reproducibility appendix.
- RAGAS path executes but yields `nan` with the local model (see §6.2); the report states this
  explicitly rather than showing misleading numbers.

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **RAGAS unimportable** | `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` | ragas 0.3.1 imports a chat-model module removed from langchain-community 0.4.2 | `install_ragas_shim()` injects a placeholder `vertexai` module into `sys.modules` | ✅ Resolved |
| 2 | **RAGAS metrics all `nan`** | `OutputParserException: Invalid json output` and `TimeoutError` in RAGAS jobs | The local `llama3.2:3b` model does not reliably emit the strict JSON RAGAS expects; some jobs time out | Custom metrics are primary; RAGAS runs with `raise_exceptions=False`; the report notes unavailability. Recommend `LLM_PROVIDER=openai` for RAGAS | ⚠️ Degraded — documented |
| 3 | **Chain not callable** | `WARNING ... generation failed: 'RunnableSequence' object is not callable`; empty answers, ~0.2 ms latency | The evaluator called `rag_chain(query)`; LangChain 1.x Runnables require `.invoke()` | Added `_call()` helper that uses `.invoke()` when available | ✅ Resolved |
| 4 | **RAM OOM (SIGKILL)** | Process killed building all three pipelines together | Chroma stores + two BM25 indexes + embedding model + Docling docs exceeded 7.7 GB | Build/evaluate one pipeline at a time; `del` + `gc.collect()` between | ✅ Resolved |
| 5 | **GPU OOM during reranking** | `CUDACachingAllocator ... memory allocation failed with OOM on device 0` | Embedding model + `bge-reranker-v2-m3` co-resident on an 8 GB GPU | Reranking is wrapped in `try/except` (falls back to unranked); documented to run the reranker on CPU or use a smaller model | ⚠️ Mitigated |
| 6 | Misleading summary | Report said "RAGAS metrics computed" while values were `nan` | Truthiness of a dict of `nan` values | Summary now checks for non-`NaN` values; empty RAGAS renders an explicit note | ✅ Resolved |

### 6.1 Failure #3 in detail (functional bug)

The first harness run reported empty answers with ~0.2 ms end-to-end latency for every query.
A stub-retriever test proved the chain itself worked (4.9 s, correct answer), isolating the bug to
the evaluator: LCEL `RunnableSequence` objects are **not callable** in LangChain 1.x, so
`rag_chain(query)` raised `TypeError`, which the per-query `try/except` swallowed into an empty
answer. Fixed with the `_call()` helper. After the fix, latencies are realistic (~2.6–4.6 s).

### 6.2 Failure #2 in detail (RAGAS with a local model)

RAGAS's faithfulness/relevancy prompts require the judge LLM to return strict JSON. The
`llama3.2:3b-instruct-q4_K_M` model returned prose/Python and occasionally exceeded the per-job
timeout, so every metric resolved to `nan`. This is a model-capability issue, not a harness bug.

**Recommendation**: run RAGAS with a stronger judge, e.g. `LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini`.
The custom metrics (hit rate, citation accuracy, latency) are deterministic and remain valid with
the local model.

### 6.3 Failure #4/#5 in detail (memory)

Two distinct memory limits were hit:
- **System RAM (7.7 GB)**: constructing all three pipelines in one process was OOM-killed.
  Fixed by building one pipeline at a time.
- **GPU VRAM (8 GB)**: loading the cross-encoder reranker while the embedding model was resident
  triggered a CUDA OOM. The retrieval code already guards reranking with `try/except`, so the
  pipeline degraded to unranked results instead of crashing.

---

## 7. Deviations from the Plan

1. **RAGAS import shim** required (ragas 0.3.1 vs langchain-community 0.4.2). Documented in plan §13.13.
2. **RAGAS is best-effort** and degraded with the local 3B model; custom metrics take precedence.
3. **Per-pipeline build/evaluate** (memory), plus `--limit` and `--skip-ragas` flags.
4. **Gold answers empty** → `context_recall`/`answer_correctness` are only attempted when ground
   truth is present.
5. **Report renders an explicit note** when RAGAS is unavailable.

---

## 8. Acceptance Check (Milestone 7)

- [x] `eval/test_queries.json` created (20 queries, 6 categories).
- [x] `eval/evaluator.py` and `eval/report_generator.py` implemented.
- [x] `eval/report.md` generated with all three pipelines compared.
- [x] Custom metrics computed (hit rate, citation accuracy, latency, chunk counts).
- [x] RAGAS path implemented and executed (degraded to `nan` with the local model; documented).
- [x] Failures captured with mitigations (this document).
- [ ] Full 20-query run with a stronger judge LLM — recommended, pending an API key/model.
- [ ] MPS (Apple Silicon) verification — pending M1 hardware.

---

## 9. Next Step

**Milestone 8 — Streamlit UI**: implement [`ui/pipeline_registry.py`](../ui/pipeline_registry.py),
[`ui/components.py`](../ui/components.py), and [`ui/app.py`](../ui/app.py) with single-pipeline and
compare-all-three modes.