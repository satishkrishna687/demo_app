# Milestone 13 — Cross-Pair Comparison Report (`RAG_EVAL_METRICS.md` deliverable)

**Status**: ✅ Complete
**Date**: 2026-09-24
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §7 (Milestone 13), §7.1 (Metric Mapping)
**Spec reference**: [`RAG_EVAL_METRICS.md`](../RAG_EVAL_METRICS.md) (Output Format §1–§5, Rules)

---

## 1. Objective

Aggregate the three per-pairing results into the comparison report required by
`RAG_EVAL_METRICS.md`: a summary table across all four metric categories, retrieval and generation
rankings, an efficiency tradeoff note, and an overall recommendation — with variance noted and
uncomputable metrics flagged.

---

## 2. Artifacts

| Path | Purpose |
|---|---|
| [`eval/comparison_report.py`](../eval/comparison_report.py) | Aggregates `result_pair*.json` → Markdown report |
| [`eval/RAG_EVAL_REPORT.md`](../eval/RAG_EVAL_REPORT.md) | **The deliverable** |
| [`eval/result_pair1.json`](../eval/result_pair1.json), [`result_pair2.json`](../eval/result_pair2.json), [`result_pair3.json`](../eval/result_pair3.json) | Inputs |
| [`docs/MILESTONE_13.md`](../docs/MILESTONE_13.md) | This document |

---

## 3. Method

```bash
python -m eval.comparison_report
# reads eval/result_pair{1,2,3}.json -> writes eval/RAG_EVAL_REPORT.md
```

The generator produces the five outputs the spec asks for, plus variance and flags:

| `RAG_EVAL_METRICS.md` Output Format | Section in `RAG_EVAL_REPORT.md` |
|---|---|
| 1. Summary table (all four categories) | §1 Summary Table |
| 2. Retrieval-quality ranking | §2 Rankings |
| 3. Generation-quality ranking | §2 Rankings |
| 4. Efficiency tradeoff note | §3 Efficiency Tradeoff Note |
| 5. Overall recommendation | §4 Overall Recommendation |
| Variance | §5 Variance |
| Flag uncomputable metrics | §6 Flagged / Uncomputable Metrics |

---

## 4. Results (report content)

### 4.1 Summary table (3-query subset)
| Pair | Recall@5 | nDCG@5 | MRR | Recall@5 (section) | Chunks | Avg chunk | Citation acc. | Ans. correctness | Index (s) | Retrieval (ms) | E2E (ms) | Index (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 Recursive + Parent-Child | 1.000 | 1.000 | 1.000 | N/A | 2203 | 671.3 | 0.000 | 0.497 | 109.9 | 61.1 | 3813.8 | 249.0 |
| P2 Section + Hybrid | 1.000 | 0.939 | 1.000 | 1.000 | 3827 | 333.9 | 0.722 | 0.547 | 70.9 | 64.7 | 3220.6 | 27.7 |
| P3 Advanced ISO + Graph | 0.667 | 0.667 | 0.667 | 1.000 | 1970 | 657.1 | **1.000** | **0.583** | 125.5 | **10513.3** | 11388.4 | 20.9 |

### 4.2 Rankings (from the report)
- **Retrieval (answer-term)**: Pairing 1 (1.0000) — runner-up Pairing 2 (1.0000).
- **Retrieval (section)**: Pairing 2 (1.0000) — runner-up Pairing 3 (1.0000).
- **Faithfulness**: Pairing 1 (0.4667).
- **Answer correctness**: Pairing 3 (0.5833).
- **Citation accuracy**: Pairing 3 (1.0000).

### 4.3 Efficiency tradeoff
> Pairing 3 has the best citation accuracy (1.0) but is ~4× slower end-to-end than Pairing 2
> (11388 ms vs 3221 ms). Retrieval alone is far more asymmetric: 61 ms for Pairing 1 vs 10513 ms
> for Pairing 3.

### 4.4 Overall recommendation
> For accuracy-critical ISO queries, **Pairing 3** is recommended (best citation accuracy 1.0),
> with Pairing 3 strongest on answer correctness. For latency-sensitive/interactive use,
> **Pairing 2** is the most responsive.

### 4.5 Flags
```
Pairing 1 | retrieval_section | not applicable: retrieved chunks carry no section_number
```

---

## 5. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **`nan` in summary table** | Pairing 1's section metric displayed `nan` | `None` serialized from JSON became `NaN` in the DataFrame | Added `_display()` which renders missing floats as `N/A` | ✅ Fixed |
| 2 | **Unrounded values** | Output showed `0.46666666666666673` | Raw floats not formatted | `_display()` rounds to 4 dp; `_fmt()` used in rankings/notes | ✅ Fixed |
| 3 | **Tradeoff note hid retrieval asymmetry** | Note compared only end-to-end latency | E2E is generation-dominated (~3.2–3.8 s for all pairs), masking the 160× retrieval gap | Note now also reports retrieval latency for the fastest vs quality-winning pair | ✅ Improved |
| 4 | **Ranking ties** | Pairing 1 and 2 both score 1.0000 on Recall@5 | Subset too small to separate them | Runner-up value is shown beside the winner so ties are visible | ⚠️ Documented |
| 5 | **No variance** | Only point estimates | Each pairing ran once (`--runs 1`) | Report §5 explicitly states variance is unavailable and how to enable it (`--runs 3`) | ⚠️ Documented |
| 6 | Tautological recommendation | Same pairing named for accuracy and correctness | Pairing 3 genuinely wins both on the subset | Accepted; the sentence still conveys the winner | ⚠️ Cosmetic |

### 5.1 Failure #1 in detail

`result_pair*.json` stores the section metric as `null` for Pairing 1 (correctly), but pandas
converted it to `NaN`, which `to_markdown` printed as `nan`. A `_display()` helper now copies the
summary frame and replaces missing values with `N/A` for rendering, while rankings still operate on
the numeric frame. The **flag** for the metric continues to appear in §6, satisfying the spec's
"flag rather than omit" rule.

---

## 6. Deviations

1. **Subset data** (3 queries per pairing) feeding the comparison, not the full 32.
2. **Local 3B judge** — scores are coarse.
3. **Variance not computed** (single runs); explicitly disclosed in the report.
4. **Index size approximate** (per-pairing directory walk).

---

## 7. Acceptance Check (Milestone 13)

- [x] `eval/comparison_report.py` implemented and runnable via `python -m eval.comparison_report`.
- [x] `eval/RAG_EVAL_REPORT.md` generated with all five required outputs.
- [x] Retrieval and generation rankings present.
- [x] Efficiency tradeoff note present (incl. retrieval latency).
- [x] Overall recommendation present.
- [x] Variance documented.
- [x] Uncomputable metrics flagged, not omitted.
- [x] Identical judge model/prompt/temperature across pairs (stated in §7 Methodology).
- [x] Test suite green (14 passed).

---

## 8. Next Step

**Milestone 14 — Cross-Platform Validation, CLI, and Docs**: wire `main.py`, ensure pytest/smoke
tests pass, validate on M1 + Ubuntu, write `README.md`, and capture UI screenshots.