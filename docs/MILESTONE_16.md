# Milestone 16 — Evaluation & Comparison Dashboard

**Status**: ✅ Complete
**Date**: 2026-09-24
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) (milestone add-on),
[`RAG_EVAL_METRICS.md`](../RAG_EVAL_METRICS.md)

---

## 1. Objective

Extend the Streamlit UI with a second tab that visualises the evaluation results and the
cross-pairing comparison using charts and tables, without needing to read the Markdown reports.

---

## 2. Files Created / Modified

| Path | Action | Purpose |
|---|---|---|
| [`ui/eval_data.py`](../ui/eval_data.py) | Created | Loads `eval/result_pair*.json` + `RAG_EVAL_REPORT.md`; builds DataFrames |
| [`ui/charts.py`](../ui/charts.py) | Created | Altair chart builders (grouped bars, bubble scatter, heatmap, simple bars) |
| [`ui/eval_dashboard.py`](../ui/eval_dashboard.py) | Created | `render_dashboard()` — the dashboard layout |
| [`ui/app.py`](../ui/app.py) | Modified | Two tabs: **💬 Ask (Chat)** and **📊 Evaluation & Comparison** |
| [`tests/test_eval_dashboard.py`](../tests/test_eval_dashboard.py) | Created | Regression tests for data shaping + chart building |

---

## 3. Design

### 3.1 Data layer ([`ui/eval_data.py`](../ui/eval_data.py))
- `load_results()` reads `eval/result_pair{1,2,3}.json` (skips missing) and adds short labels
  (e.g. `P2 Section+Hybrid`).
- `summary_df()` — one row per pairing across retrieval, generation, chunking, efficiency.
- `tidy_metric_df()` — long `[Pair, Metric, Value]` frames for grouped bars.
- `category_recall_df()` — per-category Recall@5 (index named `category`).
- `flags_df()` and `report_text()` — flagged metrics and the full comparison report.
- No heavy model imports, so the tab loads instantly.

### 3.2 Charts ([`ui/charts.py`](../ui/charts.py))
- `grouped_bar()` — retrieval and generation metric comparisons.
- `bubble_quality_vs_latency()` — x = E2E ms, y = citation accuracy, size = index size.
- `heatmap_recall()` — category × pairing Recall@5.
- `simple_bar()` — chunk count / average chunk size.

### 3.3 Dashboard ([`ui/eval_dashboard.py`](../ui/eval_dashboard.py))
Summary table, retrieval bars, generation bars, quality-vs-latency bubble, category heatmap,
chunking charts, efficiency table, flagged-metrics table, and an expander with the full
`RAG_EVAL_REPORT.md`. A **Refresh data** button clears the Streamlit data cache. If results are
missing, it shows the exact commands to generate them (graceful empty state).

### 3.4 Integrate ([`ui/app.py`](../ui/app.py))
The app now renders two tabs (chat + dashboard) sharing the same sidebar configuration.

---

## 4. Commands Executed

```bash
python -m py_compile ui/app.py ui/eval_dashboard.py ui/eval_data.py ui/charts.py
python -c "from ui import eval_data, charts; ..."      # data + chart construction
python -c "from streamlit.testing.v1 import AppTest; AppTest.from_file('...').run()"
python -m pytest tests/ -q
```

---

## 5. Verification Results

```
compile OK
results: 3 | missing: []
summary shape: (3, 18)
retrieval rows: 9 | generation rows: 12 | category shape: (3, 3)
charts: Chart Chart Chart LayerChart Chart

AppTest -> exception: []
tabs: ['💬 Ask (Chat)', '📊 Evaluation & Comparison']
dataframes rendered: 3

pytest tests/ -q -> 15 passed
```

The dashboard renders the summary, retrieval/generation bars, quality-vs-latency bubble, the
category heatmap, chunk charts, and the efficiency/flags tables without error.

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **`KeyError: 'category'`** | `melt(... id_vars='category')` raised `KeyError` | `category_recall_df` returned a frame with an unnamed index, so `reset_index()` produced `index`, not `category` | Set `df.index.name = "category"` before returning | ✅ Fixed |
| 2 | **Deprecated chart width arg** | Risk of `use_container_width` deprecation warnings in Streamlit 1.64 | The API now prefers `width="stretch"` | Used `width="stretch"` for `st.dataframe`/`st.altair_chart` after introspecting the signatures | ✅ Avoided |
| 3 | Empty-state handling | Blank tab if no results exist | `eval/result_pair*.json` may be absent | Dashboard shows the exact generation commands when results are missing | ✅ Handled |
| 4 | App import path (context) | `ModuleNotFoundError: config` (reported earlier) | Streamlit puts `ui/` on `sys.path` | Root-path bootstrap already added in Milestone 8; new modules follow the same pattern | ✅ Pre-existing fix |
| 5 | NaN handling | Missing metrics (e.g. Pairing 1 section recall) could break charts | `None` → `NaN` | Charts drop `NaN` rows; summary table renders them blank | ✅ Handled |

### 6.1 Failure #1 in detail

The category heatmap could not be built:
```
KeyError: "The following id_vars or value_vars are not present in the DataFrame: ['category']"
```
`category_recall_df()` produced a DataFrame whose index was unnamed; `reset_index()` therefore
created a column called `index`, and the subsequent `melt(id_vars="category")` failed. Naming the
index (`df.index.name = "category"`) fixed it. This was caught by both the direct chart test and
the `AppTest` run — reinforcing the value of executing the script rather than only checking imports.

---

## 7. Deviations

1. **Tabs instead of separate pages** — keeps the shared sidebar and avoids a second entry script;
   a `pages/` layout can be added later if desired.
2. **Charts read precomputed JSON** — the dashboard does not trigger evaluations in-app (evaluation
   runs are long); it links users to the CLI commands instead.
3. **Altair for charts** — bundled with Streamlit, no new dependency.

---

## 8. Acceptance Check (Milestone 16)

- [x] `ui/eval_data.py`, `ui/charts.py`, `ui/eval_dashboard.py` implemented.
- [x] `ui/app.py` exposes Chat and Evaluation & Comparison tabs.
- [x] Summary table across all four metric categories.
- [x] Retrieval and generation grouped bars.
- [x] Quality-vs-latency bubble chart.
- [x] Per-category Recall@5 heatmap.
- [x] Chunking charts, efficiency table, flags table, and the full report expander.
- [x] Graceful empty state when results are missing.
- [x] `AppTest` executes the full app without exception; pytest green (15 passed).
- [x] Failures captured with mitigations (this document).

---

## 9. Next Step

**Milestone 14 (final)**: cross-platform validation, `main.py` CLI, README polish, and UI
screenshots (now including the dashboard).