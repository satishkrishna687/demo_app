# Milestone 17 — Chunk Explorer Tab

**Status**: ✅ Complete
**Date**: 2026-09-26
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) (milestone add-on)

---

## 1. Objective

Let users browse and visualise **every chunk** produced by each pairing directly in the UI —
counts, size distributions, content types, metadata coverage, and the full text of any chunk —
without building indexes or running embeddings.

---

## 2. Files Created / Modified

| Path | Action | Purpose |
|---|---|---|
| [`ui/chunk_explorer.py`](../ui/chunk_explorer.py) | Created | Cached chunk loading, metadata table, charts, filters, full-text viewer |
| [`ui/charts.py`](../ui/charts.py) | Modified | Added `histogram()`, `stacked_bar()`, `percent_bar()` |
| [`ui/app.py`](../ui/app.py) | Modified | Third tab **🧩 Chunks** |
| [`tests/test_chunk_explorer.py`](../tests/test_chunk_explorer.py) | Created | Data-shaping + chart tests (no document parsing) |

---

## 3. Design

### 3.1 Loading ([`ui/chunk_explorer.py`](../ui/chunk_explorer.py))
- `load_chunks(pairing, variant)` — `@st.cache_data`; runs the pairing's chunker on the **cached**
  parsed documents and returns `[{text, metadata}]`.
  - Pairing 1 → `parents` or `children`
  - Pairing 2 → `section chunks`
  - Pairing 3 → `table + text chunks`
- **No embeddings or vector stores are touched**, so this tab is fast and avoids the heavy pipeline
  builds / memory pressure.

### 3.2 Shaping (pure helpers, unit-tested)
- `chunks_to_df()` — one row per chunk: `idx, document_id, section_number, section_title,
  content_type, page, chars, words, heading_path, parent_id, child_index, cross_refs, preview`.
- `coverage_df()` — % of chunks carrying each metadata field.
- `content_type_df()` / `per_document_df()` — counts for charts.

### 3.3 Charts ([`ui/charts.py`](../ui/charts.py))
- Chunk-size **histogram**.
- **Stacked bars** for content type and chunks-per-document.
- **Percent bar** for metadata coverage.

### 3.4 Filters + viewer
- Filters: document, content type, section substring, chunk-size range, free-text search.
- A virtualised table (`height=420`) with **single-row selection**; selecting a row shows the full
  chunk text and its metadata.
- An opt-in **All-pairings overview** button chunks all three pairings and compares counts/sizes.

---

## 4. Commands Executed

```bash
python -m py_compile ui/app.py ui/chunk_explorer.py ui/charts.py
python -m pytest tests/ -q
python -c "from streamlit.testing.v1 import AppTest; AppTest.from_file('...').run()"
```

---

## 5. Verification Results

```
compile OK
pytest tests/ -q -> 22 passed
AppTest -> exception: []
tabs: ['💬 Ask (Chat)', '📊 Evaluation & Comparison', '🧩 Chunks']
dataframes rendered: 4
```

The Chunks tab loads the default selection (Pairing 2 section chunks), renders the metrics row,
filters, four charts, the table, and the viewer without error.

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **Pylance false positive** | Editor reported `Expected indented block` at `tests/test_chunk_explorer.py:11` | Stale/incorrect static-analysis diagnostic; the file is syntactically valid | Verified with `pytest` (22 passed) and by reading the file; no code change needed | ✅ False positive |
| 2 | **Large chunk sets** | Pairing 1 children = 8,495 rows | Rendering everything at once is heavy | Virtualised table (`height=420`) + filters + row selection instead of full render | ✅ Handled |
| 3 | **Row-selection API variance** | `event.selection.rows` may differ across Streamlit versions | API added recently | Read defensively via `getattr(getattr(event, "selection", None), "rows", [])` | ✅ Robust |
| 4 | **Cost of chunking all three** | Overview would chunk 3 pairings | ~3–4 s of work | Made it **opt-in** behind a button; per-pairing view chunks only the selection | ✅ Handled |
| 5 | **Empty selection** | No chunk chosen | User hasn't clicked a row | Shows a hint caption instead of an empty viewer | ✅ Handled |

### 6.1 Failure #1 in detail

The editor flagged an indentation error in the new test file. Running the suite proved the file is
valid (`22 passed`), and reading the file confirmed correct indentation. This was a stale
static-analysis diagnostic, not a real defect — recorded here so it isn't mistaken for a bug later.

---

## 7. Deviations

1. **Tab, not a separate page** — consistent with the Evaluation tab; shares the sidebar.
2. **All-pairings overview is opt-in** — avoids chunking all three on every render.
3. **Charts reuse `ui/charts.py`** — no new dependency (Altair is bundled with Streamlit).

---

## 8. Acceptance Check (Milestone 17)

- [x] `ui/chunk_explorer.py` implemented with cached loading.
- [x] Pairing/variant selector (P1 parents/children; P2 sections; P3 table+text).
- [x] Headline metrics (count, avg/min/max size, % with section).
- [x] Charts: size histogram, content-type stacked bar, chunks-per-document, metadata coverage.
- [x] Filters: document, content type, section, size range, free-text.
- [x] Virtualised table with single-row selection and a full-text viewer.
- [x] Opt-in all-pairings overview.
- [x] Third tab wired into `ui/app.py`.
- [x] Tests added; `AppTest` executes the full app without exception; pytest green (22 passed).
- [x] Failures captured with mitigations (this document).

---

## 9. Next Step

**Milestone 14 (final)**: cross-platform validation, `main.py` CLI, README polish, and UI
screenshots (now including the Evaluation dashboard and Chunk Explorer).