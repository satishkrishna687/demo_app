# Milestone 8 — Streamlit UI

**Status**: ✅ Complete
**Date**: 2026-09-24
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §6.9, §7 (Milestone 8), §13 (Deviations)

---

## 1. Objective

Provide a Streamlit Q&A interface with a pipeline selector, chat history with source citations,
latency badges, an optional section filter, chat export, and a compare-all-three mode.

---

## 2. Files Created / Modified

| Path | Action | Purpose |
|---|---|---|
| [`ui/pipeline_registry.py`](../ui/pipeline_registry.py) | Created | Lazy, per-pipeline construction + query helpers |
| [`ui/components.py`](../ui/components.py) | Created | Source panel, answer, chat history, export button |
| [`ui/app.py`](../ui/app.py) | Created | Main Streamlit app |
| [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) | Modified | Added §13.14 |

---

## 3. Design

### 3.1 Registry ([`ui/pipeline_registry.py`](../ui/pipeline_registry.py))
- `load_documents()` — `@st.cache_resource`, one load per server process (Docling JSON cache
  makes it fast).
- `build_pipeline(name)` — constructs one pipeline (uncached).
- `get_pipeline(name)` — `@st.cache_resource` per name; single mode therefore holds at most one
  pipeline in memory.
- `run_query(pipeline, query, section_filter)` — retrieval (with optional Pairing 2 section
  filter) + generation.
- `compare_all(query, section_filter)` — builds each pipeline **in turn**, runs the query, then
  releases it and `gc.collect()`s (memory-safe; see §6.1).

### 3.2 Components ([`ui/components.py`](../ui/components.py))
- `render_source_panel()` — expandable list of retrieved chunks with `document_id`,
  `section_number`, `content_type`, `page_number`, and cross-references.
- `render_answer()` — answer + optional latency badge.
- `render_chat_history()` / `export_chat_button()` — conversation rendering and Markdown export.

### 3.3 App ([`ui/app.py`](../ui/app.py))
- Sidebar: pipeline selector (defaults to Pairing 3), single/compare mode, Pairing 2 section
  filter, device + provider caption, clear chat, export chat.
- Main: chat history and `st.chat_input`; single mode renders one answer with sources; compare
  mode renders all three side-by-side.
- Errors are surfaced with `st.error(...)` rather than swallowed.

---

## 4. Commands Executed

```bash
# Compile + import checks
.venv/bin/python -m py_compile ui/app.py ui/components.py ui/pipeline_registry.py
.venv/bin/python -c "import ui.components, ui.pipeline_registry as r; print(r.PIPELINE_NAMES)"

# Headless launch + health probe
nohup .venv/bin/streamlit run ui/app.py --server.headless true --server.port 8501 &
curl -s http://localhost:8501/_stcore/health      # -> ok
curl -s -o /dev/null -w '%{http_code}' http://localhost:8501/   # -> 200

# Registry logic test with fakes (no heavy model loads)
.venv/bin/python -c "from ui.pipeline_registry import run_query; ..."
```

---

## 5. Verification Results

```
=== compile check ===
compile OK
=== streamlit import surface ===
import OK | pipelines: ('Pairing 1: Recursive + Parent-Child',
                        'Pairing 2: Section + Hybrid', 'Pairing 3: Advanced ISO + Graph')
=== launch headless ===
health: ok
root http: 200
Uvicorn server started on :::8501
=== registry logic ===
no-filter: 'answer for: voltage?' | sources: 1 {'section_number': '1'}
filtered : 'answer for: voltage?' | sources: 1
run_query OK (callable + chain.invoke paths)
```

**Key checks:**
- All three UI modules compile and import cleanly.
- Streamlit serves the app headlessly (health `ok`, HTTP 200) with no startup errors.
- `run_query()` works for both the plain retriever path and the Pairing 2 section-filter path.
- Errors are surfaced in the UI (`st.error`) rather than silently dropped.

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **Compare mode would OOM** | Building all three pipelines together was SIGKILLed in Milestone 7 | Chroma stores + BM25 indexes + embedding model exceed 7.7 GB | Single mode caches only the selected pipeline; `compare_all()` builds each pipeline in turn, then `del` + `gc.collect()` | ✅ Resolved (memory-safe; slower compare) |
| 2 | **`ui/app.py` not importable offline** | Importing `app.py` outside Streamlit executes `st.set_page_config`/widgets and errors | Streamlit scripts run top-to-bottom | Verification uses `py_compile` + a live headless server + logic tests of the importable registry | ✅ Resolved |
| 3 | **WSL file-watcher warning** | `Detected WSL. Using poll-based file watching` | Watchdog is unreliable on WSL filesystems | Non-blocking; Streamlit falls back to polling. Optionally set `server.fileWatcherType = "poll"` | ⚠️ Non-blocking |
| 4 | **Section filter scope** | Filter has no effect for Pairings 1 and 3 | Only Pairing 2 exposes a vector store + `filtered_retrieve` | The filter field is shown only for Pairing 2; `run_query` ignores it otherwise | ✅ By design |
| 5 | Slow first question | First query per pipeline triggers embedding/model load | One-time pipeline construction (cached afterwards) | `@st.cache_resource` keeps it to once per process; spinner via `show_spinner` | ✅ Acceptable |

### 6.1 Failure #1 in detail

The plan's `pipeline_registry.py` built all three pipelines at startup into one cache and returned
them in a dict. In this environment that reliably exhausts memory (proven in Milestone 7). The
registry was therefore redesigned around **on-demand, single-pipeline** construction:

- Single mode: `get_pipeline(name)` caches exactly one pipeline, so memory holds one pipeline.
- Compare mode: `compare_all()` deliberately bypasses the cache and builds/releases each pipeline
  sequentially, trading speed for a bounded memory footprint.

This keeps the UI usable on an 8 GB machine while preserving the compare feature.

### 6.2 Post-verification fix: `ModuleNotFoundError: No module named 'config'`

**Reported on first real launch.** Running `streamlit run ui/app.py` failed at import time:

```
File "ui/app.py", line 12, in <module>
    from config import LLM_MODEL, LLM_PROVIDER
ModuleNotFoundError: No module named 'config'
```

**Root cause**: Streamlit executes the script with the script's own directory (`ui/`) on
`sys.path`, not the repository root — so top-level project imports (`config`, `models`, …) cannot be
resolved. The original Milestone 8 verification only probed the server's health endpoint
(`/_stcore/health`), which succeeds at server startup; the script itself runs only when a browser
client connects, so the error was not caught.

**Mitigation**:
- [`ui/app.py`](../ui/app.py) and [`ui/pipeline_registry.py`](../ui/pipeline_registry.py) now insert
  the repository root (`Path(__file__).resolve().parents[1]`) into `sys.path` before importing
  project modules.
- Added [`tests/test_ui_app.py`](../tests/test_ui_app.py), which executes the app headlessly with
  Streamlit's `AppTest` and asserts no exception — this catches script-execution errors that a
  health probe misses.

**Verification**:
```
AppTest -> exception: ElementList()   (empty = none)
sidebar selectbox options: ['Pairing 1...', 'Pairing 2...', 'Pairing 3...']
title present: True
pytest tests/ -q -> 15 passed
```

---

## 7. Deviations from the Plan

1. **Lazy per-pipeline loading** replaces the plan's "build all three once" registry (memory).
2. **Compare mode is uncached and sequential** (memory-safe, slower).
3. **Section filter limited to Pairing 2** (the only pipeline with `filtered_retrieve`).
4. **Device/provider caption** added to the sidebar for operational visibility.

---

## 8. Acceptance Check (Milestone 8)

- [x] `ui/pipeline_registry.py`, `ui/components.py`, `ui/app.py` implemented.
- [x] All modules compile and import cleanly.
- [x] Streamlit app serves at `localhost:8501` (health `ok`, HTTP 200).
- [x] `AppTest` executes `ui/app.py` without exception (added after the reported import bug).
- [x] Pipeline selector present (single + compare modes).
- [x] Answers render with source chunks and section metadata.
- [x] Chat export produces Markdown.
- [x] Errors surfaced via `st.error`.
- [x] Failures captured with mitigations (this document).
- [ ] Interactive click-through on M1 — pending M1 hardware.
- [ ] UI screenshots — planned for Milestone 9.

---

## 9. Next Step

**Milestone 9 — Cross-Platform Validation and Docs**: write `README.md` (per-OS setup + run
instructions), update `main.py` to wire all three pairings, add `tests/smoke_test.py`, and capture
UI screenshots.