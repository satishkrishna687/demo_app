# Project Debug Rules (Non-Obvious Only)

- `scripts/inspect_chunks.py` is the first diagnostic for any retrieval/chunking problem (counts, section coverage, oversized sections, table/graph stats).
- Index staleness is silent: `ensure_indexed()` reuses an index when collection count == document count, so changed chunking/embedding settings appear to "do nothing" until the persist directory (`data/indexes/pairN/` or `chroma_db/`) is deleted.
- Reranker (Pairing 3) failures are swallowed by `try/except` and degrade to unranked results — a broken reranker looks like "worse ordering", not an error. Check logs, not tracebacks.
- Judge scores of `None` in eval results mean parse failure after one retry (flagged in `flags`), not a crash — check the judge model actually responded with a number.
- RAGAS metrics showing `nan` are expected with the small local judge; use the custom judge in `eval/judge.py` instead.
- OOM during pipeline builds is a known constraint: never build all three pipelines at once (was OOM-killed on 8 GB, see docs/MILESTONE_7.md).
- Streamlit caches (`@st.cache_resource`) persist across reruns; stale pipelines after code changes need a cache clear, not just a page reload.
