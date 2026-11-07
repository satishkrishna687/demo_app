# API-Only Model Providers (OpenRouter) & Cloud Deployment

This document describes the migration from local models (Ollama + local
sentence-transformers embeddings/reranker) to fully API-based models served via
**OpenRouter**, the new cloud deployment mode, and how to set everything up.

## What changed

| Area | Change |
|---|---|
| [`config.py`](../config.py) | New env vars: `EMBEDDING_PROVIDER`, `EMBEDDING_BASE_URL`, `EMBEDDING_API_KEY_ENV`, `LLM_BASE_URL`, `LLM_API_KEY_ENV`, `CLOUD_MODE` |
| [`models/embeddings.py`](../models/embeddings.py) | `openai` provider branch (any OpenAI-compatible `/embeddings` endpoint, e.g. OpenRouter); `_NormalizedEmbeddings` L2-normalizes API vectors to match the local cosine semantics; `_RetryEmbeddings` retries HTTP 429 with exponential backoff; empty chunks are substituted with a space for encoder input only (APIs reject empty strings); batch size reduced to 64 to fit free-tier token quotas |
| [`models/llm.py`](../models/llm.py) | `get_llm()` supports `base_url` / `api_key` overrides for OpenAI-compatible endpoints |
| [`models/runtime.py`](../models/runtime.py) (new) | In-memory runtime override store (explicit args > runtime overrides > config defaults). API keys never touch disk or logs |
| [`ui/settings.py`](../ui/settings.py) (new) | Sidebar section to pick model names and enter API keys at runtime (stored in `st.session_state` only) |
| [`ui/app.py`](../ui/app.py) | Renders the settings sidebar; pipeline cache is keyed on the settings signature so changed models rebuild the pipeline |
| [`ui/pipeline_registry.py`](../ui/pipeline_registry.py) | `CLOUD_MODE=1` skips Docling entirely and recovers chunks from the shipped Chroma index (`data/indexes/pairN`); Pairing 1 is unsupported in cloud mode (no persisted vector store) |
| [`requirements-cloud.txt`](../requirements-cloud.txt) (new) | API-only dependency set (no torch / docling / sentence-transformers); `chromadb` pinned for index compatibility |
| [`.gitignore`](../.gitignore) | Ships `data/indexes/pair2` + `pair3` (pre-built indexes); `pair1` stays local-only (~140 MB, unsupported in cloud) |
| [`.env.example`](../.env.example) | OpenRouter defaults: generator `qwen/qwen3.7-flash`, embeddings `qwen/qwen3-embedding-8b`, one shared `OPENROUTER_API_KEY` |
| [`tests/test_embeddings_api.py`](../tests/test_embeddings_api.py) (new) | Deterministic tests (fake backend, no network) for provider switching, runtime overrides, normalization, table stripping, empty-chunk handling, and rate-limit retries |

## Models used

- **Generator (chat LLM):** `qwen/qwen3.7-flash` via OpenRouter
- **Embeddings:** `qwen/qwen3-embedding-8b` via OpenRouter (Nebius-served; has a
  tight token quota — the retry wrapper handles the 429s)
- **Reranker (Pairing 3 only):** still local (`BAAI/bge-reranker-v2-m3`). In
  cloud mode it degrades gracefully to unranked results (existing try/except).

## Setup (local development)

1. Copy the template and paste your key:

   ```bash
   cp .env.example .env
   # edit .env → set OPENROUTER_API_KEY=sk-or-...
   ```

2. Install dependencies (existing setup unchanged — full `requirements.txt` still
   works for local mode):

   ```bash
   pip install -r requirements.txt
   ```

3. Rebuild the indexes with the new embedding model (required after any
   embedding-model change — vectors from different models are incompatible):

   ```bash
   rm -rf chroma_db data/indexes/pair1 data/indexes/pair2 data/indexes/pair3
   python -m eval.run_pairing --pairing 1 --limit 1
   python -m eval.run_pairing --pairing 2 --limit 1
   python -m eval.run_pairing --pairing 3 --limit 1
   python scripts/inspect_chunks.py   # verify chunk/index quality
   ```

   Note: bulk embedding may hit rate limits; the retry wrapper rides through
   them automatically. If a run dies mid-way, delete that pairing's index dir
   and re-run for a clean rebuild.

## Running

**Full local mode** (Docling ingestion path, all three pairings):

```bash
streamlit run ui/app.py
```

**Cloud-mode simulation** (exactly what runs in deployment — no Docling, loads
the pre-built indexes):

```bash
CLOUD_MODE=1 streamlit run ui/app.py
```

- Pairing 1 shows a clear "not available in cloud mode" message (expected).
- Pairings 2 and 3 load `data/indexes/pair2` / `pair3` and answer via OpenRouter.

**Runtime model settings (both modes):** the sidebar's *Models & API keys*
section lets any user override the embedding model, generator model, and API
keys per session. Leave a field blank to use the `.env` defaults. Keys live in
the browser session only.

## Cloud deployment (Hugging Face Spaces — free)

1. Commit code + indexes:

   ```bash
   git add -A
   git commit -m "..."   # see commit message below
   git push origin main
   ```

   If the remote rejects large Chroma files (>100 MB), use git LFS:

   ```bash
   git lfs install
   git lfs track "data/indexes/**"
   git add .gitattributes data/indexes
   git commit -m "Track prebuilt indexes with LFS"
   git push origin main
   ```

2. Create a Space at <https://huggingface.co/new-space> with SDK = **Streamlit**,
   then push:

   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/iso15118-rag
   git push space main
   ```

3. In Space **Settings → Variables and secrets**:

   | Type | Name | Value |
   |---|---|---|
   | Secret | `OPENROUTER_API_KEY` | your key |
   | Variable | `CLOUD_MODE` | `1` |
   | Variable | `LLM_PROVIDER` | `openai` |
   | Variable | `LLM_MODEL` | `qwen/qwen3.7-flash` |
   | Variable | `LLM_BASE_URL` | `https://openrouter.ai/api/v1` |
   | Variable | `EMBEDDING_PROVIDER` | `openai` |
   | Variable | `EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b` |
   | Variable | `EMBEDDING_BASE_URL` | `https://openrouter.ai/api/v1` |

4. Open the Space URL. First load takes a minute or two (container boot + index
   open). Users pick Pairing 2 or 3 and can enter their own model names/keys in
   the sidebar.

## Important constraints (unchanged from AGENTS.md)

- Changing `EMBEDDING_MODEL`, `EMBEDDING_PROVIDER`, `EMBEDDING_USE_PROMPTS`, or
  `EMBEDDING_STRIP_TABLES` invalidates all Chroma indexes → rebuild locally and
  redeploy.
- Changing the generator or reranker model never requires an index rebuild.
- `ensure_indexed()` uses a count heuristic: a stale index is silently reused.
  Delete the persist directory to force a rebuild.
- API vectors are L2-normalized by `_NormalizedEmbeddings` so cosine similarity
  behaves identically to the local path.
- All vector-store writes still go through `add_documents_in_batches()`.

## Test results

`python -m pytest tests/ -q` → **47 passed** (no model loads, no network).
