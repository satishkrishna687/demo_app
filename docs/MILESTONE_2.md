# Milestone 2 — Model Factories

**Status**: ✅ Complete
**Date**: 2026-09-23
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §6 (Module Specifications), §7 (Milestone 2), §13 (Deviations)

---

## 1. Objective

Implement the three model factories that abstract every heavy model behind a single, provider-
and device-agnostic entry point:

- [`models/embeddings.py`](../models/embeddings.py) — local HuggingFace embeddings.
- [`models/llm.py`](../models/llm.py) — Ollama/OpenAI chat LLM switch.
- [`models/reranker.py`](../models/reranker.py) — cross-encoder reranker (Pairing 3 Stage 4).

Verify that embeddings load on the detected device, `get_llm()` returns the configured provider,
and Ollama is reachable.

---

## 2. Files Created / Modified

| Path | Action | Purpose |
|---|---|---|
| [`models/embeddings.py`](../models/embeddings.py) | Created | Cached `HuggingFaceEmbeddings` factory on the auto-detected device |
| [`models/llm.py`](../models/llm.py) | Created | `ChatOllama` / `ChatOpenAI` factory plus `ollama_reachable()` preflight |
| [`models/reranker.py`](../models/reranker.py) | Created | Cached `CrossEncoder` factory plus `rerank()` helper |
| [`models/device.py`](../models/device.py) | Modified | Added `TORCH_DISABLE_NATIVE_JIT=1` (see §7) |
| [`config.py`](../config.py) | Modified | Aligned default `LLM_MODEL` to the installed Ollama model |
| [`.env.example`](../.env.example) | Modified | Same default update with explanatory comment |
| [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) | Modified | Added §13.4–§13.6 documenting new deviations |

---

## 3. Design

### 3.1 Embeddings factory
- Wraps `langchain_huggingface.HuggingFaceEmbeddings`.
- Device resolved via [`models.device.get_device()`](../models/device.py) → `cuda` on this host.
- `encode_kwargs={"normalize_embeddings": True}` so cosine similarity behaves as expected.
- Cached in a module-level dict keyed by `(model_name, device)` so the three pipelines do not
  reload the model. `clear_embeddings_cache()` provided for tests.
- Default model: `Qwen/Qwen3-Embedding-0.6B` (1024-dim output).

### 3.2 LLM factory
- Single hybrid switch driven by `LLM_PROVIDER` (`ollama` default, `openai` optional).
- Returns a `BaseChatModel`; consumers never import a concrete provider.
- Raises a clear `ValueError` for unsupported providers.
- `ollama_reachable()` pings `{OLLAMA_BASE_URL}/api/tags` so the CLI/UI can fail fast with a
  friendly message instead of a deep connection error.
- Cached keyed by `(provider, model, temperature)`.

### 3.3 Reranker factory
- Wraps `sentence_transformers.CrossEncoder` on the auto-detected device.
- `rerank(query, docs, top_k)` scores `(query, chunk)` pairs jointly and returns the top-k in
  descending relevance order; returns `[]` for empty input.
- Default model: `BAAI/bge-reranker-v2-m3`.

---

## 4. Commands Executed

```bash
# Preflight: is Ollama installed and what models exist?
which ollama                      # /usr/local/bin/ollama
ollama list                       # llama3.2:3b-instruct-q4_K_M (2.0 GB)

# LLM factory + reachability
.venv/bin/python -c "from models.llm import get_llm, ollama_reachable; ..."

# Embeddings on CUDA
.venv/bin/python -c "from models.embeddings import get_embeddings; ..."

# Triton diagnosis
grep -rn "DISABLE" .venv/lib/python3.12/site-packages/torch/_native/
#   -> common_utils.py: TORCH_DISABLE_NATIVE_JIT=1

# Reranker + real LLM generation
.venv/bin/python -c "from models.reranker import rerank; ..."
.venv/bin/python -c "from models.llm import get_llm; get_llm().invoke('...')"
```

---

## 5. Verification Results

### 5.1 LLM factory
```
LLM_PROVIDER: ollama | LLM_MODEL: llama3.2:3b-instruct-q4_K_M
ollama_reachable: True
get_llm() -> ChatOllama | model: llama3
get_llm(bogus) raised ValueError as expected
```

### 5.2 LLM generation (real call)
```
provider: ollama model: llama3.2:3b-instruct-q4_K_M
LLM reply: 'OK'
OK llm generation
```

### 5.3 Embeddings (CUDA)
```
device: cuda (NVIDIA GeForce RTX 3050)
embed_query dim: 1024 (load+embed 16.7s)
embed_documents: 2 x 1024
OK embeddings on CUDA
```

### 5.4 Reranker (CUDA)
```
device: cuda (NVIDIA GeForce RTX 3050)
[rerank 144.2s] top:
  - The maximum AC charging voltage is 250 V.
  - AC charging current limits are specified in Table 12.
OK reranker
```
(The 144 s includes the one-time ~2.3 GB model download; ranking is correct — the voltage
sentence outranks the unrelated Paris sentence, which was correctly excluded from the top-2.)

---

## 6. Failures and Mitigations

| # | Failure | Symptom | Root cause | Mitigation | Outcome |
|---|---|---|---|---|---|
| 1 | **Triton needs a C compiler** | `RuntimeError: Failed to find C compiler` on first CUDA op | PyTorch 2.14 routes ordinary ops (Qwen3 RoPE matmul) through a native DSL backed by Triton, which JIT-compiles a C shim requiring `gcc`/`clang`; none installed and `sudo` needs a password | Set `TORCH_DISABLE_NATIVE_JIT=1` in [`models/device.py`](../models/device.py) before torch import → eager kernels | ✅ Resolved — CUDA embeddings + reranker work |
| 2 | `attn_implementation="eager"` did not help | Same Triton error | The offending op is RoPE, not attention | Abandoned; used the global kill switch instead | ✅ Resolved (see #1) |
| 3 | `TRITON_INTERPRET=1` did not help | `AttributeError: 'ConstTensorWrapper' object has no attribute 'untyped_storage'` | Triton interpreter mode is incompatible with this torch build | Abandoned | ✅ Resolved (see #1) |
| 4 | **`LLM_MODEL=llama3` not installed** | `ollama list` showed only `llama3.2:3b-instruct-q4_K_M` | Config default did not match the environment | Aligned default in [`config.py`](../config.py) + [`.env.example`](../.env.example) | ✅ Resolved — generation works |
| 5 | **OpenAI provider without key** | `OpenAIError: Missing credentials...` | No `OPENAI_API_KEY` set (expected; Ollama is the default) | Documented; set `OPENAI_API_KEY` (or `LLM_PROVIDER=ollama`) to use OpenAI | ⚠️ Expected — Ollama path used |
| 6 | HF Hub rate-limit warning | `You are sending unauthenticated requests to the HF Hub` | No `HF_TOKEN` configured | Documented in plan §13.6; set `HF_TOKEN` for faster/reliable pulls | ⚠️ Non-blocking |
| 7 | Slow first load | Embeddings ~121 s, reranker ~144 s | One-time model download (~1.2 GB + ~2.3 GB) | Models cached in `~/.cache/huggingface`; subsequent loads are seconds | ✅ Resolved (cached) |

### 6.1 Failure #1 in detail (the significant one)

The traceback terminated in Qwen3's RoPE computation:

```python
# transformers/models/qwen3/modeling_qwen3.py:132
freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
# -> torch/_native/registry.py:956 eager_router -> _dispatch -> torch.ops._native.<op>
# -> triton/runtime/driver.py -> compile_module_from_file -> RuntimeError: Failed to find C compiler
```

So even a plain matrix multiply in the model's positional-encoding step was routed through the
Triton-backed native registry. Because the machine has **no compiler and no passwordless sudo**,
installing `build-essential` was not possible during this session. Disabling the native DSL via
`TORCH_DISABLE_NATIVE_JIT=1` was the correct, zero-privilege fix.

**Cross-platform note**: macOS ships `clang` via Xcode Command Line Tools, so Apple Silicon may
not need the flag — but the flag is harmless there and is applied uniformly for parity. On a
machine that *does* have a compiler, either approach works.

---

## 7. Deviations from the Plan

1. **`TORCH_DISABLE_NATIVE_JIT=1`** added to [`models/device.py`](../models/device.py) — not in
   the original plan. Required on this torch build without a C compiler. Documented in plan §13.4.
2. **Default `LLM_MODEL` changed** from `llama3` to `llama3.2:3b-instruct-q4_K_M` — matches the
   installed Ollama model. Documented in plan §13.5.
3. **Embeddings are local-only** — per the agreed hybrid decision, only the LLM is switchable;
   OpenAI embeddings remain a documented manual swap, not a config flag.
4. **LangChain 1.x imports** — the factories use `langchain_huggingface`, `langchain_ollama`,
   `langchain_openai`, and `langchain_core` directly (no legacy `langchain.*` imports needed
   here, so no `langchain_classic` dependency in these three files).

---

## 8. Acceptance Check (Milestone 2)

- [x] `models/embeddings.py` implemented; loads on the detected device (`cuda`) and embeds.
- [x] `models/llm.py` implemented; `get_llm()` returns the configured provider (`ChatOllama`).
- [x] Ollama reachable; real generation call succeeded.
- [x] `get_llm(provider="openai")` fails with a clear credential error (documented).
- [x] `get_llm(provider="bogus")` raises `ValueError`.
- [x] `models/reranker.py` implemented; `rerank()` orders documents by relevance.
- [x] Failures captured with mitigations (this document).
- [ ] MPS (Apple Silicon) verification — pending M1 hardware.
- [ ] `HF_TOKEN` set for faster model downloads — optional, not done.

---

## 9. Next Step

**Milestone 3 — Multi-PDF Ingestion**: implement [`ingestion/loader.py`](../ingestion/loader.py)
with `load_iso_documents(pdf_dir, use_ocr=True)`, iterating all three PDFs in
[`Iso/`](../Iso), deriving `document_id` from each filename, stripping ISO headers/footers, and
preserving tables as Markdown via Docling. Test: document count > 0 per file, `document_id`
present, tables appear as Markdown.
