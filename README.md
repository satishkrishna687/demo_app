# Rags_2_Riches — ISO 15118 RAG Pipeline

A local, cross-platform Retrieval-Augmented Generation (RAG) system that ingests the three
**ISO 15118** specification PDFs (Parts 1, 2, and 3) and answers engineering questions with
citations. It implements **three chunking × retrieval "pairings"**, evaluates them rigorously, and
ships with a chat UI and a comparison report.

Runs on:

- **Apple Silicon (M1/M2/M3)** using the **MPS** backend, and
- **Ubuntu + NVIDIA GPU** using **CUDA** (verified on an RTX 3050).

Embeddings and the reranker run **locally** (no API key required). The chat **LLM is switchable**:
default is a local **Ollama** model; you can opt into **OpenAI** via a single `.env` flag.

---

## Table of contents

1. [What you get](#1-what-you-get)
2. [How it works](#2-how-it-works)
3. [Repository structure](#3-repository-structure)
4. [Prerequisites](#4-prerequisites)
5. [Setup on Apple Silicon (M1/M2/M3)](#5-setup-on-apple-silicon-m1m2m3)
6. [Setup on Ubuntu + NVIDIA](#6-setup-on-ubuntu--nvidia)
7. [Configuration reference (.env)](#7-configuration-reference-env)
8. [Quickstart](#8-quickstart)
9. [Using the chat UI](#9-using-the-chat-ui)
10. [Evaluation & reports](#10-evaluation--reports)
11. [Commands cheat sheet](#11-commands-cheat-sheet)
12. [Troubleshooting](#12-troubleshooting)
13. [Cross-platform notes](#13-cross-platform-notes)
14. [Project documentation index](#14-project-documentation-index)
15. [FAQ](#15-faq)

---

## 1. What you get

| Feature | Description |
|---|---|
| **Multi-PDF ingestion** | Parses `Iso/ISO-15118-1.pdf`, `-2`, `-3` with Docling (layout-aware; tables preserved as Markdown). |
| **Pairing 1** | Recursive chunking + **Parent-Child** retrieval. |
| **Pairing 2** | Section/clause chunking + **metadata-filtered hybrid** search (vector + BM25). |
| **Pairing 3** | Table-first + clause chunking, a **cross-reference graph**, graph-expanded hybrid retrieval + **cross-encoder reranking**. |
| **Chat UI** | Streamlit app with a pipeline selector, source citations, latency badges, and a **compare-all-three** mode. |
| **Evaluation harness** | 32-query gold set, deterministic retrieval metrics, an LLM judge, and per-pairing + cross-pairing reports (per [`RAG_EVAL_METRICS.md`](RAG_EVAL_METRICS.md)). |
| **Local-first** | Local embeddings + reranker; LLM switchable between Ollama and OpenAI. |
| **Cross-platform** | One codebase; device auto-detected (`cuda` → `mps` → `cpu`). |

---

## 2. How it works

```
PDFs (Iso/*.pdf)
   │  Docling (layout + tables → Markdown), header/footer stripping, JSON cache
   ▼
Ingestion ──► Chunking (per pairing) ──► Embeddings (local, on GPU/MPS)
   │                                          │
   │                                          ▼
   │                                    Chroma vector store  (+ BM25 for hybrid)
   ▼                                          │
Knowledge graph (Pairing 3 cross-references)  │
   │                                          │
   └──────────────► Retriever (per pairing) ◄─┘
                          │
                          ▼
                    RAG chain (LCEL)  ──►  LLM (Ollama or OpenAI)  ──►  Answer + citations
```

- **Embeddings**: `Qwen/Qwen3-Embedding-0.6B` (1024-dim) via `sentence-transformers`.
- **Reranker**: `BAAI/bge-reranker-v2-m3` (Pairing 3 only).
- **LLM**: `llama3.2:3b-instruct-q4_K_M` via Ollama by default.

---

## 3. Repository structure

```
Rags_2_Riches/
├── Iso/                         # INPUT: the three ISO 15118 PDFs
├── config.py                    # All settings (reads .env)
├── models/                      # Device detection + model factories
│   ├── device.py                #   cuda > mps > cpu, with safe env flags
│   ├── embeddings.py            #   local embedding factory
│   ├── llm.py                   #   Ollama/OpenAI switch
│   └── reranker.py              #   cross-encoder factory + rerank()
├── ingestion/                   # Loading + chunking strategies
│   ├── loader.py                #   multi-PDF loader (Docling + cache)
│   ├── chunking_pair1.py
│   ├── chunking_pair2.py
│   └── chunking_pair3.py
├── graph/builder.py             # Cross-reference knowledge graph (NetworkX)
├── retrieval/                   # One retriever per pairing
│   ├── pair1_retriever.py
│   ├── pair2_retriever.py
│   └── pair3_retriever.py
├── chain/rag_chain.py           # LCEL RAG chain
├── vectorstore/store.py         # Batched, Chroma-safe indexing
├── eval/                        # Evaluation harness
│   ├── gold_seeds.json          #   32 seed queries
│   ├── build_gold_set.py        #   derives gold sections/answers
│   ├── metrics.py               #   Recall@k, Precision@k, MRR, nDCG@k
│   ├── judge.py                 #   LLM judge + citation accuracy
│   ├── runner.py                #   per-query + aggregate metrics
│   ├── run_pairing.py           #   CLI: evaluate one pairing
│   ├── pair_report.py           #   per-pairing Markdown report
│   ├── comparison_report.py     #   cross-pairing report (RAG_EVAL_METRICS.md)
│   └── RAG_EVAL_REPORT.md       #   generated final comparison
├── ui/                          # Streamlit UI
│   ├── app.py
│   ├── pipeline_registry.py
│   └── components.py
├── tests/                       # pytest suite (fast) + smoke tests
├── docs/MILESTONE_1..13.md      # Build logs (incl. failures + mitigations)
├── plans/IMPLEMENTATION_PLAN.md # The implementation plan
├── requirements.txt             # Base dependencies
├── requirements-macos.txt       # + PyTorch for Apple Silicon
├── requirements-linux-cuda.txt  # + PyTorch (CUDA) for Ubuntu + NVIDIA
└── requirements.lock            # Frozen versions (Ubuntu-verified)
```

Runtime artifacts (git-ignored): `data/parsed/` (Docling JSON cache),
`data/indexes/pairN/` (Chroma indexes), `chroma_db/` (UI/default store).

---

## 4. Prerequisites

Common to both platforms:

- **Python 3.12**
- **Ollama** (for the default local LLM) — [ollama.com](https://ollama.com)
- **Git**
- **Disk space**: ~6–8 GB (models + indexes)
- **(Optional) Hugging Face token** — speeds up first-time model downloads and avoids rate limits

First run downloads: embedding model (~1.2 GB), reranker (~2.3 GB, Pairing 3), and Docling layout
models. Subsequent runs use the local cache.

---

## 5. Setup on Apple Silicon (M1/M2/M3)

> Uses the **MPS** backend. The code enables `PYTORCH_ENABLE_MPS_FALLBACK=1` automatically so
> unsupported ops fall back to CPU instead of crashing.

### Step 1 — Install system tools

```bash
# Homebrew if you don't have it: https://brew.sh
brew install python@3.12 tesseract ollama
xcode-select --install   # provides clang, needed by some PyTorch kernels
```

`tesseract` is only needed if you ever enable OCR for scanned PDFs (OCR is **off** by default —
the ISO PDFs contain a text layer).

### Step 2 — Get the project and create a virtual environment

```bash
git clone <your-repo-url> Rags_2_Riches
cd Rags_2_Riches

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements-macos.txt
```

This installs the base dependencies **plus** the macOS `torch` wheel, which includes MPS support
(no extra index needed).

### Step 4 — Start Ollama and pull the model

```bash
ollama serve &            # or just launch the Ollama app
ollama pull llama3.2:3b-instruct-q4_K_M
ollama run llama3.2:3b-instruct-q4_K_M "reply with OK"   # sanity check
```

### Step 5 — Configure

```bash
cp .env.example .env
# Edit .env only if needed. The defaults work out of the box.
```

### Step 6 — Verify the setup

```bash
python -c "from models.device import describe_device; print(describe_device())"
# Expected on M1:  mps (Apple Silicon)

python -m pytest tests/ -q
```

### Step 7 — Continue to [Quickstart](#8-quickstart)

---

## 6. Setup on Ubuntu + NVIDIA

> Uses the **CUDA** backend. Verified on Ubuntu 22.04 (WSL) with an RTX 3050.
> The code sets `TORCH_DISABLE_NATIVE_JIT=1` automatically so PyTorch uses eager kernels and does
> **not** require a C compiler at runtime.

### Step 1 — Verify your NVIDIA driver

```bash
nvidia-smi
```

You should see your GPU and driver version. If not, install the NVIDIA driver first.

### Step 2 — Install system packages

```bash
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev git curl \
                        tesseract-ocr libgl1 libglib2.0-0
```

Notes:
- `python3.12`/`python3.12-venv` may require the **deadsnakes PPA** on older Ubuntu releases, or
  install Python 3.12 via `pyenv`. Check with `python3.12 --version`.
- `libgl1` and `libglib2.0-0` are needed by Docling's OCR/image stack.
- `tesseract-ocr` is only needed if you enable OCR (off by default).
- `build-essential` is **optional** (the code avoids runtime compilation); install it if you hit a
  compiler error: `sudo apt-get install -y build-essential`.

### Step 3 — Install Ollama (with GPU support)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b-instruct-q4_K_M
ollama run llama3.2:3b-instruct-q4_K_M "reply with OK"   # sanity check; should use the GPU
```

### Step 4 — Get the project and create a virtual environment

```bash
git clone <your-repo-url> Rags_2_Riches
cd Rags_2_Riches

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
```

### Step 5 — Install Python dependencies (CUDA)

```bash
pip install -r requirements-linux-cuda.txt
```

This installs the base dependencies and the CUDA-enabled `torch` wheel from the PyTorch index. If
your CUDA version differs, edit the `--extra-index-url` line in
[`requirements-linux-cuda.txt`](requirements-linux-cuda.txt) (e.g. `cu121`, `cu124`, `cu126`) to
match your driver, then re-run.

### Step 6 — Configure

```bash
cp .env.example .env
# Defaults are fine. See the configuration table below.
```

### Step 7 — Verify the setup

```bash
python -c "from models.device import describe_device; print(describe_device())"
# Expected:  cuda (NVIDIA GeForce RTX ...)

python -m pytest tests/ -q
```

### Step 8 — Continue to [Quickstart](#8-quickstart)

---

## 7. Configuration reference (.env)

Copy [`.env.example`](.env.example) to `.env` and adjust as needed.

| Variable | Default | What it does |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `ollama` (local) or `openai`. |
| `LLM_MODEL` | `llama3.2:3b-instruct-q4_K_M` | Model name (use e.g. `gpt-4o-mini` with OpenAI). |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL. |
| `OPENAI_API_KEY` | _(empty)_ | Required only when `LLM_PROVIDER=openai`. |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Local embedding model. |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder used by Pairing 3. |
| `DEVICE` | _(empty → auto)_ | Force `cuda`, `mps`, or `cpu`. |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Default vector store location (UI). |
| `TOP_K` | `5` | Results returned per retriever. |
| `HYBRID_VECTOR_WEIGHT` | `0.6` | Hybrid search weight for the vector retriever. |
| `HYBRID_BM25_WEIGHT` | `0.4` | Hybrid search weight for BM25. |
| `RERANK_TOP_N` | `20` | Max candidates reranked in Pairing 3. |
| `ISO_PDF_DIR` | `./Iso` | Folder containing the three PDFs. |
| `DOCLING_CACHE_DIR` | `./data/parsed` | Parsed-document JSON cache. |
| `DOCLING_BATCH_SIZE` | `1` | Lower = less RAM. Raise on high-memory machines. |
| `VECTORSTORE_BATCH_SIZE` | `1000` | Upsert batch size (Chroma caps ~5461). |
| `HF_TOKEN` | _(empty)_ | Optional; speeds up/authenticates model downloads. |

---

## 8. Quickstart

Assuming your virtual environment is active (`source .venv/bin/activate`) and Ollama is running.

### 8.1 Run the tests

```bash
python -m pytest tests/ -q
```

### 8.2 Build the evaluation gold set

```bash
python -m eval.build_gold_set
# -> writes eval/gold_set.json (32 queries mapped to corpus sections)
```

### 8.3 Launch the chat UI

```bash
streamlit run ui/app.py
# Open http://localhost:8501
```

### 8.4 Evaluate a pairing (produces a per-pairing report)

```bash
# 3-query smoke run (fast)
python -m eval.run_pairing --pairing 2 --limit 3

# Full run over all 32 queries
python -m eval.run_pairing --pairing 2
```

### 8.5 Generate the cross-pairing comparison report

```bash
python -m eval.comparison_report
# -> writes eval/RAG_EVAL_REPORT.md
```

> **Note:** `main.py` (a single end-to-end CLI demo) is planned for Milestone 14. Until then, use
> the commands above.

---

## 9. Using the chat UI

1. Start it with `streamlit run ui/app.py` and open `http://localhost:8501`.
2. In the sidebar:
   - **Pipeline** — choose Pairing 1, 2, or 3 (default: Pairing 3).
   - **Mode** — *Single pipeline* or *Compare all three*.
   - **Filter by section** — appears only for Pairing 2 (e.g. `3.1.1`).
   - **Device / LLM** — shows what will be used.
   - **Export chat** — downloads the conversation as Markdown.
3. Ask a question in the chat box. The answer appears with an expandable **Sources** panel showing
   document id, section number, content type, page, and cross-references.

**First question per pipeline** triggers index/model loading (can take a minute or two); later
questions are fast.

> **Memory tip:** *Compare all three* builds each pipeline sequentially to stay within ~8 GB RAM,
> so it is slower than single mode.

### 9.2 Evaluation & Comparison tab

The app has a second tab, **📊 Evaluation & Comparison**, that visualises the latest evaluation
results (no need to open the Markdown reports):

- **Summary table** across all four metric categories.
- **Retrieval quality** grouped bars (Recall@5 / nDCG@5 / MRR per pairing).
- **Generation quality** grouped bars (faithfulness / relevance / correctness / citation accuracy).
- **Quality vs latency** bubble chart (bubble size = index size).
- **Recall@5 by query category** heatmap.
- **Chunking characteristics** bars, **efficiency** table, and **flagged metrics** table.
- An expander with the full [`eval/RAG_EVAL_REPORT.md`](eval/RAG_EVAL_REPORT.md).

It reads `eval/result_pair{1,2,3}.json`, so generate those first (see §8.4); a **Refresh data**
button reloads them. If no results exist yet, the tab shows the exact commands to run.

---

## 10. Evaluation & reports

The evaluation follows [`RAG_EVAL_METRICS.md`](RAG_EVAL_METRICS.md):

- **Retrieval**: Recall@k, Precision@k, MRR, nDCG@k (k = 1, 3, 5, 10).
- **Chunking**: chunk count, average size, relevance density, boundary coherence.
- **Generation**: faithfulness, answer relevance, answer correctness, citation accuracy.
- **Efficiency**: indexing time, query latency, index size, tokens/LLM calls per query.

Relevance is computed two ways: **answer-term** matching (works for all pairings) and
**section-based** matching (Pairings 2/3). Uncomputable metrics are **flagged**, never omitted.

Outputs:

| File | Contents |
|---|---|
| [`eval/report_pair1.md`](eval/report_pair1.md) | Pairing 1 full report |
| [`eval/report_pair2.md`](eval/report_pair2.md) | Pairing 2 full report |
| [`eval/report_pair3.md`](eval/report_pair3.md) | Pairing 3 full report |
| [`eval/RAG_EVAL_REPORT.md`](eval/RAG_EVAL_REPORT.md) | **Cross-pairing comparison** (the deliverable) |

> **Tip:** For higher-quality judge scores, run the evaluation with a stronger judge LLM:
> ```bash
> LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini python -m eval.run_pairing --pairing 2
> ```

---

## 11. Commands cheat sheet

```bash
# Activate environment
source .venv/bin/activate            # macOS/Linux

# Tests
python -m pytest tests/ -q

# Device check
python -c "from models.device import describe_device; print(describe_device())"

# Gold set
python -m eval.build_gold_set

# Evaluate one pairing
python -m eval.run_pairing --pairing 1 --limit 3      # fast
python -m eval.run_pairing --pairing 1                # full

# Cross-pairing report
python -m eval.comparison_report

# UI
streamlit run ui/app.py

# Ollama
ollama list
ollama pull llama3.2:3b-instruct-q4_K_M

# Optional: switch to OpenAI
LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini python -m eval.run_pairing --pairing 2 --limit 3
```

---

## 12. Troubleshooting

Each fix below addresses a real failure encountered during development (details in
[`docs/`](docs)).

### 12.1 `RuntimeError: Failed to find C compiler` (or Triton errors)

**Cause**: PyTorch 2.14 routes some ops through a Triton JIT path that wants a C compiler.
**Already handled**: `config.py` and `ingestion/loader.py` set `TORCH_DISABLE_NATIVE_JIT=1` before
torch loads, forcing eager kernels. If it still appears:
- macOS: `xcode-select --install`
- Ubuntu: `sudo apt-get install -y build-essential`

### 12.2 Process killed / `Killed` while parsing a large PDF (out of memory)

**Cause**: Docling with OCR/big batches exceeded RAM on an 8 GB machine.
**Handled by defaults**: `use_ocr=False` and `DOCLING_BATCH_SIZE=1`. On a machine with more RAM,
raise `DOCLING_BATCH_SIZE` in `.env`. Only enable OCR for scanned PDFs.

### 12.3 `Batch size ... is greater than max batch size of 5461` (Chroma)

**Handled**: all indexing goes through `add_documents_in_batches()`
([`vectorstore/store.py`](vectorstore/store.py)). Lower `VECTORSTORE_BATCH_SIZE` if needed.

### 12.4 GPU out of memory during reranking (Pairing 3)

**Cause**: the embedding model and the cross-encoder reranker together exceed ~8 GB VRAM.
**Mitigations**:
- Reranking is `try/except`-guarded, so it degrades to unranked results instead of crashing.
- Put the reranker on CPU: set `DEVICE=cpu` (slower but stable), or
- Reduce `RERANK_TOP_N` in `.env`.

### 12.5 `SelfQueryRetriever unavailable (ImportError: DatabricksVectorSearch)`

**Cause**: a `langchain-community` version mismatch. Pairing 3 falls back to a plain vector
retriever automatically (logged as a warning). Automatic NL→filter translation is inactive; pin a
compatible `langchain-community` if you need it.

### 12.6 RAGAS metrics show `nan`

**Cause**: RAGAS needs a judge that emits strict JSON; the small local model does not.
**Mitigation**: use the built-in custom judge (default) or run RAGAS with
`LLM_PROVIDER=openai LLM_MODEL=gpt-4o-mini`.

### 12.7 `ollama_reachable: False` / connection refused

```bash
ollama serve &            # ensure the server is running
ollama list               # confirm the model is present
ollama pull llama3.2:3b-instruct-q4_K_M
```

### 12.8 First run is very slow

Models are being downloaded (~1.2 GB embeddings, ~2.3 GB reranker) and documents are being parsed
and cached. Subsequent runs use `data/parsed/` and `data/indexes/`. Set `HF_TOKEN` to speed up
downloads.

### 12.9 "Detected WSL. Using poll-based file watching"

Harmless Streamlit notice on WSL. Ignore it.

### 12.10 Wrong device selected

```bash
echo $DEVICE              # must be empty for auto-detect
python -c "from models.device import describe_device; print(describe_device())"
```
Force it explicitly with `DEVICE=cuda`, `DEVICE=mps`, or `DEVICE=cpu` in `.env`.

### 12.11 `ModuleNotFoundError: No module named 'config'` when running the UI

This was a real bug and is **fixed**: Streamlit puts the script's directory (`ui/`) on `sys.path`
instead of the repo root. [`ui/app.py`](ui/app.py) and
[`ui/pipeline_registry.py`](ui/pipeline_registry.py) now add the repository root to `sys.path`
automatically, so just run:

```bash
streamlit run ui/app.py      # from the project root
```

If you run the app from inside `ui/`, change back to the project root first (or set
`PYTHONPATH=.`). Regression-tested by [`tests/test_ui_app.py`](tests/test_ui_app.py).

---

## 13. Cross-platform notes

- **Device auto-detection**: `cuda` → `mps` → `cpu`, unless `DEVICE` is set.
- **Apple MPS**: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically.
- **Ubuntu CUDA**: install PyTorch via [`requirements-linux-cuda.txt`](requirements-linux-cuda.txt);
  adjust the CUDA index to match your driver.
- **OCR**: off by default (the ISO PDFs have a text layer). Enable per call if you have scanned
  documents.
- **Verified environment**: Ubuntu + NVIDIA RTX 3050 (driver 616.92, Python 3.12.13). The macOS/MPS
  path is code-complete but has not yet been run on M1 hardware.

---

## 14. Project documentation index

- **Plan**: [`plans/IMPLEMENTATION_PLAN.md`](plans/IMPLEMENTATION_PLAN.md)
- **Specs / guides**: [`RAG_SPECIFICATION.md`](RAG_SPECIFICATION.md),
  [`PIPELINE_IMPLEMENTATION_GUIDE.md`](PIPELINE_IMPLEMENTATION_GUIDE.md),
  [`EVALUATION_AND_UI.md`](EVALUATION_AND_UI.md),
  [`OPEN_SOURCE_MODELS.md`](OPEN_SOURCE_MODELS.md),
  [`RAG_EVAL_METRICS.md`](RAG_EVAL_METRICS.md)
- **Build logs (with failures & mitigations)**: [`docs/`](docs)
  - [`docs/MILESTONE_1.md`](docs/MILESTONE_1.md) … [`docs/MILESTONE_13.md`](docs/MILESTONE_13.md)

---

## 15. FAQ

**Do I need an API key?**
No. By default everything runs locally (Ollama). An OpenAI key is optional.

**Can I use a different LLM?**
Yes. Set `LLM_MODEL` (and `LLM_PROVIDER=ollama` or `openai`) in `.env` — no code changes.

**Where are models and indexes stored?**
Models in `~/.cache/huggingface`; parsed documents in `data/parsed/`; indexes in
`data/indexes/pairN/` and `chroma_db/`.

**How do I reset everything?**
Delete the caches: `rm -rf data/ chroma_db/ .venv` and re-run setup. (Keep `Iso/`.)

**Which pairing should I use?**
Pairing 3 is most accurate (tables + graph + reranking) but slowest; Pairing 2 is the best
balance; Pairing 1 is the fastest baseline. See [`eval/RAG_EVAL_REPORT.md`](eval/RAG_EVAL_REPORT.md).

---

_This project is for research/evaluation on ISO 15118 specification documents. Ensure you comply
with ISO document licensing when using the bundled PDFs._
