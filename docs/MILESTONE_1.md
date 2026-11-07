# Milestone 1 — Skeleton and Cross-Platform Bootstrap

**Status**: ✅ Complete
**Date**: 2026-09-23
**Verified on**: Ubuntu (WSL) + NVIDIA GeForce RTX 3050, driver 616.92, Python 3.12.13
**Plan reference**: [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) §7 (Milestone 1), §3 (Cross-Platform Strategy), §4 (Repository Layout)

---

## 1. Objective

Establish the project skeleton, cross-platform dependency strategy, shared configuration, and
runtime device detection so the RAG pipeline runs on both Apple Silicon M1 (MPS) and
Ubuntu + NVIDIA (CUDA) from one codebase.

---

## 2. Files Created / Modified

| Path | Action | Purpose |
|---|---|---|
| [`requirements.txt`](../requirements.txt) | Modified | Platform-agnostic base dependencies (LangChain, Docling, Chroma, BM25, NetworkX, RAGAS, Streamlit, utils) |
| [`requirements-macos.txt`](../requirements-macos.txt) | Created | macOS/Apple Silicon install file (default PyPI torch wheel includes MPS) |
| [`requirements-linux-cuda.txt`](../requirements-linux-cuda.txt) | Created | Ubuntu/NVIDIA install file (torch from CUDA index `cu124`) |
| [`requirements.lock`](../requirements.lock) | Created | Frozen resolved versions from the verified Ubuntu install (via `pip freeze`) |
| [`.env.example`](../.env.example) | Created | Environment template: provider switch, model selection, device override, retrieval knobs, corpus path |
| [`config.py`](../config.py) | Created | Shared configuration loaded from env vars with defaults |
| [`models/__init__.py`](../models/__init__.py) | Created | Package marker |
| [`models/device.py`](../models/device.py) | Created | Cross-platform device detection (`cuda > mps > cpu`) with env override |
| [`ingestion/__init__.py`](../ingestion/__init__.py) | Created | Package marker |
| [`graph/__init__.py`](../graph/__init__.py) | Created | Package marker |
| [`retrieval/__init__.py`](../retrieval/__init__.py) | Created | Package marker |
| [`chain/__init__.py`](../chain/__init__.py) | Created | Package marker |
| [`eval/__init__.py`](../eval/__init__.py) | Created | Package marker |
| [`ui/__init__.py`](../ui/__init__.py) | Created | Package marker |
| [`tests/__init__.py`](../tests/__init__.py) | Created | Package marker |
| [`.gitignore`](../.gitignore) | Modified | Ignore `.venv/`, `.env`, `__pycache__/`, `chroma_db/`, generated `eval/report.md` |
| [`plans/IMPLEMENTATION_PLAN.md`](../plans/IMPLEMENTATION_PLAN.md) | Modified | Added §13 "Verified Environment and Deviations" |

---

## 3. Repository Structure After Milestone 1

```
Rags_2_Riches/
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-macos.txt
├── requirements-linux-cuda.txt
├── requirements.lock
├── config.py
├── models/
│   ├── __init__.py
│   └── device.py
├── ingestion/__init__.py
├── graph/__init__.py
├── retrieval/__init__.py
├── chain/__init__.py
├── eval/__init__.py
├── ui/__init__.py
├── tests/__init__.py
├── docs/
│   └── MILESTONE_1.md          # this file
├── plans/
│   └── IMPLEMENTATION_PLAN.md
└── Iso/                        # ISO-15118-1/2/3.pdf (input)
```

---

## 4. Key Design Decisions

### 4.1 Cross-platform device detection
[`models/device.py`](../models/device.py) resolves the compute device at runtime:
- `DEVICE` env var override first (if set).
- Then `cuda` if `torch.cuda.is_available()`.
- Then `mps` if the Apple backend is available.
- Else `cpu`.

`PYTORCH_ENABLE_MPS_FALLBACK=1` is set before torch use so unsupported ops on Apple MPS fall
back to CPU instead of crashing.

### 4.2 Per-OS dependency files
PyTorch is deliberately excluded from the base `requirements.txt` and installed by the per-OS
file, because the correct wheel differs by accelerator. This prevents a macOS arm64 user from
accidentally installing a CUDA wheel and vice versa.

### 4.3 Provider-agnostic configuration
[`config.py`](../config.py) imports no heavy libraries, so it is safe to import from any module.
It exposes `LLM_PROVIDER` (ollama/openai), model names, retrieval weights, and the corpus path.

---

## 5. Commands Executed

```bash
# Environment inspection
python3 --version                                  # Python 3.12.13
python3 -m pip --version                            # -> "No module named pip"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
#   -> NVIDIA GeForce RTX 3050, 616.92

# Virtual environment (system python had no pip)
python3 -m venv .venv
.venv/bin/python -m pip --version                   # pip 25.0.1 -> upgraded to 26.2.1

# Dependency install (Ubuntu + NVIDIA)
.venv/bin/python -m pip install --upgrade pip wheel setuptools
.venv/bin/python -m pip install -r requirements-linux-cuda.txt

# Verification
.venv/bin/python -c "from models.device import get_device, describe_device; ..."
.venv/bin/python -m pip freeze > requirements.lock
```

---

## 6. Verification Results

### 6.1 Config load
```
config: LLM_PROVIDER=ollama LLM_MODEL=llama3 EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
config: ISO_PDF_DIR=./Iso CHROMA=./chroma_db TOP_K=5
```

### 6.2 Device detection (Ubuntu + NVIDIA)
```
device:   cuda
describe: cuda (NVIDIA GeForce RTX 3050)
torch:    2.14.0+cu130, cuda_available: True
gpu:      NVIDIA GeForce RTX 3050
```

The CUDA path is confirmed working on this machine. The MPS (Apple Silicon) path is
code-complete but **not yet verified** — it requires execution on an actual M1 machine.

### 6.3 Import surface probe
All planned imports were probed. Confirmed working:
`HuggingFaceEmbeddings`, `ChatOllama`, `ChatOpenAI`, `OpenAIEmbeddings`, `Chroma`,
`RecursiveCharacterTextSplitter`, `Document`, `ChatPromptTemplate`, `RunnablePassthrough`,
`StrOutputParser`, `BM25Retriever`, `ParentDocumentRetriever`, `EnsembleRetriever`,
`SelfQueryRetriever`, `InMemoryStore`, `AttributeInfo`.

Confirmed **missing** (relocated): `langchain.retrievers`, `langchain.storage`.

---

## 7. Deviations from the Plan

The dependency resolver pulled **newer major versions than the `>=` floors assumed**. This is
recorded authoritatively in plan §13.

### 7.1 Resolved versions (Ubuntu + NVIDIA)

| Package | Resolved |
|---|---|
| `langchain` | 1.4.2 |
| `langchain-core` | 1.6.4 |
| `langchain-classic` | 1.0.8 |
| `langchain-community` | 0.4.2 |
| `langchain-chroma` | 1.1.0 |
| `langchain-huggingface` | 1.2.2 |
| `langchain-ollama` | 1.1.0 |
| `langchain-openai` | 1.6.5 |
| `langchain-text-splitters` | 1.1.2 |
| `docling` | 2.130.0 |
| `docling-core` | 2.98.0 |
| `chromadb` | 1.5.9 |
| `sentence-transformers` | 6.1.0 |
| `transformers` | 5.17.0 |
| `ragas` | 0.3.1 |
| `streamlit` | 1.64.0 |
| `torch` | 2.14.0+cu130 |
| `networkx` | 3.7 |
| `numpy` | 2.5.3 |
| `pandas` | 3.0.6 |
| `pydantic` | 2.13.5 |
| `rank-bm25` | 0.2.2 |

### 7.2 Import-path changes (LangChain 1.x)

Legacy retrievers/chains moved out of `langchain` into `langchain_classic`. Later milestones
**must** use these imports:

| Class | Correct import |
|---|---|
| `ParentDocumentRetriever` | `from langchain_classic.retrievers import ParentDocumentRetriever` |
| `EnsembleRetriever` | `from langchain_classic.retrievers import EnsembleRetriever` |
| `SelfQueryRetriever` | `from langchain_classic.retrievers import SelfQueryRetriever` |
| `InMemoryStore` | `from langchain_classic.storage import InMemoryStore` |
| `AttributeInfo` | `from langchain_classic.chains.query_constructor.base import AttributeInfo` |
| `BM25Retriever` | `from langchain_community.retrievers import BM25Retriever` |

`langchain-classic>=1.0.0` was added explicitly to [`requirements.txt`](../requirements.txt)
because the code imports it directly.

---

## 8. Issues Encountered and Resolutions

| Issue | Resolution |
|---|---|
| System `python3` had no `pip` (`No module named pip`) | Created a virtualenv (`python3 -m venv .venv`), which bootstraps pip |
| `torch` not installed initially | Installed via [`requirements-linux-cuda.txt`](../requirements-linux-cuda.txt) |
| `langchain.retrievers` / `langchain.storage` no longer exist | Documented move to `langchain_classic`; add explicit dependency |
| Plan targeted `langchain>=0.3` but 1.x installed | Recorded deviation; code will target 1.x APIs |

---

## 9. Acceptance Check (Milestone 1)

- [x] Folder structure created per plan §4.
- [x] `requirements.txt`, `requirements-macos.txt`, `requirements-linux-cuda.txt` added.
- [x] `.env.example` added.
- [x] `config.py` implemented and loads.
- [x] `models/device.py` implemented; `get_device()` verified returning `cuda` on Ubuntu+NVIDIA.
- [ ] `get_device()` verified returning `mps` on Apple M1 — **pending M1 hardware**.
- [x] `requirements.lock` generated for reproducibility.

---

## 10. Next Step

**Milestone 2 — Model Factories**: implement
[`models/embeddings.py`](../models/embeddings.py) (`HuggingFaceEmbeddings` on the detected
device), [`models/llm.py`](../models/llm.py) (Ollama/OpenAI switch), and
[`models/reranker.py`](../models/reranker.py) (`BAAI/bge-reranker-v2-m3` cross-encoder).
Verify embeddings load on `cuda` and that `get_llm()` returns the configured provider, and that
Ollama is reachable when `LLM_PROVIDER=ollama`.
