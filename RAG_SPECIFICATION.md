# Implementation Plan: ISO Document RAG Pipeline with LangChain

> **Purpose**: This document is a complete implementation specification for building a RAG (Retrieval-Augmented Generation) pipeline for ISO engineering specification documents (e.g., ISO 15118). It is designed to be fed to a coding LLM (Claude Code, Cursor, Aider, etc.) for direct implementation.

---

## 1. Project Overview

### 1.1 Goal
Build a RAG system that ingests ISO specification PDFs and answers natural-language questions with high precision by exploiting the document's structural hierarchy (clauses, tables, cross-references).

### 1.2 Scope
Implement **three progressively sophisticated chunking + retrieval pipelines**, all orchestrated by **LangChain**:

| # | Chunking Strategy | Retrieval Strategy | Complexity |
|---|---|---|---|
| 1 | Recursive Character Splitting | Parent-Child Retrieval | Low |
| 2 | Section-Based (Header-Aware) Chunking | Metadata-Filtered Hybrid Search | Medium |
| 3 | Advanced ISO Chunking (Table + Semantic + Metadata + Cross-Refs) | Graph-Enhanced Hybrid Retrieval | High |

### 1.3 Target Document
ISO 15118 (or any ISO engineering specification) — structured, table-heavy, dense with cross-references.

### 1.4 Non-Goals
- Fine-tuning embedding models
- Building a UI (CLI/notebook usage is sufficient)
- Multi-user authentication or deployment infra
- Real-time streaming

---

## 2. Tech Stack

| Component | Choice | Notes |
|---|---|---|
| Orchestration | **LangChain** (`>=0.3.0`) | Core framework |
| PDF Parsing | **Docling** (`docling`, `langchain-docling`) | Layout-aware, OCR, table extraction |
| Vector Store | **Chroma** (`langchain-chroma`) | Local persistence, metadata filtering |
| Embeddings | **OpenAI** `text-embedding-3-small` (default) or `BAAI/bge-large-en-v1.5` (local) | Configurable |
| LLM | **OpenAI** `gpt-4o-mini` (default) or local Ollama model | Configurable |
| Keyword Search | **BM25** (`rank_bm25`, `langchain-community`) | Exact term matching |
| Graph | **NetworkX** (prototype) | Cross-reference graph |
| Reranking (optional) | `sentence-transformers` cross-encoder | For Pairing 3 Stage 4 |

### 2.1 Dependencies (`requirements.txt`)

```txt
# Core LangChain
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
langchain-chroma>=0.2.0
langchain-openai>=0.2.0
langchain-text-splitters>=0.3.0

# PDF parsing
docling>=2.0.0
langchain-docling>=0.1.0

# Retrieval
rank_bm25>=0.2.2
chromadb>=0.5.0

# Knowledge graph
networkx>=3.0

# Embeddings (local fallback)
sentence-transformers>=3.0
langchain-huggingface>=0.1.0

# Utilities
python-dotenv>=1.0
pydantic>=2.0
```

---

## 3. Architecture

### 3.1 High-Level Flow

```
PDF ──► Loader (Docling) ──► Clean ──► Chunking Strategy ──► Vector Store + Metadata
                                                              │
                                                              ▼
User Query ──► Retrieval Strategy ──► Rerank ──► LLM ──► Answer
```

### 3.2 Folder Structure

```
iso-rag-project/
├── .env                          # API keys, config values
├── requirements.txt
├── config.py                     # Shared configuration constants
├── main.py                       # Entry point / demo script
├── ingestion/
│   ├── __init__.py
│   ├── loader.py                 # PDF loading + header/footer stripping
│   ├── chunking_pair1.py         # Recursive + Parent-Child chunks
│   ├── chunking_pair2.py         # Section-based chunks + metadata
│   └── chunking_pair3.py         # Advanced ISO chunks (tables, cross-refs)
├── graph/
│   ├── __init__.py
│   └── builder.py                # Cross-reference knowledge graph
├── retrieval/
│   ├── __init__.py
│   ├── pair1_retriever.py        # Parent-Child retriever
│   ├── pair2_retriever.py        # Metadata-filtered hybrid retriever
│   └── pair3_retriever.py        # Graph-enhanced hybrid retriever
├── chain/
│   ├── __init__.py
│   └── rag_chain.py              # LCEL RAG chain
└── tests/
    └── smoke_test.py             # Basic end-to-end sanity check
```

---

## 4. Configuration

### 4.1 `.env` Template

```env
# LLM & Embeddings
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...

# Optional: Local model via Ollama
# LLM_MODEL=ollama/llama3
# OLLAMA_BASE_URL=http://localhost:11434

# Vector Store
CHROMA_PERSIST_DIR=./chroma_db

# Retrieval
TOP_K=5
HYBRID_VECTOR_WEIGHT=0.6
HYBRID_BM25_WEIGHT=0.4

# PDF Input
ISO_PDF_PATH=./data/iso_15118.pdf
```

### 4.2 `config.py`

```python
import os
from dotenv import load_dotenv

load_dotenv()

# LLM & Embedding
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chunking — Pairing 1
PAIR1_CHILD_CHUNK_SIZE = 256
PAIR1_CHILD_OVERLAP = 25
PAIR1_PARENT_CHUNK_SIZE = 1024
PAIR1_PARENT_OVERLAP = 100

# Chunking — Pairings 2 & 3
PAIR2_CHUNK_SIZE = 512
PAIR2_CHUNK_OVERLAP = 50
PAIR3_CHUNK_SIZE = 512
PAIR3_CHUNK_OVERLAP = 50

# Retrieval
TOP_K = int(os.getenv("TOP_K", 5))
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", 0.6))
HYBRID_BM25_WEIGHT = float(os.getenv("HYBRID_BM25_WEIGHT", 0.4))

# Vector Store
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Input
ISO_PDF_PATH = os.getenv("ISO_PDF_PATH", "./data/iso_15118.pdf")
```

---

## 5. Module Specifications

### 5.1 `ingestion/loader.py`

**Purpose**: Load an ISO PDF into LangChain `Document` objects using Docling. Strip headers/footers.

**Function signature**:
```python
def load_iso_document(pdf_path: str, use_ocr: bool = True) -> list[Document]
```

**Behavior**:
1. Use `DoclingLoader` from `langchain_docling`.
2. Enable OCR (`pipeline_options.do_ocr = True`) and table structure recovery (`do_table_structure = True`).
3. For each returned document, strip:
   - `Page X of Y` patterns
   - Standalone page numbers
   - ISO copyright lines (`© ISO YYYY ... All rights reserved`)
4. Attach metadata: `source_file`, `page_number`.

**Output**: `list[Document]` with `page_content` (Markdown-formatted, tables preserved as Markdown) and cleaned metadata.

---

### 5.2 `ingestion/chunking_pair1.py`

**Purpose**: Produce parent and child chunks for Parent-Child Retrieval.

**Function signature**:
```python
def create_parent_child_chunks(
    documents: list[Document],
    child_chunk_size: int = 256,
    child_overlap: int = 25,
    parent_chunk_size: int = 1024,
    parent_overlap: int = 100,
) -> tuple[list[Document], list[Document]]
```

**Behavior**:
1. Instantiate two `RecursiveCharacterTextSplitter` instances (parent = 1024, child = 256).
2. For each source document:
   - Split into parent chunks.
   - Assign each parent a unique `parent_id` (format: `{source_file}_parent_{i}`).
   - Split each parent into child chunks; propagate `parent_id` to children.
3. Attach metadata: `parent_id`, `chunk_level` (`"parent"` or `"child"`), `child_index`.

**Separators**: `["\n\n", "\n", ". ", " ", ""]`

**Return**: `(parent_chunks, child_chunks)`

---

### 5.3 `ingestion/chunking_pair2.py`

**Purpose**: Split documents at clause boundaries and attach rich metadata.

**Function signature**:
```python
def create_section_based_chunks(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Document]
```

**Behavior**:
1. Detect section headings via regex: `^(#{1,4}\s*)?(\d+(?:\.\d+)*)\s+(.+)$` (matches `4.1`, `## 4.1.2`, etc.).
2. Slice text between consecutive headings.
3. For each section:
   - If longer than `chunk_size * 1.5`, further split with `RecursiveCharacterTextSplitter`.
   - Otherwise keep as one chunk.
4. Build a `heading_path` (e.g., `"4.2 Message Flow -> 4.2.1 Supported Messages"`) by walking parent section numbers.
5. Attach metadata to each chunk:
   - `section_number` (e.g., `"4.1.2"`)
   - `section_title` (e.g., `"Supported Messages"`)
   - `heading_path`
   - `source_file`
   - `page_number`
   - `content_type = "text"`

**Helper**: `_build_heading_path(section_number, title, all_matches) -> str`

**Fallback**: If no headings detected in a document, use `RecursiveCharacterTextSplitter` and mark `section_number="unknown"`.

---

### 5.4 `ingestion/chunking_pair3.py`

**Purpose**: Advanced ISO chunking combining four techniques.

**Function signature**:
```python
def create_advanced_iso_chunks(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Document]
```

**Behavior**:
1. **Table extraction** (`_extract_tables_as_chunks`):
   - Detect Markdown tables via regex `((?:^\|.*\|$\n?)+)` (Docling output).
   - Extract each table as a standalone chunk, prefixed with its caption (`**Table N: ...**`).
   - Set `content_type = "table"` and `table_caption`.
   - Remove table text from the main body.
2. **Clause splitting** (`_split_at_clauses`):
   - Same logic as `chunking_pair2.py`.
   - Attach `section_number`, `section_title`, `heading_path`, `content_type = "text"`.
3. **Cross-reference extraction** (`_extract_cross_references`):
   - Regex patterns to capture:
     - `see (\d+(?:\.\d+)*)`
     - `refer to (\d+(?:\.\d+)*)`
     - `in (?:clause|section) (\d+(?:\.\d+)*)`
     - `as (?:defined|specified) in (\d+(?:\.\d+)*)`
     - `Annex ([A-Z])`
     - `Table (\d+)`
     - `Figure (\d+)`
   - Store results in `metadata["cross_references"]` (list of strings).
4. Attach all metadata listed in Pairing 2, plus `cross_references`.

**Helper**: `_find_section_for_table(text, table_pos) -> str` — walks headings to find nearest preceding section.

**Output**: Flat `list[Document]` containing both text and table chunks, each with rich metadata.

---

### 5.5 `graph/builder.py`

**Purpose**: Build a directed graph linking chunks by cross-references.

**Function signature**:
```python
def build_cross_reference_graph(chunks: list[Document]) -> networkx.DiGraph
```

**Behavior**:
1. **Pass 1**: Add each chunk as a node. Node ID format: `{section_number}_{md5(content)[:8]}`.
   - Store the original `Document` as a node attribute.
2. **Pass 2**: For each chunk's `cross_references`, find matching chunks:
   - Match if `target_section == ref` OR `target_section.startswith(ref + ".")`.
   - Add a directed edge `(source_id → target_id)` with `type="cross_reference"` and `ref` attributes.

**Helper**: `_chunk_id(chunk) -> str`

**Output**: `networkx.DiGraph`

---

### 5.6 `retrieval/pair1_retriever.py`

**Purpose**: Parent-Child retriever using LangChain's `ParentDocumentRetriever`.

**Function signature**:
```python
def build_parent_child_retriever(
    child_chunks: list[Document],
    parent_chunks: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = "pair1_parent_child",
) -> ParentDocumentRetriever
```

**Behavior**:
1. Instantiate `OpenAIEmbeddings` (or configured embeddings).
2. Create `Chroma` vector store.
3. Create `InMemoryStore` for parent docs.
4. Construct `ParentDocumentRetriever(vectorstore, docstore)`.
5. Add parent chunks via `retriever.add_documents(parent_chunks, ids=[p.metadata["parent_id"] ...])`.
6. Add child chunks to the vectorstore directly.
7. Return the retriever.

**Query behavior**: matches against child embeddings, returns parent documents.

---

### 5.7 `retrieval/pair2_retriever.py`

**Purpose**: Hybrid search (vector + BM25) with optional metadata filtering.

**Function signatures**:
```python
def build_metadata_filtered_hybrid_retriever(
    chunks: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = "pair2_section_hybrid",
    vector_weight: float = HYBRID_VECTOR_WEIGHT,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
) -> tuple[EnsembleRetriever, Chroma]

def filtered_retrieve(
    vectorstore: Chroma,
    query: str,
    section_filter: str | None = None,
    k: int = TOP_K,
) -> list[Document]
```

**Behavior (build)**:
1. Create `Chroma` vector store, add all chunks.
2. Create vector retriever via `vectorstore.as_retriever(search_kwargs={"k": TOP_K})`.
3. Create BM25 retriever: `BM25Retriever.from_documents(chunks)`, set `k = TOP_K`.
4. Combine with `EnsembleRetriever(retrievers=[vector, bm25], weights=[0.6, 0.4])`.
5. Return both the ensemble retriever and the vectorstore (for filtered queries).

**Behavior (filtered_retrieve)**:
- If `section_filter` provided, add `filter={"section_number": {"$eq": section_filter}}` to `similarity_search`.

---

### 5.8 `retrieval/pair3_retriever.py`

**Purpose**: Four-stage graph-enhanced hybrid retrieval.

**Function signatures**:
```python
def build_graph_enhanced_retriever(
    chunks: list[Document],
    graph: networkx.DiGraph,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = "pair3_graph_hybrid",
) -> tuple[EnsembleRetriever, Chroma, networkx.DiGraph]

def graph_expanded_retrieve(
    query: str,
    hybrid_retriever: EnsembleRetriever,
    graph: networkx.DiGraph,
    max_depth: int = 1,
    top_k: int = TOP_K,
) -> list[Document]
```

**Behavior (build)**:
1. Create Chroma vector store with all chunks.
2. Configure `SelfQueryRetriever` with:
   - LLM: `ChatOpenAI(model=LLM_MODEL, temperature=0)`
   - Metadata fields: `section_number`, `content_type`, `source_file`
   - `search_kwargs={"k": TOP_K}`
3. Create BM25 retriever over all chunks.
4. Combine into `EnsembleRetriever` with weights `[0.6, 0.4]`.

**Behavior (graph_expanded_retrieve)**:
1. Stage 1+2: `initial_results = hybrid_retriever.invoke(query)`.
2. Stage 3: BFS expand via graph:
   - For each initial result, look up its node ID.
   - Traverse successors and predecessors up to `max_depth`.
   - Collect neighbor chunks not yet seen.
3. Stage 4 (optional): Deduplicate and return top `top_k * 2` chunks.
4. Return combined list.

**Helper**: `_chunk_id(chunk)` — must match graph node ID format.

---

### 5.9 `chain/rag_chain.py`

**Purpose**: Assemble the LCEL RAG chain.

**Function signature**:
```python
def build_rag_chain(retriever, llm_model: str = LLM_MODEL):
```

**Behavior**:
1. Instantiate `ChatOpenAI(model=llm_model, temperature=0)`.
2. Prompt template (system + human):

```
System: You are an expert assistant for ISO 15118 engineering specifications.
Answer the user's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information."
When referencing information, cite the section number (e.g., "According to Section 4.2...").
If the context includes tables, reproduce the relevant table values accurately.

Human:
Context:
{context}

Question: {question}
```

3. `format_docs` helper: joins retrieved docs with `[Section {section_number}]` prefix.
4. Build LCEL chain: `{"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()`.
5. Return the chain.

---

### 5.10 `main.py`

**Purpose**: Demo entry point demonstrating all three pairings.

**Behavior**:
```python
# Load
docs = load_iso_document(ISO_PDF_PATH, use_ocr=True)

# --- Pairing 1 ---
parent_chunks, child_chunks = create_parent_child_chunks(docs)
retriever1 = build_parent_child_retriever(child_chunks, parent_chunks)
chain1 = build_rag_chain(retriever1)
print("Pairing 1:", chain1.invoke("What are the AC charging voltage requirements?"))

# --- Pairing 2 ---
chunks2 = create_section_based_chunks(docs)
retriever2, vectorstore2 = build_metadata_filtered_hybrid_retriever(chunks2)
chain2 = build_rag_chain(retriever2)
print("Pairing 2:", chain2.invoke("What does Section 4.1 specify about communication setup?"))

# --- Pairing 3 ---
chunks3 = create_advanced_iso_chunks(docs)
graph = build_cross_reference_graph(chunks3)
retriever3, vectorstore3, graph = build_graph_enhanced_retriever(chunks3, graph)
chain3 = build_rag_chain(
    lambda q: graph_expanded_retrieve(q, retriever3, graph)
)
print("Pairing 3:", chain3.invoke("What voltage requirements are referenced in Section 4.2?"))
```

---

## 6. Implementation Order (Milestones)

Follow this exact order. Do not skip ahead.

### ✅ Milestone 1: Project Skeleton
- Create folder structure.
- Add `requirements.txt`, `.env`, `config.py`.
- Install dependencies.

### ✅ Milestone 2: Ingestion
- Implement `ingestion/loader.py`.
- Test: load the ISO PDF and print the first 3 documents + metadata.
- Verify: tables appear as Markdown in `page_content`.

### ✅ Milestone 3: Pairing 1 (Baseline)
- Implement `ingestion/chunking_pair1.py` and `retrieval/pair1_retriever.py`.
- Implement `chain/rag_chain.py` (needed for all pairings).
- Test: `main.py` only running Pairing 1. Verify answers reference parent context.

### ✅ Milestone 4: Pairing 2 (Structured)
- Implement `ingestion/chunking_pair2.py` and `retrieval/pair2_retriever.py`.
- Test: verify that `section_number` and `heading_path` are correctly extracted.
- Test: run a filtered query on a specific section number.

### ✅ Milestone 5: Pairing 3 (Advanced)
- Implement `ingestion/chunking_pair3.py`.
- Implement `graph/builder.py`.
- Implement `retrieval/pair3_retriever.py`.
- Test: verify table chunks are standalone, cross-references populate metadata, and graph traversal works.

### ✅ Milestone 6: Smoke Test
- Implement `tests/smoke_test.py` running all three chains on a fixed set of queries.
- Success = all three return non-empty answers with section citations.

---

## 7. Testing Strategy

### 7.1 Unit-level
- **Loader**: PDF → Document count > 0, tables preserved.
- **Pairing 1**: parent chunks > child chunks; every child has a `parent_id` that exists in parent set.
- **Pairing 2**: `section_number` matches expected regex on all text chunks; `heading_path` non-empty.
- **Pairing 3**: at least one `content_type="table"` chunk; `cross_references` populated for clauses that contain "see X" text.
- **Graph**: node count == chunk count; edge count > 0 for documents with cross-references.

### 7.2 Retrieval-level
For each pairing, run these example queries and inspect retrieved metadata:
1. `"What is the maximum charging voltage?"` → should return a table chunk or Section with parameter.
2. `"Describe the SLAC communication setup."` → should match Section on SLAC.
3. `"What does Section 4.1 say about X?"` → should include Section 4.1 chunks.

### 7.3 End-to-end
Run `main.py` end-to-end without errors. Each chain must return a string with a section citation.

---

## 8. Acceptance Criteria

A successful implementation satisfies:

- [ ] All three pipelines run end-to-end without errors.
- [ ] Pairing 3 chunks contain both `content_type="table"` and `content_type="text"` entries.
- [ ] Cross-reference graph contains at least one edge if source doc has "see X" phrases.
- [ ] Every retrieved chunk in Pairing 2 and 3 has `section_number` populated.
- [ ] `main.py` prints three answers with citations.
- [ ] All code is type-hinted and imports cleanly.
- [ ] No hardcoded secrets; all config via `.env`.

---

## 9. Known Pitfalls & Mitigations

| Pitfall | Mitigation |
|---|---|
| Docling produces empty table Markdown for scanned PDFs | Ensure `do_ocr=True`; verify Tesseract is installed |
| Section regex misses non-numeric headings (e.g., "Annex A") | Add alternative regex for `Annex [A-Z]` |
| Cross-reference matches wrong target (e.g., `4.1` matching `4.10`) | Use `target_section == ref or target_section.startswith(ref + ".")` |
| Chroma metadata filter fails on `$eq` | Verify Chroma version; use `where={"section_number": section_filter}` if needed |
| BM25 retriever requires in-memory docs | BM25 loads all chunks into memory; fine for ISO-size docs, bad for >10k chunks |
| Parent-Child retriever returns duplicates | Deduplicate by `parent_id` after retrieval |
| Graph expansion adds too many chunks | Limit `max_depth=1` and cap final results at `top_k * 2` |

---

## 10. Optional Enhancements (Post-MVP)

- **Cross-encoder reranking** for Pairing 3 Stage 4: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Parent-Child + Graph hybrid**: combine Pairing 1 and 3 ideas for even richer context.
- **Multi-document support**: ingest multiple ISO standards (15118-1, -2, -3) with a `document_id` metadata field.
- **Streaming**: LCEL supports `.stream()` — swap `invoke` for `stream` in the chain.
- **Evaluation**: use `ragas` to measure faithfulness, answer relevance, context precision.

---

## 11. Commands Cheat Sheet

```bash
# Setup
python -m venv .venv
source .venv/bin/activate          # Linux/Mac
.venv\Scripts\activate             # Windows
pip install -r requirements.txt

# Run
python main.py

# Tests
python tests/smoke_test.py
```

---

**END OF IMPLEMENTATION PLAN**

> **Instructions to coding LLM**: Implement the modules in the order listed in Section 6. Follow the function signatures in Section 5 exactly. After each milestone, run the corresponding test described in Section 7 before proceeding. If a library API differs from the spec, prefer the version in the currently installed package and note the deviation. Do not skip type hints.