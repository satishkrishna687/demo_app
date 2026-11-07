## Chapter 3: System Requirements Specification

### 3.1 System Overview

The system is a fully local retrieval-augmented question-answering system over the three ISO 15118 part documents (Parts 1, 2 and 3) [12]. It is organised as an ingestion stage that turns PDFs into stored chunks, a retrieval stage that turns a question into a ranked set of clauses, and a generation stage that turns those clauses into a cited answer. The ingestion stage converts each PDF with Docling [10] in layout-aware mode with OCR disabled, caches the conversion as JSON, removes running headers and footers, and keeps each PDF as a single document whose page boundaries are derived from a form-feed placeholder. A structure-recovery pass converts Docling's `section_header` and `TableItem` labels into an outline of sections and blocks with normalised clause numbers, heading paths and page numbers. The chosen chunking strategy then turns that outline into LangChain documents carrying metadata such as `document_id`, `section_number`, `section_title` and `page_start`.

Retrieval differs per pairing but always returns ranked documents with metadata intact. Pairing 1 retrieves small child chunks and expands each hit to its larger parent. Pairing 2 combines dense Chroma search with BM25 through reciprocal rank fusion and can be restricted to a clause by a metadata filter. Pairing 3 adds a NetworkX cross-reference graph over extracted references such as "see 4.2" or "refer to Annex A", expands candidates breadth-first, ensembles the retrievers, and reranks the top candidates with a cross-encoder, degrading gracefully to the unranked list if the reranker runs out of memory. Generation is a LangChain LCEL chain that condenses a follow-up question against the conversation history, restricts the system prompt to the retrieved context, requires document and clause citations, and provides an explicit refusal phrase when the context is insufficient.

Evaluation and presentation close the loop. The evaluation harness replays a gold question set against a selected pairing, computes deterministic retrieval metrics and LLM-as-judge quality metrics, and writes a JSON result that the reporting layer turns into a per-pairing report and a comparison table. The Streamlit user interface exposes three views: an ask/chat view with a citation panel, an evaluation and comparison view, and a chunk browser for inspecting what was actually indexed.

The requirements in this chapter are stated so that each is either measurable or explicitly qualitative. Where a numerical threshold is given, it is labelled as a proposed acceptance criterion or a target to be verified, not as a measured result. The measured results themselves are presented in Chapter 6.

### 3.2 User Requirements

The system is intended for three classes of user, whose needs are stated below.

**UR-01 — The standards implementer.** An engineer implementing ISO 15118 needs to locate the clause that governs a specific message, field, parameter or procedure, and to see the exact source text so that the answer can be checked. The user requires clause-citable answers and a source-passage panel.

**UR-02 — The student or researcher.** A user learning the standard needs definitional and procedural explanations that are grounded in the document rather than paraphrased from memory. The user requires answers that decline when the retrieved evidence is insufficient, rather than guessing.

**UR-03 — The evaluator or reviewer.** A reviewer needs to compare the three pipeline configurations on the same questions and to inspect the retrieved chunks and the measured metrics. The user requires a reproducible evaluation view and a chunk browser.

**UR-04 — The privacy-conscious operator.** A user handling licensed or sensitive material needs the corpus, the index and the retrieval stack to remain on the local machine, with only the optional chat completion permitted to use a network service.

### 3.3 Functional Requirements

The functional requirements are listed in Table 3.1. They are derived from the implemented system and are grouped by stage: ingestion (FR-01 to FR-11), retrieval (FR-12 to FR-16), generation (FR-17 to FR-19), evaluation (FR-20 to FR-23), and operation and presentation (FR-24 to FR-26). Each requirement carries a priority of Must or Should.

**Table 3.1: Functional requirements (FR-01 … FR-26)**

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | Parse the three ISO 15118 PDFs with Docling using layout analysis, table-structure recovery, `use_ocr=False` and `DOCLING_BATCH_SIZE=1`, and persist the parsed result as JSON under `data/parsed/`. | Must |
| FR-02 | Reuse the cached parse when it is present and current, so that no UI or evaluation action re-parses the corpus. | Must |
| FR-03 | Strip repeated running headers and footers from the parsed text. | Must |
| FR-04 | Recover a section/block outline from Docling labels, normalising numeric clause numbers and annex labels, merging a clause number emitted separately from its title, and filtering furniture and front-matter headings. | Must |
| FR-05 | Attach `document_id`, `section_number`, `section_title`, `heading_path` and page numbers to every chunk. | Must |
| FR-06 | Provide three chunking strategies over the same structured corpus: recursive parent/child (child 256, overlap 25; parent 1024, overlap 100), section-based (512, overlap 50), and advanced ISO chunking (512, overlap 50). | Must |
| FR-07 | Keep Markdown tables atomic; split only above `TABLE_MAX_CHARS=4000`, on row boundaries, with the header row repeated. | Must |
| FR-08 | Emit standalone table chunks with their captions in the advanced pairing, and extract cross-references such as "see 4.2" or "refer to Annex A". | Must |
| FR-09 | Index chunks in Chroma with normalised local embeddings, applying the model's instruction prompt to queries only and flattening tables for encoding without altering stored text. | Must |
| FR-10 | Perform all vector-store writes through a single batched helper with `VECTORSTORE_BATCH_SIZE=1000` and sanitised metadata. | Must |
| FR-11 | Rebuild the index automatically when the chunking configuration changes. | Must |
| FR-12 | Provide hybrid retrieval combining dense similarity with BM25, fused by reciprocal rank fusion using vector weight 0.6 and BM25 weight 0.4. | Must |
| FR-13 | Expand retrieved clauses along the cross-reference graph by a bounded breadth-first traversal. | Must |
| FR-14 | Rerank the top `RERANK_TOP_N=20` candidates with a cross-encoder and return the unranked list unchanged if reranking raises an error, for example on GPU out-of-memory. | Must |
| FR-15 | Support retrieval restricted by `section_number` metadata filter. | Should |
| FR-16 | Return `TOP_K=5` clauses per query, with metadata intact. | Must |
| FR-17 | Generate answers only from the retrieved context, require a document and clause citation, and instruct the model to state that it lacks sufficient information when the context does not support an answer. | Must |
| FR-18 | Handle conversational follow-ups by condensing them into standalone questions against the history before retrieval. | Must |
| FR-19 | Show the retrieved passages used for each answer in the interface. | Must |
| FR-20 | Evaluate a fixed gold question set with deterministic retrieval metrics at k = 1, 3, 5, 10 and LLM-as-judge quality metrics at temperature 0, persisting results as JSON and rendering per-pairing and comparison reports. | Must |
| FR-21 | Determine citation accuracy deterministically by comparing the clause numbers cited in an answer with the sections actually retrieved. | Should |
| FR-22 | Record uncomputable judge scores as `None` and expose them as flags rather than silently dropping them. | Must |
| FR-23 | Provide a chunk-inspection diagnostic reporting counts, section coverage, oversized sections, and table and graph statistics. | Should |
| FR-24 | Operate fully offline for parsing, embedding, reranking, storage and evaluation; allow the chat model to be pointed at a cloud provider only by explicit configuration. | Must |
| FR-25 | Provide a UI with an ask/chat view with citations, an evaluation and comparison view, and a chunk browser, selectable from the three registered pairings. | Must |
| FR-26 | Construct pipelines one at a time and release each before building the next in compare mode. | Must |

### 3.4 Non-Functional Requirements

The non-functional requirements are listed in Table 3.2. They cover performance, reliability, privacy, usability, maintainability, testability, reproducibility and portability. Where a requirement is qualitative, it is stated as a goal rather than as a measurable threshold; where a threshold is given, it is a proposed acceptance criterion to be verified.

**Table 3.2: Non-functional requirements (NFR-01 … NFR-08)**

| ID | Area | Requirement |
|---|---|---|
| NFR-01 | Performance | Interactive query latency should remain within a few seconds on CPU-class hardware after warm-up, with the reranking step as the dominant cost. A full re-index of the three documents should complete without exceeding the available memory; the pipeline is designed for an 8 GB machine. |
| NFR-02 | Reliability | Parsed documents are cached, so a failed run does not repeat conversion cost. Reranking degrades to unranked results rather than failing the query. Index freshness is checked so that a stale index cannot silently answer from outdated chunks. |
| NFR-03 | Privacy | Corpus text, embeddings, index and reranking all run locally. Only the chat completion may leave the machine, and only when `LLM_PROVIDER` is set to a hosted provider. |
| NFR-04 | Usability | The interface must present the answer together with the exact retrieved passages, so that any citation can be checked against the source panel, and must allow the user to switch between the three pairings and compare their outputs on the same question. |
| NFR-05 | Maintainability | Adding a pairing requires one chunking module, one retriever module and one branch in the pipeline registry. All modules use `from __future__ import annotations`, and configuration is centralised in [`config.py`](config.py:1). |
| NFR-06 | Testability | The test suite runs without loading models or parsing PDFs, using fabricated Docling items and stub embeddings, so it is fast and deterministic on any machine. |
| NFR-07 | Reproducibility | The corpus is fixed, the judge runs at temperature 0 with a single retry, device selection is explicit, and evaluation runs are persisted as JSON so that results can be regenerated and diffed. |
| NFR-08 | Portability | Device resolution supports CUDA, Apple MPS and CPU; PyTorch is installed from the per-OS requirement files. |

### 3.5 Hardware Requirements

The system is designed to run on a single workstation without a dedicated server. The development and evaluation environment is an 8 GB machine, and the design decisions that follow from that ceiling — Docling batch size 1, single-pipeline caching, sequential construction in compare mode, and the guarded reranking step — are recorded in Chapter 4. The hardware requirements are stated in Table 3.3. A GPU is optional: device resolution is `cuda > mps > cpu`, so the system runs on CPU alone, with a CUDA or Apple MPS device used when available.

### 3.6 Software Requirements

The software stack is entirely open and locally installable. The principal components are Python 3.12, LangChain and LangChain Classic for the document, retriever and LCEL primitives, Docling for layout-aware PDF conversion, Chroma for the local vector store, `rank_bm25` for sparse retrieval, NetworkX for the cross-reference graph, `sentence-transformers` and `transformers` for local embedding and cross-encoder inference, Ollama for the local chat model, Streamlit for the interface, and pytest for the test suite. PyTorch is installed separately per operating system because the correct wheel differs between CUDA and Apple Silicon. The hardware and software requirements are combined in Table 3.3.

**Table 3.3: Hardware and software requirements**

| Category | Requirement | Notes |
|---|---|---|
| Processor | x86-64 or Apple Silicon CPU | CPU-only operation is supported |
| Memory | 8 GB RAM minimum | Design target; Docling batch size 1 and single-pipeline caching are consequences |
| GPU | Optional CUDA or Apple MPS device | Device resolution is `cuda > mps > cpu`; overridable via `DEVICE` |
| Storage | Sufficient disk for model checkpoints and the Chroma index | Index size is reported per pairing in Chapter 6 |
| Operating system | Linux or macOS | PyTorch installed from `requirements-linux-cuda.txt` or `requirements-macos.txt` |
| Language runtime | Python 3.12 | Every module uses `from __future__ import annotations` |
| Orchestration | `langchain`, `langchain-core`, `langchain-community` (≥0.3); `langchain-classic` (≥1.0) | Document, retriever and LCEL primitives; classic line supplies parent-document, ensemble and self-query retrievers |
| PDF parsing | `docling` (≥2.0) with `langchain-docling` | Layout analysis and table-structure recovery; OCR disabled |
| Vector store | `chromadb` (≥0.5) via `langchain-chroma` (≥0.2) | Local, file-backed; supports metadata filtering |
| Sparse retrieval | `rank_bm25` (≥0.2.2) | In-process lexical retrieval |
| Graph | `networkx` (≥3.0) | Directed cross-reference graph and bounded breadth-first expansion |
| Embeddings / reranking | `sentence-transformers` (≥3.0), `transformers` (≥4.40) | Local inference; `Qwen/Qwen3-Embedding-0.6B` and `BAAI/bge-reranker-v2-m3` |
| Chat model | Ollama (default) or OpenAI (optional) | Default `llama3.2:3b-instruct-q4_K_M`; provider selected by `LLM_PROVIDER` |
| Interface | Streamlit (≥1.38) | Ask/chat, evaluation and chunk-browser views |
| Testing | pytest (≥8.0) | Fast, model-free suite under `tests/` |

### 3.7 System Constraints

The system operates under the following constraints, each of which is a direct consequence of the target environment or the corpus.

**C-01 — Fixed corpus.** The corpus is the three ISO 15118 PDFs held in `Iso/`. The parsing and chunking strategies are designed for their structure, and the system does not attempt general web-scale retrieval or arbitrary-PDF processing.

**C-02 — No OCR by default.** The source PDFs are digitally native and contain a text layer, so OCR is disabled. Scanned-document handling is out of scope.

**C-03 — Memory ceiling.** The development machine has 8 GB of memory. Docling batch size 1, single-pipeline caching, sequential construction in compare mode, and the guarded reranking step are all direct consequences of this limit.

**C-04 — Import order.** [`config.py`](config.py:14) sets the PyTorch environment variables that must exist before torch is first imported, because torch caches that state. Configuration is therefore a leaf module with no heavy imports.

**C-05 — Working directory.** Streamlit places `ui/` on the path rather than the repository root, so the UI modules insert the root into `sys.path` themselves and the application must be launched from the repository root.

**C-06 — Licensed material.** The corpus is licensed, so the system must not reproduce extensive ISO text; answers cite clauses and the source panel shows the retrieved passages to the user.

### 3.8 Assumptions and Dependencies

The system assumes that the three PDFs are present in `Iso/` and are the digitally native editions with a text layer; that the local embedding, reranker and chat models are available (downloaded on first use and cached thereafter); that the machine has sufficient memory for a single pipeline at a time; and that the user launches the interface from the repository root. The system depends on the open-source components listed in Table 3.3, on the Docling structured labels for structure recovery, and on the standard's own cross-references for the graph edges. It does not depend on any hosted service for parsing, embedding, reranking, storage or evaluation; only the optional chat completion may use a network service, and only by explicit configuration.

### 3.9 Test Case Specification

The test suite under `tests/` is deliberately fast and deterministic: it uses fabricated Docling items and stub embeddings rather than real models or PDFs, so a full run completes in seconds on any machine. Table 3.4 maps each test case to the requirement it exercises, its scenario, its input, its expected outcome, and its status. Test cases that correspond to an existing pytest function are marked **Executed (pass)**; test cases that describe behaviour verified only by manual or end-to-end runs, or that are planned, are marked **Not yet executed**.

**Table 3.4: Test case specification**

| Test ID | Requirement | Test scenario | Input | Expected outcome | Status |
|---|---|---|---|---|---|
| TC-01 | FR-04 | Parse numeric and annex headings | `"4.1 Scope"`, `"Annex A"`, `"A.1 Relation of modes"`, `"Key"` | Numeric and annex headings parsed; furniture headings return `None` | Executed (pass) — [`test_structure.py`](tests/test_structure.py:47) |
| TC-02 | FR-03, FR-04 | Filter furniture and front-matter headings | `"© ISO 2019"`, `"iTeh STANDARD PREVIEW"`, `"4.1 Scope"` | Furniture detected; real headings not | Executed (pass) — [`test_structure.py`](tests/test_structure.py:57) |
| TC-03 | FR-04 | Merge a bare clause number with its title and build heading paths | Fake Docling items for `3`, `3.1`, `Annex A`, `A.1` | Sections `["3","3.1","A","A.1"]`; titles and heading paths correct | Executed (pass) — [`test_structure.py`](tests/test_structure.py:68) |
| TC-04 | FR-04 | Skip furniture and table-of-contents items | Fake items with `document_index` and `Contents` | Only the real section `1` is retained | Executed (pass) — [`test_structure.py`](tests/test_structure.py:93) |
| TC-05 | FR-05 | Attach table block and page provenance | Fake table item with caption and page 11 | Block kind `table`, page 11, caption preserved | Executed (pass) — [`test_structure.py`](tests/test_structure.py:108) |
| TC-06 | FR-07 | Repeat the header when splitting an oversized table | 200-row Markdown table, limit 500 | Multiple parts, each containing the header and separator | Executed (pass) — [`test_structure.py`](tests/test_structure.py:125) |
| TC-07 | FR-07 | Leave small or non-table text unchanged | Small table and plain text | Returned unchanged as a single part | Executed (pass) — [`test_structure.py`](tests/test_structure.py:136) |
| TC-08 | FR-07 | Keep tables atomic during packing | Units with a table in the middle | Table remains a single unit | Executed (pass) — [`test_structure.py`](tests/test_structure.py:142) |
| TC-09 | FR-06 | Split oversized text units | 100-char unit, budget 30 | Four chunks, each within budget | Executed (pass) — [`test_structure.py`](tests/test_structure.py:154) |
| TC-10 | FR-05, FR-06 | Section chunks carry section metadata and heading | One section `4.1 Scope` | Metadata `section_number`, `heading_path`, `page_number`; content prefixed with heading | Executed (pass) — [`test_structure.py`](tests/test_structure.py:174) |
| TC-11 | FR-07, FR-08 | Advanced pairing emits standalone, bounded table chunks | Section with a 300-row table | Table chunks present, header present, size bounded, section metadata intact | Executed (pass) — [`test_structure.py`](tests/test_structure.py:197) |
| TC-12 | FR-20 | Answer-term relevance matching | Docs with and without the term | Correct binary relevance flags | Executed (pass) — [`test_metrics.py`](tests/test_metrics.py:16) |
| TC-13 | FR-20 | Gold answer included in relevance terms | Entry with empty terms and a gold answer | Gold answer matched | Executed (pass) — [`test_metrics.py`](tests/test_metrics.py:22) |
| TC-14 | FR-20 | Section-based relevance signal | Docs with matching and non-matching sections | Correct section flags | Executed (pass) — [`test_metrics.py`](tests/test_metrics.py:28) |
| TC-15 | FR-20 | Recall, precision, MRR and nDCG computation | Known flag vectors | Correct metric values | Executed (pass) — [`test_metrics.py`](tests/test_metrics.py:34) |
| TC-16 | FR-20 | Aggregate retrieval metrics across queries | Two per-query flag lists | Correct aggregate recall, precision and MRR | Executed (pass) — [`test_metrics.py`](tests/test_metrics.py:59) |
| TC-17 | FR-20 | Chunk statistics and `None` handling | Chunk lists, including empty | Correct counts and sizes; empty returns `None` | Executed (pass) — [`test_metrics.py`](tests/test_metrics.py:70) |
| TC-18 | FR-09 | Flatten Markdown tables for encoding, preserve caption | Markdown table with caption | Pipes and separators removed; caption preserved | Executed (pass) — [`test_embeddings.py`](tests/test_embeddings.py:26) |
| TC-19 | FR-09 | Strip tables for documents only, not queries | Wrapper over a fake embedder | Documents stripped; queries passed through untouched | Executed (pass) — [`test_embeddings.py`](tests/test_embeddings.py:42) |
| TC-20 | FR-09 | Prompt selection auto-detects Qwen3 and can be forced | Model names and `EMBEDDING_USE_PROMPTS` values | Prompts enabled for Qwen3, disabled otherwise; forced values honoured | Executed (pass) — [`test_embeddings.py`](tests/test_embeddings.py:61) |
| TC-21 | FR-23, FR-25 | Chunk-explorer data shaping and charts | Synthetic chunk records | Correct columns, coverage percentages, content-type and per-document frames; charts build | Executed (pass) — [`test_chunk_explorer.py`](tests/test_chunk_explorer.py:10) |
| TC-22 | FR-20, FR-25 | Evaluation dashboard data shaping and charts | Persisted result JSON (if present) | Summary, tidy and category frames; charts build | Executed (pass) — [`test_eval_dashboard.py`](tests/test_eval_dashboard.py:10) |
| TC-23 | FR-25 | UI executes headlessly and renders the pipeline selector | `ui/app.py` via Streamlit `AppTest` | No exception; selectbox renders three options | Executed (pass) — [`test_ui_app.py`](tests/test_ui_app.py:15) |
| TC-24 | FR-24, NFR-08 | Configuration loads and device detection is valid | `config` and `get_device()` | Provider valid, model set, `TOP_K > 0`, device in `{cuda, mps, cpu}` | Executed (pass) — [`smoke_test.py`](tests/smoke_test.py:15) |
| TC-25 | FR-20 | Gold seeds present and correctly shaped | `eval/gold_seeds.json` | At least 30 seeds with required keys | Executed (pass) — [`smoke_test.py`](tests/smoke_test.py:29) |
| TC-26 | FR-01, FR-02 | Parse the three PDFs and reuse the JSON cache | The three PDFs in `Iso/` | Parsed JSON written to `data/parsed/`; second run reuses it | Not yet executed (manual/end-to-end) |
| TC-27 | FR-10, FR-11 | Batched vector-store writes and index rebuild on change | Chunks and a changed chunking fingerprint | Writes batched and sanitised; index rebuilt when the fingerprint changes | Not yet executed (manual/end-to-end) |
| TC-28 | FR-12, FR-13, FR-14 | Hybrid retrieval, graph expansion and reranking | A gold query against Pairing 3 | Fused, expanded and reranked candidate list; unranked fallback on reranker failure | Not yet executed (manual/end-to-end) |
| TC-29 | FR-17, FR-18 | Citation-grounded generation and follow-up condensation | A question and a follow-up with history | Answer cites a retrieved clause; follow-up condensed to a standalone question | Not yet executed (manual/end-to-end) |
| TC-30 | FR-21, FR-22 | Deterministic citation accuracy and flagged `None` scores | An answer with citations and an uncomputable judge score | Citation accuracy computed; uncomputable score recorded as `None` and flagged | Not yet executed (manual/end-to-end) |

### 3.10 Requirements Traceability Matrix

Table 3.5 maps each functional requirement to the module that implements it and to the test case that exercises it. The matrix makes explicit which requirements are covered by the fast, model-free suite and which are verified only by manual or end-to-end runs. Requirements with no automated test are marked accordingly, so that the coverage boundary is visible rather than assumed.

**Table 3.5: Requirements traceability matrix**

| Requirement | Implementing module(s) | Test case(s) | Automated coverage |
|---|---|---|---|
| FR-01 | [`ingestion/loader.py`](ingestion/loader.py:1) | TC-26 | No (end-to-end) |
| FR-02 | [`ingestion/loader.py`](ingestion/loader.py:1) | TC-26 | No (end-to-end) |
| FR-03 | [`ingestion/loader.py`](ingestion/loader.py:1), [`ingestion/structure.py`](ingestion/structure.py:1) | TC-02 | Yes |
| FR-04 | [`ingestion/structure.py`](ingestion/structure.py:1) | TC-01, TC-03, TC-04 | Yes |
| FR-05 | [`ingestion/chunking_pair2.py`](ingestion/chunking_pair2.py:1), [`ingestion/chunking_pair3.py`](ingestion/chunking_pair3.py:1) | TC-05, TC-10 | Yes |
| FR-06 | [`ingestion/chunking_pair1.py`](ingestion/chunking_pair1.py:1), [`ingestion/chunking_pair2.py`](ingestion/chunking_pair2.py:1), [`ingestion/chunking_pair3.py`](ingestion/chunking_pair3.py:1) | TC-09, TC-10 | Yes |
| FR-07 | [`ingestion/chunk_utils.py`](ingestion/chunk_utils.py:1) | TC-06, TC-07, TC-08, TC-11 | Yes |
| FR-08 | [`ingestion/chunking_pair3.py`](ingestion/chunking_pair3.py:1) | TC-11 | Yes |
| FR-09 | [`models/embeddings.py`](models/embeddings.py:1) | TC-18, TC-19, TC-20 | Yes |
| FR-10 | [`vectorstore/store.py`](vectorstore/store.py:1) | TC-27 | No (end-to-end) |
| FR-11 | [`vectorstore/store.py`](vectorstore/store.py:1) | TC-27 | No (end-to-end) |
| FR-12 | [`retrieval/pair2_retriever.py`](retrieval/pair2_retriever.py:1), [`retrieval/pair3_retriever.py`](retrieval/pair3_retriever.py:1) | TC-28 | No (end-to-end) |
| FR-13 | [`graph/builder.py`](graph/builder.py:1), [`retrieval/pair3_retriever.py`](retrieval/pair3_retriever.py:1) | TC-28 | No (end-to-end) |
| FR-14 | [`models/reranker.py`](models/reranker.py:1), [`retrieval/pair3_retriever.py`](retrieval/pair3_retriever.py:1) | TC-28 | No (end-to-end) |
| FR-15 | [`retrieval/pair2_retriever.py`](retrieval/pair2_retriever.py:1) | TC-28 | No (end-to-end) |
| FR-16 | [`retrieval/pair1_retriever.py`](retrieval/pair1_retriever.py:1), [`retrieval/pair2_retriever.py`](retrieval/pair2_retriever.py:1), [`retrieval/pair3_retriever.py`](retrieval/pair3_retriever.py:1) | TC-28 | No (end-to-end) |
| FR-17 | [`chain/rag_chain.py`](chain/rag_chain.py:1) | TC-29 | No (end-to-end) |
| FR-18 | [`chain/rag_chain.py`](chain/rag_chain.py:1) | TC-29 | No (end-to-end) |
| FR-19 | [`ui/app.py`](ui/app.py:1), [`ui/components.py`](ui/components.py:1) | TC-23 | Yes (render) |
| FR-20 | [`eval/metrics.py`](eval/metrics.py:1), [`eval/judge.py`](eval/judge.py:1), [`eval/runner.py`](eval/runner.py:1) | TC-12–TC-17, TC-22, TC-25 | Yes |
| FR-21 | [`eval/judge.py`](eval/judge.py:1) | TC-30 | No (end-to-end) |
| FR-22 | [`eval/judge.py`](eval/judge.py:1) | TC-30 | No (end-to-end) |
| FR-23 | [`scripts/inspect_chunks.py`](scripts/inspect_chunks.py:1), [`ui/chunk_explorer.py`](ui/chunk_explorer.py:1) | TC-21 | Yes |
| FR-24 | [`config.py`](config.py:1), [`models/llm.py`](models/llm.py:1) | TC-24 | Yes |
| FR-25 | [`ui/app.py`](ui/app.py:1), [`ui/pipeline_registry.py`](ui/pipeline_registry.py:1) | TC-21, TC-22, TC-23 | Yes |
| FR-26 | [`ui/pipeline_registry.py`](ui/pipeline_registry.py:1) | TC-23 | Yes (render) |

The matrix shows that the deterministic, model-free suite covers the parsing, chunking, embedding-text-handling, metric and UI-shaping logic, while the retrieval, generation and end-to-end evaluation paths are verified by the evaluation harness and manual runs rather than by unit tests. This division is deliberate: the unit suite must stay fast and free of model loads, and the end-to-end behaviour is measured by the evaluation harness described in Chapter 6. The next chapter presents the methodology that these requirements specify.
