## Chapter 5: Implementation Details

### 5.1 Development Environment

The system is implemented in Python 3.12 and runs on a single workstation. The development and evaluation environment is an 8 GB machine, and the implementation reflects that ceiling throughout: Docling runs with batch size 1, only one pipeline is cached at a time, pipelines are constructed sequentially in compare mode, and the reranking step is guarded against resource failure. Device resolution is `cuda > mps > cpu`, overridable through the `DEVICE` environment variable, so the same code runs on a CUDA GPU, an Apple MPS device, or CPU alone. Every module begins with `from __future__ import annotations`, and configuration is centralised in [`config.py`](config.py:1), which is a leaf module with no heavy imports because it must set the PyTorch environment variables before torch is first imported.

### 5.2 Software and Tools

The implementation uses the open-source stack listed in Table 3.3. The principal tools are LangChain and LangChain Classic for the document, retriever and LCEL primitives; Docling for layout-aware PDF conversion; Chroma for the local vector store; `rank_bm25` for sparse retrieval; NetworkX for the cross-reference graph; `sentence-transformers` and `transformers` for local embedding and cross-encoder inference; Ollama for the local chat model; Streamlit for the interface; and pytest for the test suite. PyTorch is installed separately per operating system from `requirements-linux-cuda.txt` or `requirements-macos.txt`, because the correct wheel differs between CUDA and Apple Silicon. The pinned versions are recorded in `requirements.txt` and `requirements.lock`.

### 5.3 Project Directory Structure

The repository is organised into packages by responsibility. Figure 5.1 shows the directory structure.

**Fig. 5.1: Project directory structure**

```text
Rags_2_Riches/
├── config.py                     # central configuration (leaf module, no heavy imports)
├── requirements.txt              # platform-agnostic dependencies
├── requirements-linux-cuda.txt   # PyTorch for CUDA
├── requirements-macos.txt        # PyTorch for Apple Silicon
├── Iso/                          # the three ISO 15118 PDFs (corpus)
│   ├── ISO-15118-1.pdf
│   ├── ISO-15118-2.pdf
│   └── ISO-15118-3.pdf
├── ingestion/                    # PDF parsing, structure recovery, chunking
│   ├── loader.py                 # Docling DocumentConverter driver + JSON cache
│   ├── structure.py              # clause outline from Docling labels
│   ├── chunk_utils.py            # pack_units(), split_table_markdown()
│   ├── chunking_pair1.py         # recursive parent/child
│   ├── chunking_pair2.py         # section-aware
│   └── chunking_pair3.py         # table-first + cross-references
├── vectorstore/
│   └── store.py                  # batched, sanitised Chroma writes
├── retrieval/
│   ├── pair1_retriever.py        # parent-document retriever
│   ├── pair2_retriever.py        # hybrid dense + BM25 + RRF + filter
│   └── pair3_retriever.py        # self-query + ensemble + graph + rerank
├── graph/
│   └── builder.py                # NetworkX cross-reference graph + expansion
├── chain/
│   └── rag_chain.py              # condensation + answer LCEL chains
├── models/
│   ├── device.py                 # cuda > mps > cpu resolution
│   ├── embeddings.py             # Qwen3 wrapper, prompt + table stripping
│   ├── reranker.py               # cross-encoder wrapper
│   └── llm.py                    # cached chat clients (Ollama / OpenAI)
├── eval/                         # gold set, metrics, judge, runners, reports
│   ├── build_gold_set.py
│   ├── metrics.py
│   ├── judge.py
│   ├── runner.py
│   ├── run_pairing.py
│   ├── pair_report.py
│   └── comparison_report.py
├── ui/                           # Streamlit application
│   ├── app.py
│   ├── pipeline_registry.py
│   ├── charts.py
│   ├── components.py
│   ├── eval_dashboard.py
│   ├── eval_data.py
│   └── chunk_explorer.py
├── scripts/
│   └── inspect_chunks.py         # chunk-quality diagnostic
└── tests/                        # fast, model-free unit tests
    ├── test_structure.py
    ├── test_metrics.py
    ├── test_embeddings.py
    ├── test_chunk_explorer.py
    ├── test_eval_dashboard.py
    ├── test_ui_app.py
    └── smoke_test.py
```

Runtime artifacts are git-ignored: `data/parsed/` (the Docling JSON cache), `data/indexes/pairN/`, and `chroma_db/`.

### 5.4 Document Parsing Module

The document parsing module is [`ingestion/loader.py`](ingestion/loader.py:1). It drives Docling's `DocumentConverter` directly rather than using a Markdown export helper, so that each PDF remains one LangChain document and page boundaries can be derived from a form-feed placeholder inserted during serialisation. Running headers and footers are stripped, OCR is disabled because the source PDFs already contain a text layer, and the converted structure is cached as JSON under `data/parsed/` so that repeated runs skip conversion entirely. The public entry point `load_iso_structures()` returns a list of `StructuredDocument` objects. The module re-asserts the PyTorch environment variables defensively, because it may be imported before `config.py` in some entry points.

### 5.5 Chunking and Metadata Modules

The structure-recovery module is [`ingestion/structure.py`](ingestion/structure.py:1). It turns Docling's item labels into the outline: `section_header` items become sections, `TableItem`s become table blocks, clause numbers are normalised (numeric clauses such as `4.1.2`, annex labels such as `A` and `A.1`), and a heading emitted as two consecutive headers is merged into one clause number and title. Front-matter and furniture headings are dropped. The module exposes `parse_heading()`, `is_furniture_heading()`, `build_sections()`, and the `Section`, `Block` and `StructuredDocument` types.

The shared chunking helpers are in [`ingestion/chunk_utils.py`](ingestion/chunk_utils.py:1): `pack_units()` fills a character budget with whole units, and `split_table_markdown()` breaks an oversized table on row boundaries while repeating the header row. The three chunkers all consume `list[StructuredDocument]`. [`ingestion/chunking_pair1.py`](ingestion/chunking_pair1.py:1) produces recursive parent/child chunk pairs. [`ingestion/chunking_pair2.py`](ingestion/chunking_pair2.py:1) produces section-bounded chunks with atomic tables. [`ingestion/chunking_pair3.py`](ingestion/chunking_pair3.py:1) produces standalone table chunks with captions, clause-level text chunks, and records the cross-references found in each chunk. The metadata attached by the chunkers is described in Table 4.2.

### 5.6 Indexing Module

The indexing module is [`vectorstore/store.py`](vectorstore/store.py:1), which owns all writes to Chroma. `add_documents_in_batches()` enforces a batch size below the store's single-upsert cap and removes `None` and empty-list metadata values that Chroma rejects. An `ensure_indexed` helper checks whether the persisted index matches the current chunking fingerprint and triggers a rebuild only when it does not. Routing every write through this single helper removes a whole class of ingestion failures: the store's upsert limit and its metadata validation cannot be violated from a second code path.

### 5.7 Hybrid Retrieval Modules

The retrieval modules implement the three pairings. [`retrieval/pair1_retriever.py`](retrieval/pair1_retriever.py:1) builds a parent-document retriever over the child/parent pairs. [`retrieval/pair2_retriever.py`](retrieval/pair2_retriever.py:1) builds the dense-plus-BM25 retriever fused with reciprocal rank fusion and exposes a metadata-filtered variant keyed on `section_number`. [`retrieval/pair3_retriever.py`](retrieval/pair3_retriever.py:1) composes the self-query retriever, the BM25 retriever, graph expansion and the cross-encoder reranker over the top candidate list, wrapping the reranking step in a guard so that an out-of-memory condition returns the unreranked list.

### 5.8 Cross-Reference Graph Module

The graph module is [`graph/builder.py`](graph/builder.py:1). It creates a `networkx.DiGraph` whose node ids are `{document_id}:{section}_{md5[:8]}` and whose edges come from the cross-references recorded during chunking. A breadth-first helper expands a seed set by a bounded number of hops and returns the neighbouring chunks with a reduced score. Node identifiers are qualified by document part so that identically numbered clauses in different parts do not collide.

### 5.9 Reranking Module

The reranking module is [`models/reranker.py`](models/reranker.py:1), which loads the `BAAI/bge-reranker-v2-m3` cross-encoder and scores query–passage pairs jointly. The maximum sequence length is raised to 1024 because table chunks exceed the default 512. The reranker is invoked only over the top `RERANK_TOP_N = 20` candidates, and the call is guarded so that a resource failure degrades to the unranked list.

### 5.10 Answer Generation Module

The generation module is [`chain/rag_chain.py`](chain/rag_chain.py:1), which builds two LCEL chains: a condensation chain that rewrites a follow-up question into a standalone question using the conversation history, and the answer chain, whose system prompt confines the answer to the supplied context, requires a document and clause citation, and instructs the model to state that it lacks sufficient information otherwise. Retrieval is executed by the caller so that the identical context object feeds both the prompt and the UI source panel. The chat clients are cached in [`models/llm.py`](models/llm.py:1), which switches provider between a local Ollama server and an optional OpenAI endpoint.

### 5.11 Evaluation Harness

The evaluation harness is the `eval/` package. [`eval/build_gold_set.py`](eval/build_gold_set.py:1) assembles the question set with gold answers, answer terms and gold clause references. [`eval/metrics.py`](eval/metrics.py:1) computes deterministic retrieval and chunk statistics using answer-term relevance, plus a section-based signal when clause metadata is available. [`eval/judge.py`](eval/judge.py:1) scores qualitative properties — relevance density, boundary coherence, faithfulness, answer relevance and answer correctness — and checks citations deterministically by comparing the clause numbers cited in the answer with the retrieved sections; uncomputable scores are stored as `None` and surfaced as flags rather than dropped. [`eval/runner.py`](eval/runner.py:1) executes a gold set against a pipeline, [`eval/run_pairing.py`](eval/run_pairing.py:1) is the command-line entry point for one pairing, and [`eval/pair_report.py`](eval/pair_report.py:1) and [`eval/comparison_report.py`](eval/comparison_report.py:1) render the markdown output.

### 5.12 Streamlit User Interface

The interface is the `ui/` package. [`ui/pipeline_registry.py`](ui/pipeline_registry.py:39) exposes `PIPELINE_NAMES` and `build_pipeline()`, which constructs one pipeline at a time and caches it. [`ui/app.py`](ui/app.py:1) renders the three tabs, and [`ui/charts.py`](ui/charts.py:1), [`ui/components.py`](ui/components.py:1), [`ui/eval_dashboard.py`](ui/eval_dashboard.py:1), [`ui/eval_data.py`](ui/eval_data.py:1) and [`ui/chunk_explorer.py`](ui/chunk_explorer.py:1) provide charting, citation panels, the evaluation dashboard and the chunk browser. Because Streamlit places `ui/` on the path rather than the repository root, the UI modules insert the root into `sys.path` themselves, and the application must be launched from the repository root.

### 5.13 Integration of Modules

The modules are integrated through the pipeline registry. `build_pipeline()` in [`ui/pipeline_registry.py`](ui/pipeline_registry.py:39) maps each of the three display names to a branch that imports its own chunker and retriever, constructs the chain, and returns the retriever, chain, condensation chain, vector store and graph. Only the selected pipeline is cached in single-pipeline mode; in compare mode the registry builds each pipeline in turn, collects its answers, and releases it before building the next one, because holding all three pipelines simultaneously was out-of-memory on the development machine. The evaluation harness invokes the same retrievers and chain, so the measurement path and the interactive path observe the same behaviour. The chunk-inspection diagnostic [`scripts/inspect_chunks.py`](scripts/inspect_chunks.py:1) is run after any chunking change; it reports chunk counts, per-document and per-section coverage, oversized sections, and table and graph statistics, so that a regression in structure recovery is visible before retrieval quality is measured.

### 5.14 Implementation Status

Table 5.1 records the implementation status of every component, with file-level evidence. All core components of the three pairings, the evaluation harness and the interface are implemented. The items marked "Not yet implemented" are the extensions identified as future work in Chapter 7: NLI-based grounding evaluation, adaptive-k context selection, multi-run variance analysis, and incremental indexing.

**Table 5.1: Implementation status of system components**

| Component | Planned technology | Intended responsibility | Status | Evidence / outstanding work |
|---|---|---|---|---|
| PDF parsing | Docling `DocumentConverter` | Layout-aware conversion, JSON cache, page boundaries | Implemented | [`ingestion/loader.py`](ingestion/loader.py:1) |
| Structure recovery | Docling labels | Clause outline, annex handling, page provenance | Implemented | [`ingestion/structure.py`](ingestion/structure.py:1) |
| Chunking — baseline | Recursive splitter | Parent/child chunks | Implemented | [`ingestion/chunking_pair1.py`](ingestion/chunking_pair1.py:1) |
| Chunking — section-aware | Section-bounded packer | Clause-aligned chunks, atomic tables | Implemented | [`ingestion/chunking_pair2.py`](ingestion/chunking_pair2.py:1) |
| Chunking — table-first | Table-first packer | Standalone table chunks, cross-references | Implemented | [`ingestion/chunking_pair3.py`](ingestion/chunking_pair3.py:1) |
| Chunk helpers | `pack_units`, `split_table_markdown` | Table atomicity, row-boundary splitting | Implemented | [`ingestion/chunk_utils.py`](ingestion/chunk_utils.py:1) |
| Vector store | Chroma via `langchain-chroma` | Batched, sanitised writes; freshness check | Implemented | [`vectorstore/store.py`](vectorstore/store.py:1) |
| Dense embeddings | `Qwen/Qwen3-Embedding-0.6B` | Query prompt, table flattening, normalisation | Implemented | [`models/embeddings.py`](models/embeddings.py:1) |
| Sparse retrieval | `rank_bm25` | BM25 lexical index | Implemented | [`retrieval/pair2_retriever.py`](retrieval/pair2_retriever.py:1) |
| Hybrid fusion | Reciprocal rank fusion | Dense + BM25 combination | Implemented | [`retrieval/pair2_retriever.py`](retrieval/pair2_retriever.py:1) |
| Cross-reference graph | NetworkX `DiGraph` | Deterministic edges, bounded BFS expansion | Implemented | [`graph/builder.py`](graph/builder.py:1) |
| Metadata-filtered retrieval | `SelfQueryRetriever` | Clause-constrained search | Implemented | [`retrieval/pair3_retriever.py`](retrieval/pair3_retriever.py:1) |
| Reranking | `BAAI/bge-reranker-v2-m3` | Cross-encoder second pass, guarded | Implemented | [`models/reranker.py`](models/reranker.py:1) |
| Answer generation | LangChain LCEL + Ollama | Cited, grounded answers; condensation | Implemented | [`chain/rag_chain.py`](chain/rag_chain.py:1) |
| Chat clients | Ollama / OpenAI | Cached provider switch | Implemented | [`models/llm.py`](models/llm.py:1) |
| Device resolution | `cuda > mps > cpu` | Compute-device selection | Implemented | [`models/device.py`](models/device.py:1) |
| Gold set | `build_gold_set.py` | 32-query gold set with terms and sections | Implemented | [`eval/gold_set.json`](eval/gold_set.json:1) |
| Retrieval metrics | `metrics.py` | Recall, Precision, nDCG, MRR, chunk stats | Implemented | [`eval/metrics.py`](eval/metrics.py:1) |
| LLM-as-judge | `judge.py` | Faithfulness, relevance, correctness, citation accuracy | Implemented | [`eval/judge.py`](eval/judge.py:1) |
| Evaluation runner | `runner.py`, `run_pairing.py` | Replay gold set, persist JSON | Implemented | [`eval/runner.py`](eval/runner.py:1) |
| Reports | `pair_report.py`, `comparison_report.py` | Per-pairing and comparison markdown | Implemented | [`eval/comparison_report.py`](eval/comparison_report.py:1) |
| Chunk diagnostic | `inspect_chunks.py` | Counts, coverage, oversized sections, table/graph stats | Implemented | [`scripts/inspect_chunks.py`](scripts/inspect_chunks.py:1) |
| Interface | Streamlit | Ask/chat, evaluation, chunk browser | Implemented | [`ui/app.py`](ui/app.py:1) |
| Pipeline registry | `pipeline_registry.py` | One-at-a-time pipeline construction | Implemented | [`ui/pipeline_registry.py`](ui/pipeline_registry.py:39) |
| Test suite | pytest | Fast, model-free unit tests | Implemented | [`tests/`](tests) |
| NLI-based grounding | `cross-encoder/nli-deberta-v3-base` | Entailment-based grounding check | Not yet implemented | Future work (Chapter 7) |
| Adaptive-k selection | Score-gap heuristic | Variable context size (1–8) | Not yet implemented | Future work (Chapter 7) |
| Multi-run variance | Repeated evaluation runs | Mean ± std of judged scores | Not yet implemented | Future work (Chapter 7) |
| Incremental indexing | Fingerprint-based updates | Re-index only changed chunks | Not yet implemented | Future work (Chapter 7) |

The implementation status table shows that the system described in Chapter 4 is fully implemented for the three pairings, the evaluation harness and the interface, and that the extensions named in the original proposal — NLI-based grounding and adaptive-k selection — are explicitly deferred to future work rather than claimed as implemented. The next chapter presents the measured results obtained from this implementation.
