# Plan: M.Tech Phase 1 Project Report Generation

**Output file:** `Grounded_RAG_QA_ISO15118_Phase1_Report.md` (repo root, per `Report_generation.md` §9)
**Source of truth:** the actual repository (per user decision) — three pairings, reranker, real measured results from `eval/RAG_EVAL_REPORT.md`. `Report_generation.md` supplies the *structure, style and integrity rules*; where its named technologies differ from the repo, the repo wins and the deviation is documented in the report itself.

## Guiding rules (from Report_generation.md, adapted)

1. **Evidence discipline.** Distinguish: established background / implemented work / measured results / expectations. The repo HAS measured results (32 gold queries, single run), so Chapter 6 presents them as real but single-run (no variance yet). Anything not implemented stays "planned" with placeholders.
2. **No fabrication.** No invented papers, DOIs, page numbers, personal/institutional details, or metrics beyond what exists in `eval/RAG_EVAL_REPORT.md`, `eval/result_pairN.json`, and `config.py`.
3. **No copyrighted ISO text.** Explain concepts in original words; brief clause references only.
4. **Academic style.** Formal prose paragraphs (not all bullets), numbered chapters/sections/tables/figures, IEEE numeric citations `[n]`, Mermaid diagrams, LaTeX where useful, 30–45 pages target.
5. **Consistency.** Reuse the 12-entry IEEE reference list from `docs/STAGE1_REFERENCES.md` (master numbering) and the terminology of the Stage 1 documents.

## Source map (what feeds each chapter)

| Chapter | Primary sources |
|---|---|
| Preliminary | `Report_generation.md` §2 (title/placeholders), `docs/STAGE1_FIRST_REVIEW_PPT.md` slide 1–2 |
| 1 Introduction | `docs/STAGE1_FIRST_REVIEW_PPT.md` slides 2–4, `RAG_SPECIFICATION.md` |
| 2 Literature Survey | `docs/STAGE1_LITERATURE_SURVEY.md` (reuse + expand), `docs/STAGE1_REFERENCES.md` |
| 3 Requirements | `docs/STAGE1_PROPOSED_SYSTEM.md` §8–9 (FR-01…FR-26, NFRs), `config.py`, `tests/` |
| 4 Methodology | `ingestion/loader.py`, `ingestion/structure.py`, `ingestion/chunking_pair1/2/3.py`, `ingestion/chunk_utils.py`, `retrieval/pair1/2/3_retriever.py`, `graph/builder.py`, `chain/rag_chain.py`, `models/embeddings.py`, `models/reranker.py`, `config.py` |
| 5 Implementation | module tree, `vectorstore/store.py`, `ui/pipeline_registry.py`, `ui/app.py`, `scripts/inspect_chunks.py`, `tests/`, `requirements.txt` |
| 6 Results & Discussion | `eval/RAG_EVAL_REPORT.md`, `eval/result_pair1/2/3.json`, `eval/metrics.py`, `eval/judge.py`, `eval/runner.py`, `eval/gold_set.json` |
| 7 Conclusion | Stage 1 docs + repo state |
| References | `docs/STAGE1_REFERENCES.md` (verbatim 12 entries) |

## Execution steps (each step = one delegated task, sequential)

- [ ] **Step 0 — Consistency brief.** (Done implicitly: each task receives the canonical reference list, terminology glossary — "pairing", "chunk", "clause", "RRF", "reranker" — and the evidence rules above.)
- [ ] **Step 1 — Preliminary pages + Chapter 1 (Introduction).** Title page, certificate/acknowledgement templates (placeholders), abstract (mentions measured results exist but single-run), TOC skeleton, list of figures/tables (placeholder page numbers), abbreviations. §1.1–1.10 per the prompt, adapted to the real system (three pairings instead of NLI/adaptive-k; NLI/adaptive-k may appear in future work only).
- [ ] **Step 2 — Chapter 2 (Literature Survey).** §2.1–2.13. Expand `docs/STAGE1_LITERATURE_SURVEY.md` into prose; keep the 12 canonical references; comparative table with the prompt's five columns; research gap = combination of structure-aware parsing + hybrid retrieval + cross-reference graph + reranking + reproducible evaluation on standards text.
- [ ] **Step 3 — Chapter 3 (System Requirements).** §3.1–3.10. Reuse FR-01…FR-26 and NFRs from `docs/STAGE1_PROPOSED_SYSTEM.md`; add test-case table (mark executed pytest coverage as executed, others "Not yet executed") and traceability matrix mapping FRs → modules → tests.
- [ ] **Step 4 — Chapter 4 (Proposed Methodology).** §4.1–4.17, restructured to the real system: architecture, workflow, Docling ingestion, structure recovery, three chunking strategies (with table atomicity), metadata enrichment, dense indexing (Qwen3-Embedding-0.6B), BM25, hybrid retrieval + RRF, cross-reference graph expansion, cross-encoder reranking, generation with citations, evaluation methodology. Pseudocode for chunking, RRF fusion, graph expansion, reranking. Mermaid architecture diagram with baseline (Pairing 1) vs proposed (Pairings 2/3) paths.
- [ ] **Step 5 — Chapter 5 (Implementation Details).** §5.1–5.13. Real directory tree, module responsibilities, implementation-status table (all core components = implemented with file evidence; future items = "Not yet implemented").
- [ ] **Step 6 — Chapter 6 (Results and Discussion).** §6.1–6.14. Present the real measured results from `eval/RAG_EVAL_REPORT.md` (Recall@5, nDCG@5, MRR, section recall, faithfulness, answer correctness, citation accuracy, latency, index size, tokens/query per pairing); discuss why Pairing 3 wins retrieval and Pairing 2 wins generation/latency; note single-run limitation (no variance) and Pairing 1's N/A section recall; limitations section.
- [ ] **Step 7 — Chapter 7 + References.** Conclusion, contributions (design + measured evaluation), limitations, future work (multi-run variance, NLI-based grounding check as an extension, corpus expansion, incremental indexing). References = the 12 canonical IEEE entries verbatim.
- [ ] **Step 8 — Assembly and quality check.** Merge all parts into `Grounded_RAG_QA_ISO15118_Phase1_Report.md`; verify against `Report_generation.md` §10 checklist (all 7 chapters, baseline vs proposed distinguished, no fabricated data, consistent citations, figures/tables/abbreviations lists complete, TOC links valid, Mermaid syntax valid).

## Key adaptations to record in the report (deviations from the prompt's assumed design)

| Prompt assumed | Repo reality | Handling |
|---|---|---|
| `bge-small-en-v1.5` embeddings | `Qwen/Qwen3-Embedding-0.6B` | Use repo model; cite HF model card |
| NLI evaluator (`nli-deberta-v3-base`) | LLM-as-judge (`eval/judge.py`) + deterministic citation accuracy | Use judge as the implemented evaluation; NLI as future work |
| Adaptive-k (1–8) | Fixed `TOP_K=5` + `RERANK_TOP_N=20` + graph expansion | Describe real mechanism; adaptive-k as future work |
| Baseline = fixed k=5 without NLI | Baseline = Pairing 1 (recursive + parent-child) | Use Pairing 1 as baseline |
| No results exist | Results exist (single run, 32 queries) | Present as measured, single-run, variance pending |
| Source file `stage1_1_ppt.md` | Does not exist | Use `docs/STAGE1_FIRST_REVIEW_PPT.md` + repo as primary sources |