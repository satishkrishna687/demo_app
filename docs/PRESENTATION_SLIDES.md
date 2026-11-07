# Structure-Aware RAG for ISO 15118
### Benchmarking Chunking × Retrieval Strategies
#### Presentation slides (one section per slide; `---` separates slides)

> Source of numbers: [`eval/RAG_EVAL_REPORT.md`](../eval/RAG_EVAL_REPORT.md) (32 queries).
> Each results **graph slide** is followed by a dedicated **explanation slide** (what we compare + the result).

---

# Slide 1 — Title

## Structure-Aware Retrieval-Augmented Generation
### A Comparative Study of Chunking and Retrieval Strategies on ISO 15118

- Local, citation-grounded Q&A over the three ISO 15118 specification PDFs
- Answers with verifiable **clause/table citations**
- Rigorous comparison of **3 chunking × retrieval "pairings"**

---

# Slide 2 — Why this problem?

- **The corpus resists search**: ISO 15118 Parts 1–3 = ~450 pages of dense clauses, parameter tables, cross-references.
- **Engineers need precision, not vibes**: answers must cite the exact clause/table; hallucination is unacceptable.
- **Keyword search misses paraphrases; chat LLMs hallucinate.**
- **Goal**: a *local, private* RAG pipeline **plus** a fair comparison of three chunking × retrieval strategies.

---

# Slide 3 — Models & stack

| Stage | Choice |
|---|---|
| PDF parsing | **Docling** (layout + TableFormer); tables → Markdown |
| Embeddings | **Qwen/Qwen3-Embedding-0.6B** (1024-d, local, query-instruction) |
| Keyword search | **BM25** (hybrid) |
| Reranker | **BAAI/bge-reranker-v2-m3** (Pairing 3) |
| Knowledge graph | **NetworkX** (cross-references) |
| Vector store | **Chroma** |
| LLM | **Llama 3.2 3B** via Ollama (OpenAI optional) |
| Orchestration | **LangChain (LCEL)** |

*Everything runs locally — no API key required.*

---

# Slide 4 — Pairing 1: Recursive + Parent-Child

- **Chunking**: recursive character splitting → **parent (1024)** + **child (256)** chunks.
- **Retrieval**: match small **children**, return the larger **parent** (`ParentDocumentRetriever`).
- **Idea**: small chunks embed precisely; parents restore context.
- ✅ Fast, simple, strong baseline
- ⚠️ Structure-blind — cuts straight through clauses/tables

---

# Slide 5 — Pairing 2: Section + Hybrid

- **Chunking**: **structure-aware** — splits at clause/annex boundaries (Docling labels); tables kept atomic.
- **Retrieval**: **hybrid** = dense (Chroma) + sparse (BM25) fused by `EnsembleRetriever`; supports **section filtering**.
- **Idea**: respect the document hierarchy; combine semantics + exact terms.
- ✅ Precise clause scope; best precision@5 among P1/P2
- ⚠️ No table- or reference-level reasoning

---

# Slide 6 — Pairing 3: Advanced ISO + Graph

- **Chunking**: **table-first** standalone table chunks + clause chunks + **cross-reference metadata**.
- **Retrieval (4 stages)**: metadata candidate gen → hybrid (vector + BM25) → **graph expansion** → **cross-encoder rerank**.
- **Idea**: ISO docs are tables + references — follow the references.
- ✅ Best retrieval & citation accuracy
- ⚠️ Slowest retrieval (graph + rerank)

---

# Slide 7 — Retrieval quality (graph)

**What the chart shows:** grouped bars of Recall@5, nDCG@5, MRR for the three pairings.

| Metric | P1 | P2 | P3 |
|---|---|---|---|
| Recall@5 | 0.94 | 0.88 | **1.00** |
| nDCG@5 | 0.89 | 0.84 | **0.94** |
| MRR | 0.90 | 0.85 | **0.93** |
| Precision@5 | 0.54 | 0.61 | **0.72** |

> 🖼️ **Attach here**: *Retrieval quality* grouped bar (Recall@5 / nDCG@5 / MRR).
> Optional second image: *Recall@5 by query category* heatmap.

---

# Slide 8 — Retrieval quality (what & result)

**What we are comparing** — how well each pairing *finds the right passage*:
- **Recall@5** — is at least one relevant chunk in the top 5? (coverage)
- **nDCG@5** — are relevant chunks ranked near the top? (ranking quality)
- **MRR** — how high is the first relevant chunk? (first-hit quality)
- **Precision@5** — what fraction of the top 5 is actually relevant? (noise)

**Result of comparing:**
- **Pairing 3 wins all four** — Recall@5 = **1.00** (never misses), nDCG 0.94, MRR 0.93, precision 0.72.
- **Pairing 1** is a strong second on ranking (0.94 / 0.89 / 0.90) but noisier (precision 0.54).
- **Pairing 2** has the lowest recall/nDCG but better precision than P1 (0.61).
- **Why**: graph expansion + reranking (P3) surfaces the right clauses; section-aware hybrid (P2) is precise but misses some; recursive (P1) finds content with more noise.

---

# Slide 9 — Generation quality (graph)

**What the chart shows:** grouped bars of Faithfulness, Answer relevance, Answer correctness, Citation accuracy.

| Metric | P1 | P2 | P3 |
|---|---|---|---|
| Faithfulness | **0.61** | 0.57 | 0.51 |
| Answer relevance | 0.67 | 0.69 | **0.73** |
| Answer correctness | 0.70 | 0.70 | **0.73** |
| Citation accuracy | 0.00 | 0.93 | **0.95** |

> 🖼️ **Attach here**: *Generation quality* grouped bar (Faithfulness / Relevance / Correctness / Citation accuracy).

---

# Slide 10 — Generation quality (what & result)

**What we are comparing** — the quality of the *final answer*:
- **Faithfulness** — is the answer grounded in the retrieved context? (no hallucination)
- **Answer relevance** — does it actually address the question?
- **Answer correctness** — does it match the gold answer?
- **Citation accuracy** — do the cited sources support the claims?

**Result of comparing:**
- **Pairing 3 wins relevance (0.73), correctness (0.73) and citations (0.95)** — better retrieval → better answers.
- **Pairing 2 is a close second** on relevance (0.69), correctness (0.70) and citations (0.93) — structure + hybrid gives solid, verifiable answers.
- **Pairing 1 has the best faithfulness (0.61)** but its **citation accuracy is 0.00**.
- **Why P1 citations = 0**: its recursive chunks carry **no section numbers**, so citations can't be verified — a structural gap, not a model failure.
- **Faithfulness is close across all (0.51–0.61)** — the small 3B LLM/judge is the limiting factor, not retrieval.

---

# Slide 11 — Quality vs latency & cost (graph)

**What the chart shows:** bubble chart — x = E2E latency, y = citation accuracy, bubble size = index size.

| | P1 | P2 | P3 |
|---|---|---|---|
| E2E (ms) | **2,459** | 7,633 | 6,756 |
| Retrieval (ms) | **108** | 5,745 | 4,122 |
| Index (MiB) | 55 | 29 | **27** |
| Tokens/query | 2,285 | 2,693 | **1,893** |

> 🖼️ **Attach here**: *Quality vs Latency* bubble chart.
> Add below it: the *Efficiency* table (Indexing s / Retrieval ms / E2E ms / Index MiB / Tokens per query).

---

# Slide 12 — Quality vs latency & cost (what & result)

**What we are comparing** — the **trade-off** between answer quality and speed/cost:
- **x-axis**: end-to-end latency (lower is better)
- **y-axis**: citation accuracy (higher is better)
- **bubble size**: index size (smaller is better)

**Result of comparing:**
- **Pairing 1 is fastest**: 108 ms retrieval, 2.5 s end-to-end — but lowest citation accuracy.
- **Pairing 3 has the best quality at ~2.7× the latency** (6.8 s); the cost is almost entirely **retrieval** (4.1 s vs 108 ms) from graph expansion + reranking.
- **Pairing 2 sits in the middle on quality** (0.93 citations) but its latency is **noisy** — it measured 7.6 s here vs 2.3 s in an earlier run, so treat single-run latency as indicative, not precise.
- **Index sizes are similar** (27–55 MiB); **P3 uses the fewest tokens/query** (1,893) because graph + rerank return tighter context.
- **Ideal point** = top-left, small bubble → P1 is fast-but-lower; P3 is high-but-right.

---

# Slide 13 — Chunking characteristics (graph)

**What the chart shows:** bars of *Chunks per pairing* and *Avg chunk size (chars)*.

| | P1 | P2 | P3 |
|---|---|---|---|
| Chunks | 2,084 | 2,128 | 2,007 |
| Avg chunk (chars) | 709 | 678 | 710 |
| Relevance density | 0.14 | 0.22 | **0.26** |
| Boundary coherence | **0.35** | 0.28 | 0.33 |

> 🖼️ **Attach here**: *Chunks per pairing* and *Avg chunk size (chars)* bars (Chunking characteristics section).

---

# Slide 14 — Chunking characteristics (what & result)

**What we are comparing** — the *chunk corpus* each pairing produces:
- **Chunks** — count (index size / cost proxy)
- **Avg chunk size** — how much context each chunk carries
- **Relevance density** — how much of a retrieved chunk is actually useful
- **Boundary coherence** — are chunks self-contained, or split mid-thought/table?

**Result of comparing:**
- **Scale is similar** across pairings (~2,000–2,130 chunks; ~680–710 chars) — so differences come from *structure*, not size.
- **Relevance density rises with structure**: P1 **0.14** → P2 **0.22** → P3 **0.26** — structure-aware, table-first chunking packs more useful content per chunk.
- **Boundary coherence is low across the board (0.28–0.35)** — a clear weak spot and the top chunking improvement target.
- **Takeaway**: clause/annex boundaries + atomic tables + references (P3) → densest, most useful chunks.

---

# Slide 15 — Recommendation

- **Accuracy-critical / offline → Pairing 3**: Recall@5 = 1.00, citation accuracy 0.95.
- **Interactive / latency-sensitive → Pairing 1**: 108 ms retrieval, 2.5 s end-to-end.
- **Best balance → Pairing 2**: strong precision (0.61), citations 0.93, smallest index after P3.
- All three run **fully locally**; the chat LLM can switch to OpenAI via one `.env` flag.

---

# Slide 16 — What can be improved (achievable)

- **Chat memory**: now implemented in the UI (history-aware conversational RAG).
- **Fix P1 citations**: attach section metadata to recursive chunks (citation accuracy 0 → high).
- **Speed up Pairing 3**: cap graph fan-out, cache/rerank in batches (4.1 s retrieval → target < 1 s).
- **Larger embeddings**: Qwen3-Embedding-4B or BGE-M3, reranker on CPU (fits RTX 3050, 8 GB).
- **Raise boundary coherence** (0.28–0.35): sentence-aware packing; never split clauses/tables.
- **More robust evaluation**: judge all 32 queries and add `--runs 3` for mean ± std.

---

## Appendix — Graph attachment map (quick reference)

| Slide | Graph to attach | Where in the UI |
|---|---|---|
| 7 – Retrieval quality (graph) | Retrieval quality grouped bar (+ category heatmap) | Evaluation & Comparison tab → "Retrieval quality" |
| 9 – Generation quality (graph) | Generation quality grouped bar | Evaluation & Comparison tab → "Generation quality" |
| 11 – Quality vs latency (graph) | Quality vs Latency bubble + Efficiency table | Evaluation & Comparison tab → "Quality vs latency" / "Efficiency" |
| 13 – Chunking characteristics (graph) | Chunks per pairing + Avg chunk size bars | Evaluation & Comparison tab → "Chunking characteristics" |
| Extra (optional) | Chunk-size histogram, content-type breakdown, metadata coverage | 🧩 Chunks tab |

*Numbers are a single run (`--runs 1`, 32 queries); generation metrics are averaged over the judged subset (see `judge_coverage`). Latency varies between runs on this hardware — treat it as indicative.*
