# Presentation Guide — Detailed Speaker Notes for `PRESENTATION_SLIDES.md`

Companion to [`PRESENTATION_SLIDES.md`](PRESENTATION_SLIDES.md). For every slide this gives you:
**what to say**, **what to emphasise**, **what to point at**, and **likely questions with answers**.

> Pacing: ~60–90 seconds per slide. Spend most time on the results slides (7–14).
> Honesty first: where a number is noisy or based on a subset, say so — mentors reward candour over polish.

---

## Before you start (30 seconds)

> "This is a **structure-aware RAG system** over the ISO 15118 EV-charging specifications. I built **three chunking × retrieval strategies** and evaluated them head-to-head on a 32-query gold set, so the comparison is evidence-based rather than opinion."

Set expectations: "I'll cover the problem, the models, the three strategies, the results, and what I'd improve next."

---

## Slide 1 — Title

**Say:** "The project is *Structure-Aware Retrieval-Augmented Generation* — a comparative study of chunking and retrieval strategies on ISO 15118. It answers questions over the three spec PDFs with verifiable clause/table citations, and it runs fully locally."

**Emphasise:** local/private · citations · *comparative* study (the contribution is the comparison, not just a chatbot).

**Point at:** the title, then the three bullet points.

**Likely questions**
- **"What's the one-line summary?"** → Three chunking × retrieval strategies for a hard, table-heavy standard, compared with a reproducible evaluation harness.
- **"Why this project?"** → Grounded Q&A over standards is a real industrial need, and standards are a genuinely hard RAG testbed (tables, cross-references, dense clauses).
- **"What's novel here?"** → The *structure-aware* chunking (using Docling's document tree, not a regex) plus the graph-expanded retrieval, and the head-to-head evaluation.

---

## Slide 2 — Why this problem?

**Say:** "The corpus is ~450 pages of dense clauses, parameter tables, and cross-references. Engineers need the *exact* clause or table, so hallucination is unacceptable. Keyword search misses paraphrases, and plain chat LLMs hallucinate. So the goal is grounded retrieval **plus** a fair comparison of strategies."

**Emphasise:** the problem is **grounding + precision**, not "chat with PDFs".

**Point at:** the four bullets.

**Likely questions**
- **"Why ISO 15118 specifically?"** → It's a real, table-heavy, cross-referenced standard — a hard testbed with clear ground truth (clause numbers).
- **"Why not just use a bigger LLM?"** → Bigger models still hallucinate without retrieval; and we want local/private. Retrieval quality is the lever.
- **"Why RAG and not fine-tuning?"** → Fine-tuning teaches style, not facts; it can't cite the exact clause and can't be updated when the standard changes. RAG grounds answers in retrievable text.
- **"Who is the user?"** → An engineer or compliance reviewer who needs a specific clause/table value with a citation.

---

## Slide 3 — Models & stack

**Say:** walk the table top to bottom. "Docling parses layout and tables into Markdown. Embeddings are Qwen3-Embedding-0.6B, local. BM25 adds keyword search. A BGE cross-encoder reranks in Pairing 3. NetworkX holds the cross-reference graph. Chroma is the vector store. The chat LLM is Llama 3.2 3B via Ollama. Everything is orchestrated with LangChain LCEL."

**Emphasise:** **everything runs locally — no API key**; only the chat LLM is switchable (Ollama ↔ OpenAI).

**Point at:** each row as you mention it.

**Likely questions**
- **"Why Qwen3-Embedding-0.6B?"** → Strong open-source technical retrieval, small enough for an 8 GB GPU, instruction-aware (we apply its query prompt), and a config flag to scale up.
- **"Why Docling?"** → It preserves layout and reconstructs tables as Markdown, and exposes a structured document tree we use for chunking — crucial for ISO's parameter tables.
- **"Why two retrievers (dense + BM25)?"** → Dense handles paraphrase; BM25 nails exact terms like parameter names and numbers.
- **"Why a cross-encoder reranker?"** → It scores query–chunk pairs jointly, far more accurate than first-stage similarity — applied only to a small candidate set.
- **"Why LangChain?"** → LCEL gives composable, testable retrieval/generation chains, and standard retriever abstractions (ParentDocument, Ensemble, SelfQuery).
- **"Is any data sent externally?"** → No. Models run locally; OpenAI is optional and off by default.

---

## Slide 4 — Pairing 1: Recursive + Parent-Child

**Say:** "This is the naive baseline. Recursive character splitting into parent (1024) and child (256) chunks. At query time we match the small children and return the larger parent."

**Emphasise:** it's the **control** — deliberately structure-blind, so it shows what you get *without* exploiting document structure.

**Point at:** the chunk sizes and the "structure-blind" warning.

**Likely questions**
- **"Why parent-child?"** → Small chunks embed precisely; parents restore surrounding context for the LLM.
- **"Why 1024/256?"** → Standard splitter defaults that balance context vs precision; they follow the original spec.
- **"What's its weakness?"** → It cuts through clauses and tables, and its chunks have no section numbers — hence citation accuracy 0.
- **"Why keep it if it's the weakest?"** → As a baseline; you can't claim structure helps without showing what unstructured chunking does.

---

## Slide 5 — Pairing 2: Section + Hybrid

**Say:** "This one is structure-aware: it splits at clause and annex boundaries using Docling's structured labels, and keeps tables atomic. Retrieval is hybrid — dense vectors plus BM25 — fused by an ensemble, with optional section filtering."

**Emphasise:** boundaries come from **document structure, not a regex**; hybrid = semantics + exact terms.

**Point at:** "structure-aware" and "hybrid".

**Likely questions**
- **"How are section boundaries detected?"** → From Docling's `section_header` labels and the document tree — including annexes (`Annex A`, `A.1`) and non-numeric headings that a regex would miss.
- **"What is the ensemble doing?"** → Reciprocal Rank Fusion combines the vector and BM25 rankings into one list.
- **"How does section filtering work?"** → A Chroma metadata filter on `section_number`, so you can scope retrieval to one clause.
- **"Why are tables kept atomic?"** → Splitting mid-table destroys header–row alignment; tables become single chunks (with the header repeated if oversized).

---

## Slide 6 — Pairing 3: Advanced ISO + Graph

**Say:** "The most advanced. Table-first chunking — each table becomes a standalone chunk — plus clause chunks and cross-reference metadata. Retrieval has four stages: metadata-aware candidates, hybrid search, graph expansion over cross-references, then cross-encoder reranking."

**Emphasise:** ISO docs are **tables + references**; the graph lets us follow references (multi-hop).

**Point at:** the four retrieval stages.

**Likely questions**
- **"What is the cross-reference graph?"** → Nodes are chunks; edges are "see Clause X" / "refer to Annex A" references, plus hierarchy edges (section → parent section).
- **"How does graph expansion work?"** → After the first retrieval, we add neighbours of the retrieved nodes (BFS, depth 1) to pull in referenced clauses.
- **"Why rerank last?"** → Reranking is expensive, so it runs only on the small expanded candidate set.
- **"What's the cost?"** → Latency: graph expansion + cross-encoder make retrieval ~4 s vs ~0.1 s for the baseline.

---

## Slide 7 — Retrieval quality (graph)

**Say:** "This chart compares how well each pairing finds the right passage. Pairing 3 is highest on every bar."

**Emphasise:** all four metrics, P3 first.

**Point at:** the grouped bar — Recall@5, nDCG@5, MRR.

**Likely questions**
- **"What do these metrics mean?"** → Recall@5 = found a relevant chunk at all; nDCG@5 = ranked near the top; MRR = first relevant hit is high; precision@5 = low noise.
- **"Why is P3's recall exactly 1.0?"** → Over 32 queries, at least one gold-relevant chunk was in the top-5 every time — the graph + rerank recovers the right clause.
- **"Is 1.0 too good?"** → It's 32 queries and answer-term relevance; with more queries we'd expect it to drop slightly.

---

## Slide 8 — Retrieval quality (what & result)

**Say:** "I compare four things: Recall@5 — did we find a relevant chunk; nDCG@5 — is it ranked near the top; MRR — how high is the first hit; and Precision@5 — how much of the top-5 is noise. Pairing 3 wins all four, with Recall@5 = 1.00. Pairing 1 ranks well but is noisier; Pairing 2 is precise but misses more."

**Emphasise:** **P3 Recall@5 = 1.00** is the headline.

**Point at:** the numbers in the table.

**Likely questions**
- **"Why is P2's recall lower than P1's?"** → Coarser section boundaries and RRF fusion trade some recall for precision; P1's small child chunks match more queries but with more noise.
- **"Precision looks low (0.54–0.72) — is that bad?"** → Top-5 deliberately includes context; lower precision is expected when returning several chunks. Recall and ranking matter more here.
- **"Why report both answer-term and section relevance?"** → Answer-term works for all pairings; section matching is stricter and only meaningful when chunks carry section numbers (P2/P3).

---

## Slide 9 — Generation quality (graph)

**Say:** "This compares the quality of the final answer. Pairing 3 is best on relevance, correctness and citations; Pairing 1 has the best faithfulness but zero citation accuracy."

**Emphasise:** P3 leads on three of four; P1's citation zero.

**Point at:** the grouped bar — Faithfulness, Relevance, Correctness, Citation accuracy.

**Likely questions**
- **"Who judges these?"** → A custom LLM judge at temperature 0 with fixed prompts; citation accuracy is rule-based, not judged.
- **"Why not RAGAS?"** → RAGAS returns `nan` with the small local judge; the custom judge is the reliable default.
- **"What's a good faithfulness?"** → Higher is better; ~0.5–0.6 is modest and mainly reflects the small 3B model.

---

## Slide 10 — Generation quality (what & result)

**Say:** "I compare faithfulness — is the answer grounded; relevance — does it answer the question; correctness — does it match the gold answer; and citation accuracy — do the cited sources support the claims. Pairing 3 wins relevance (0.73), correctness (0.73) and citations (0.95). Pairing 2 is a close second. Pairing 1 has the best faithfulness (0.61) but its citation accuracy is zero, because its recursive chunks carry no section numbers — a structural gap, not a model failure. Faithfulness is close across all three, which tells us the 3B model is the limit."

**Emphasise:** the **P1 citation = 0 explanation** and that **P3 and P2 both produce verifiable answers**.

**Point at:** the table, then the P1 citation row.

**Likely questions**
- **"Why is P1's citation accuracy exactly 0?"** → The metric checks whether cited section numbers were retrieved; P1's chunks have no section metadata, so citations can't be verified — structural, not a wrong answer.
- **"Why is faithfulness only ~0.5–0.6?"** → The local 3B LLM and judge; a stronger judge/LLM would raise it. It's model-limited, not retrieval-limited.
- **"Why is P3's relevance the highest?"** → Better retrieval → the context actually contains the answer, so the LLM stays on-topic.
- **"Is the difference between P3 (0.73) and P2 (0.69) significant?"** → It's small over 10 judged queries; treat it as a weak signal, not a decisive gap.

---

## Slide 11 — Quality vs latency & cost (graph)

**Say:** "This bubble chart trades quality against cost. X is end-to-end latency, Y is citation accuracy, and bubble size is index size. Pairing 1 is fastest; Pairing 3 has the best quality; Pairing 2's latency was variable."

**Emphasise:** the axes and what "ideal" looks like (top-left, small bubble).

**Point at:** the bubble chart, then the efficiency table below it.

**Likely questions**
- **"How is latency measured?"** → Wall-clock per query: retrieval, then generation, timed separately and summed for E2E.
- **"Why is index size similar across pairings?"** → Same corpus, similar chunk counts; differences are modest (27–55 MiB).
- **"Why fewer tokens for P3 if it does more?"** → Graph + rerank return a tighter, more relevant context set, so the prompt is smaller.

---

## Slide 12 — Quality vs latency & cost (what & result)

**Say:** "The trade-off: Pairing 1 is fastest — 108 ms retrieval, 2.5 s end-to-end. Pairing 3 has the best quality but is about 2.7× slower, and the cost is almost entirely retrieval, 4.1 s versus 108 ms, from graph expansion and reranking. Pairing 2's latency was noisy — 7.6 s here versus 2.3 s in an earlier run — so I treat single-run latency as indicative."

**Emphasise:** this is the **key engineering decision** — quality vs responsiveness — and that **latency is noisy on this hardware**.

**Point at:** the E2E and retrieval rows.

**Likely questions**
- **"Why did P2's latency change so much between runs?"** → Single runs on a shared WSL/GPU machine; background load and Ollama scheduling vary. It's a measurement caveat, not a pipeline change.
- **"Can P3 be made faster?"** → Yes — cap graph fan-out, batch/cache the reranker, and tune candidate counts. Target < 1 s retrieval.
- **"Which would you deploy?"** → Pairing 2 as a balanced default, Pairing 3 for accuracy-critical/offline, Pairing 1 when latency dominates.
- **"Is the latency difference worth it?"** → Only if the query needs the extra recall/citations; for simple lookups P1 is fine.

---

## Slide 13 — Chunking characteristics (graph)

**Say:** "This shows the chunk corpus each pairing produces — chunk counts and average chunk size. The scale is similar across pairings."

**Emphasise:** similar scale → differences are structural, not size-driven.

**Point at:** the *Chunks per pairing* and *Avg chunk size* bars.

**Likely questions**
- **"Why is average chunk size similar?"** → All three target the same nominal size; the difference is *where* the boundaries fall.
- **"Does chunk count matter?"** → More chunks = larger index and more embedding cost; fewer = coarser retrieval.

---

## Slide 14 — Chunking characteristics (what & result)

**Say:** "I compare chunk count, average size, relevance density — how much of a chunk is actually useful — and boundary coherence — whether chunks are self-contained. The scale is similar, so differences come from structure. Relevance density rises with structure: 0.14, 0.22, 0.26. Boundary coherence is low everywhere, 0.28 to 0.35 — the clear weak spot."

**Emphasise:** **relevance density rises with structure** and **boundary coherence is the top chunking improvement target**.

**Point at:** the two density/coherence rows.

**Likely questions**
- **"What is relevance density?"** → An LLM-judge score of how much of a retrieved chunk is relevant to the query, versus padding.
- **"What is boundary coherence?"** → Whether a chunk is self-contained (1) or fragmented mid-sentence/table/clause (0).
- **"Why is boundary coherence low?"** → Long clauses and tables still get split; this is the highest-value chunking fix (sentence-aware packing).
- **"Why does P3 have the densest chunks?"** → Table-first chunks plus clause boundaries pack more genuine content and less filler.

---

## Slide 15 — Recommendation

**Say:** "So: for accuracy-critical or offline use, Pairing 3 — perfect recall and 0.95 citation accuracy. For interactive use, Pairing 1 — 108 ms retrieval, 2.5 s end-to-end. For the best balance, Pairing 2 — strong precision, 0.93 citations, and a small index. All three run fully locally."

**Emphasise:** a clear, evidence-backed recommendation per use case.

**Point at:** each bullet.

**Likely questions**
- **"Which would you ship as the default?"** → Pairing 2 — it balances precision, citations, and index size, and avoids P3's latency.
- **"What if the corpus grows 10×?"** → BM25 and the graph scale with corpus size; P3's retrieval cost would grow most — cap fan-out and cache aggressively.
- **"Could you route between pairings?"** → Yes — a lightweight router (by query type or latency budget) is a natural next step.

---

## Slide 16 — What can be improved (achievable)

**Say:** "Six concrete next steps, all achievable on this hardware: add chat memory (now implemented); fix Pairing 1's citations by attaching section metadata; speed up Pairing 3 by capping graph fan-out and batching the reranker; try a larger embedding with the reranker on CPU; raise boundary coherence with sentence-aware packing; and make the evaluation more robust by judging all 32 queries with multiple runs."

**Emphasise:** each is **scoped and realistic**, not hand-wavy.

**Likely questions**
- **"What's the single biggest win?"** → Pairing 3 latency (4.1 s → < 1 s) and boundary coherence — both directly improve usability and quality.
- **"What's the biggest limitation?"** → The small 3B LLM caps faithfulness; and the evaluation is a single run over a modest gold set.
- **"Would fine-tuning help?"** → Possibly for style/domain, but it wouldn't fix grounding or citations; retrieval is the higher-leverage investment.

---

## Appendix — Graph attachment map

| Slide | Graph | Where in the UI |
|---|---|---|
| 7 | Retrieval quality grouped bar (+ category heatmap) | Evaluation & Comparison → "Retrieval quality" |
| 9 | Generation quality grouped bar | Evaluation & Comparison → "Generation quality" |
| 11 | Quality vs Latency bubble + Efficiency table | Evaluation & Comparison → "Quality vs latency" / "Efficiency" |
| 13 | Chunks per pairing + Avg chunk size bars | Evaluation & Comparison → "Chunking characteristics" |
| Extra | Chunk-size histogram, content-type, coverage | 🧩 Chunks tab |

---

## General Q&A bank (likely mentor questions)

### Problem & motivation
- **"Why is this a research problem?"** → It studies how document structure and retrieval strategy interact for RAG on standards, with a reproducible evaluation protocol.
- **"What's the hypothesis?"** → Structure-aware chunking + graph-expanded retrieval improves retrieval and citation accuracy, at a latency cost.

### Models & stack
- **"Why a local stack?"** → Privacy (specifications may be sensitive), no API cost, reproducibility.
- **"Why not GPT-4?"** → We support OpenAI as an option, but the comparison holds models constant so chunking/retrieval is isolated.

### Design choices
- **"Why three pairings?"** → To isolate the contribution of structure (P2) and references/tables (P3) against a baseline (P1), increasing sophistication monotonically.
- **"Why a graph instead of just metadata filters?"** → Cross-references are relational; a graph supports multi-hop expansion that a flat filter can't.

### Methodology
- **"How did you build the gold set?"** → 32 seed queries in `eval/gold_seeds.json`, expanded to gold sections/answers by `eval/build_gold_set.py`; same set for all pairings.
- **"How is relevance judged?"** → Answer-term matching (all pairings) + section matching (P2/P3); uncomputable metrics are **flagged, never dropped**.
- **"Who judges generation?"** → A custom LLM judge (`eval/judge.py`) at temperature 0, identical prompts across pairings.
- **"Is it statistically robust?"** → It's a **single run** (`--runs 1`); re-run with `--runs 3` for mean ± std. Also, generation metrics are averaged over the **judged subset** (see `judge_coverage`).

### Results
- **"Why does P3 win retrieval?"** → Table-first chunks, graph expansion over cross-references, and cross-encoder reranking.
- **"Why is P1's citation accuracy 0?"** → No section metadata on recursive chunks — structural.
- **"Why is faithfulness low across the board?"** → The small 3B LLM/judge is the limiting factor.

### Limitations & threats to validity
- **"What could bias the results?"** → LLM-judge bias, a small gold set, single runs, and noisy latency on shared hardware.
- **"Why did latency vary between runs?"** → Single runs on WSL/GPU under varying load; treat as indicative.
- **"Why is indexing time ~11–13 s?"** → The rebuild reused the persisted Chroma index (warm build); delete `chroma_db/` and `data/indexes/` for a cold measurement.

### Future work & scalability
- **"How would this scale?"** → Multi-PDF loader and per-document metadata already support it; cap graph fan-out and cache reranking for large corpora.
- **"Could two-stage routing help?"** → Yes — route easy lookups to P1 and hard/multi-hop queries to P3.
- **"Would larger embeddings help?"** → Likely yes (Qwen3-4B / BGE-M3), with the reranker on CPU given 8 GB VRAM.

---

## Tough-question tactics

- **"I don't know" is fine** → "I didn't measure that; I'd test it by …". Never invent a number.
- **Aim at the mechanism** → If unsure about a number, explain *why* the result should go that way.
- **Separate signal from noise** → Small gaps (e.g. 0.73 vs 0.69 over 10 judged queries) are weak; say so.
- **Own the limitations** → Single run, small judge subset, noisy latency — name them before the mentor does.
- **Bring it back to the decision** → "Under a latency budget, I'd pick P1; when accuracy matters, P3."

---

## Closing line

> "The takeaway: **structure-aware chunking plus graph-expanded retrieval gives the best retrieval and citation accuracy**, at a latency cost — and the evaluation framework lets us quantify that trade-off rather than guess."
