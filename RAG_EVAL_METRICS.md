# RAG Evaluation Framework: Comparing Chunking × Retrieval Strategy Pairs

## Objective
Evaluate and compare 3 (chunking strategy, retrieval strategy) pairs for a RAG pipeline across retrieval quality, generation quality, chunking characteristics, and efficiency. Produce a structured comparison table and a written analysis of tradeoffs.

## Inputs You Will Be Given
- Description of each of the 3 pairs (chunking method + retrieval method)
- A fixed evaluation question set with gold answers and/or gold relevant-passage labels
- Retrieved chunks and generated answers for each pair, per question
- (Optional) timing/cost logs per pair

## Task
For each pair, compute and report the metrics below. Use the same evaluation question set across all 3 pairs for fair comparison. Where a metric requires judgment (not exact match), use consistent scoring criteria across all pairs — do not adjust standards between pairs.

---

## 1. Retrieval-Level Metrics
*(Requires gold relevant-chunk labels per query)*

| Metric | Definition | How to Compute |
|---|---|---|
| Recall@k | Fraction of queries where at least one gold-relevant chunk is in top-k retrieved | For each query, check if any gold chunk ID appears in top-k retrieved chunk IDs |
| Precision@k | Fraction of top-k retrieved chunks that are actually relevant | relevant_retrieved / k, averaged over queries |
| MRR | Mean Reciprocal Rank of first relevant chunk | 1/rank of first relevant hit, averaged over queries |
| nDCG@k | Ranking quality with graded relevance (if relevance is graded, not binary) | Standard nDCG formula |

Report k at multiple values (e.g., k=1, 3, 5, 10) if possible.

## 2. Chunking-Specific Metrics

| Metric | Definition | How to Compute |
|---|---|---|
| Avg chunk size | Mean chunk length (tokens or chars) | Simple average over corpus |
| Chunk count | Total number of chunks produced from corpus | Count |
| Relevance density | How much of a retrieved chunk is actually relevant vs. padding | LLM-judge: rate 0-1 per retrieved chunk given the query |
| Boundary coherence | Whether chunks split mid-thought / mid-table / mid-clause | LLM-judge: rate each chunk as self-contained (1) or fragmented (0), average |

## 3. End-to-End Generation Metrics
*(Requires the generated answer per query, per pair)*

| Metric | Definition | How to Compute |
|---|---|---|
| Faithfulness / groundedness | Does the answer only state things supported by retrieved context? | LLM-judge: score 0-1, check each claim in answer against retrieved chunks |
| Answer relevance | Does the answer address the actual question asked? | LLM-judge: score 0-1 |
| Answer correctness | Does the answer match the gold answer? | Exact match / F1 for short answers; LLM-judge score for open-ended |
| Citation accuracy | If system cites sources, are citations correct? | Check cited chunk ID actually supports the cited claim |

## 4. Efficiency / Cost Metrics

| Metric | Definition |
|---|---|
| Indexing time | Time to chunk + embed the full corpus |
| Query latency | Time from query to final answer (retrieval + generation) |
| Index storage size | Disk/memory size of the resulting index |
| Tokens / LLM calls per query | Cost proxy, especially relevant if retrieval is multi-hop or agentic |

---

## Output Format

Produce:

1. **Summary table** — one row per pair, one column per metric (all four categories above).
2. **Retrieval quality ranking** — which pair wins on Recall@k / MRR / nDCG, and by how much.
3. **Generation quality ranking** — which pair wins on faithfulness / correctness.
4. **Efficiency tradeoff note** — does the best-quality pair also cost the most in latency/tokens?
5. **Overall recommendation** — 2-3 sentences on which pair is best suited for which use case (e.g., "Pair B has the highest recall but 3x the latency of Pair A — recommended only if quality is prioritized over responsiveness").

## Rules
- Use identical scoring criteria across all 3 pairs — never relax judging standards for one pair.
- If an LLM-judge is used for any metric, use the same judge model, prompt, and temperature=0 for all pairs.
- Report variance (e.g., across multiple runs or question subsets) where possible, not just point estimates.
- Flag any metric that could not be computed due to missing data, rather than omitting it silently.
