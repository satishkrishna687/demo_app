## Chapter 7: Conclusion and Future Work

### 7.1 Summary of the Project

This project developed a fully local retrieval-augmented question-answering system over the three ISO 15118 part documents (Parts 1, 2 and 3) [12], and used it to answer an empirical question: which combination of chunking strategy and retrieval mechanism yields the best clause-accurate retrieval and the best grounded generation over a standards corpus? The system parses the PDFs with Docling [10], recovers a clause-level outline from the parser's structured labels, chunks the outline under three strategies, indexes the chunks with a local instruction-aware embedder and a BM25 lexical index, retrieves clauses with a different mechanism per pairing, and generates cited answers with a local language model. A shared evaluation harness scores all three pairings on a common 32-query gold set with deterministic retrieval metrics, an LLM-as-judge for answer quality, and a deterministic citation-accuracy metric.

The project was organised as a controlled comparison of three chunking × retrieval pairings over a shared ingestion substrate. Pairing 1 (recursive chunking with parent-child retrieval) served as the baseline. Pairing 2 added section-aware chunking and hybrid dense/BM25 retrieval fused by reciprocal rank fusion. Pairing 3 added table-first chunking, a deterministic cross-reference graph with bounded expansion, metadata-filtered retrieval, and cross-encoder reranking. Because every pairing consumed the same parsed representation, a measured difference between pairings was attributable to chunking and retrieval rather than to parsing.

The measured results answered the research question. Pairing 3 achieved the best retrieval quality (Recall@5 = 1.0, nDCG@5 = 0.9401, MRR = 0.9307), while Pairing 2 achieved the best generation quality (faithfulness 0.5690, answer correctness 0.7010, citation accuracy 0.9600) and the lowest end-to-end latency (2660.5 ms). The baseline was competitive on retrieval but could not cite clauses, because its chunks carry no clause identity. The most important finding is that retrieval quality and generation quality are separable: the pairing with the best retrieval was not the pairing with the best generation, which demonstrates that the two stages must be measured independently.

### 7.2 Contributions

The project makes the following contributions, each of which is both a design contribution and, where measured, an empirical one.

1. **Structure-aware chunking derived from parser labels.** A chunking approach in which section boundaries, annex handling, and table atomicity are derived from Docling's structured labels [10] rather than fixed token windows, with a documented strategy for oversized sections and atomic tables. This is implemented in [`ingestion/structure.py`](ingestion/structure.py:1) and [`ingestion/chunk_utils.py`](ingestion/chunk_utils.py:1), and its value is demonstrated by the baseline's inability to cite clauses.

2. **A three-pairing empirical comparison.** Three complete chunking × retrieval pipelines — recursive/parent-child (baseline), section-aware/hybrid, and table-first/graph/reranked — implemented over a shared ingestion substrate and compared on a common gold set. The comparison provides measured evidence on the chunking–retrieval interaction for standards text, and it shows that the best retrieval and the best generation are achieved by different pairings.

3. **Cross-reference graph retrieval expansion.** A deterministic cross-reference graph over clause references, built with NetworkX in [`graph/builder.py`](graph/builder.py:1), with a retrieval-expansion mechanism that pulls in clauses reached through the graph — an explicit, auditable alternative to relying on chunk co-occurrence for clause dependencies. Its contribution is visible in Pairing 3's Recall@5 of 1.0.

4. **A deterministic citation-accuracy metric.** An evaluation metric that checks, without a judge model, whether the clause citations in a generated answer correspond to the clauses of the retrieved evidence, implemented in [`eval/judge.py`](eval/judge.py:1). It complements LLM-as-judge faithfulness with a fully reproducible check, and it is what exposes the baseline's citation failure.

5. **A reproducible evaluation harness.** A single-command harness that scores all pairings on the same gold set with the same judge configuration, records uncomputable metrics as flagged rather than omitted, and reports measured results with their run count. The harness is implemented in the `eval/` package and is the source of every number reported in Chapter 6.

### 7.3 Limitations

The project has the following limitations, which are stated so that the contributions are not overstated.

**Single-run measurement.** Each pairing was evaluated with a single run, so the reported values are point measurements without variance. The judge runs at temperature 0 to reduce variance, but run-to-run variation has not been quantified.

**Differing judge coverage.** The generation metrics for Pairing 2 were computed over a smaller judged subset (10 queries) than those for Pairings 1 and 3 (32 queries), which weakens the direct comparability of the faithfulness, answer-relevance and answer-correctness figures.

**Outlier-dominated retrieval latency.** The mean retrieval latency for Pairings 1 and 3 is distorted by a few very slow queries, so the mean is not a reliable summary of typical retrieval cost.

**Answer-term relevance as a proxy.** The deterministic retrieval metrics use answer-term matching, which is a proxy for true relevance and may count a non-governing chunk as relevant or miss a governing chunk that phrases the answer differently.

**Small, fixed corpus and gold set.** The corpus is three documents and the gold set is 32 hand-built queries, so the results characterise this corpus rather than standards text in general.

**Deferred extensions.** The NLI-based grounding evaluation and adaptive-k context selection named in the original proposal are not implemented; they are identified as future work below.

**Parsing dependence.** The structure recovery depends on Docling's labels, and a parsing failure would propagate into chunking and retrieval.

### 7.4 Future Work

The limitations above motivate the following future work.

**Multi-run variance analysis.** Re-run each pairing with `--runs 3` or more to report mean ± standard deviation for the judged metrics, so that the differences between pairings can be assessed for statistical significance rather than read as point values.

**NLI-based grounding evaluation.** Add a sentence-level Natural Language Inference check, using a model such as `cross-encoder/nli-deberta-v3-base`, to classify each answer statement as entailed by, contradicted by, or unsupported by the retrieved evidence. This would complement the LLM-as-judge faithfulness score with a model-based entailment signal, and it would be reported as an estimate rather than as proof of correctness.

**Adaptive-k context selection.** Replace the fixed `TOP_K = 5` with a variable number of chunks selected from score gaps, bounded between 1 and 8, so that easy questions use a smaller context and difficult questions use a larger one. The candidate algorithm would require experimental validation against the fixed-k baseline.

**Corpus expansion.** Extend the corpus beyond the three ISO 15118 parts to related standards, to test whether the structure-aware chunking and cross-reference graph generalise to other clause-hierarchical, table-heavy documents.

**Incremental indexing.** Replace the current fingerprint-based full rebuild with an incremental update that re-indexes only the chunks that changed, so that a corpus update does not require a full re-embedding.

**Improved citation accuracy.** Investigate why Pairing 3's citation accuracy (0.9481) is slightly below Pairing 2's (0.9600) despite its better retrieval, and whether the table-first chunking can be adjusted to give the generator a cleaner context without losing retrieval quality.

**Independent adjudication of the gold set.** Have the gold relevance judgements independently reviewed, so that the retrieval metrics rest on adjudicated rather than single-author judgements.

### 7.5 Concluding Remarks

The project set out to determine how the pipeline between a PDF and a language model should be designed so that retrieval over an engineering standard is clause-accurate and generation is grounded. It answered this by implementing three complete pipelines over a shared ingestion substrate and measuring them on a common gold set. The measured results show that structure-aware chunking, hybrid retrieval, cross-reference graph expansion and cross-encoder reranking together produce the best clause-accurate retrieval, that section-aware chunking with hybrid retrieval produces the best-grounded and best-cited answers at the lowest latency, and that a conventional recursive pipeline, while competitive on retrieval, cannot cite the governing clause at all. The most durable lesson is methodological: retrieval quality and generation quality are separable, and a system that measures them together cannot tell which stage a change has improved. The project's principal contribution is therefore not a single winning configuration but a reproducible, evidence-disciplined pipeline and evaluation harness that makes the effect of each design decision empirically attributable.

### 7.6 Deviations from the Originally Proposed Design

The report was generated against the actual repository, and in several places the implemented system differs from the design assumed in the original project brief. Table 7.1 records each deviation and its rationale, so that the report is transparent about where the implementation departs from the proposal.

**Table 7.1: Deviations from the originally proposed design and their rationale**

| Originally proposed | Implemented in the repository | Rationale |
|---|---|---|
| `bge-small-en-v1.5` embeddings | `Qwen/Qwen3-Embedding-0.6B` | The repository uses an instruction-aware embedder suited to the unlabelled ISO corpus; the model card is cited for its input formatting. |
| NLI evaluator (`cross-encoder/nli-deberta-v3-base`) | LLM-as-judge ([`eval/judge.py`](eval/judge.py:1)) plus deterministic citation accuracy | The custom judge is the implemented evaluation; NLI-based grounding is deferred to future work. |
| Adaptive-k context selection (1–8) | Fixed `TOP_K = 5` with `RERANK_TOP_N = 20` and graph expansion | The implemented mechanism is described as it is; adaptive-k is identified as future work. |
| Baseline = fixed k = 5 without NLI | Baseline = Pairing 1 (recursive + parent-child) | Pairing 1 is the implemented baseline against which the structure-aware pairings are measured. |
| No experimental results exist | Results exist (single run, 32 queries) | The measured results are presented as real but single-run, with variance pending. |
| Source file `stage1_1_ppt.md` | Does not exist | The report uses [`docs/STAGE1_FIRST_REVIEW_PPT.md`](docs/STAGE1_FIRST_REVIEW_PPT.md:1) and the repository as primary sources. |

---

## References

[1] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[2] V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," in Proc. Conf. Empirical Methods Natural Language Processing (EMNLP), 2020, pp. 6769–6781.

[3] S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333–389, 2009.

[4] G. V. Cormack, C. L. A. Clarke, and S. Buettcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in Proc. Int. ACM SIGIR Conf. Research and Development in Information Retrieval (SIGIR), 2009, pp. 758–759.

[5] O. Khattab et al., "ColBERT: Efficient and effective passage search via contextualized late interaction over BERT," in Proc. Int. ACM SIGIR Conf. Research and Development in Information Retrieval (SIGIR), 2020, pp. 391–400.

[6] W. Gao et al., "Precise zero-shot dense retrieval without relevance labels," in Proc. Conf. Empirical Methods Natural Language Processing (EMNLP), 2023, pp. 7215–7228.

[7] J. Chen et al., "BAAI bge-reranker-v2-m3: A multilingual cross-encoder model for text ranking," BAAI model card, 2024.

[8] S. Es et al., "RAGAS: Automated evaluation of retrieval augmented generation," in Proc. Demonstrations Eur. Conf. Adv. Methods Natural Language Processing (EACL), 2024, pp. 809–816.

[9] D. Edge et al., "From local to global: A graph RAG approach to query-focused summarization," arXiv preprint arXiv:2404.16130, 2024.

[10] M. Nigheti et al., "Docling technical report," arXiv preprint, 2024.

[11] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-Transformer networks," in Proc. Conf. Empirical Methods Natural Language Processing (EMNLP), 2019, pp. 3982–3992.

[12] International Organization for Standardization, ISO 15118-1, ISO 15118-2, and ISO 15118-3, Road vehicles — Vehicle to grid communication interface — Part 1/2/3, Geneva, Switzerland, 2024.
