# Stage 1 Literature Survey — RAG over ISO 15118

**Project:** Retrieval-Augmented Generation over ISO 15118 EV Charging Communication Standards
**Document type:** Academic Stage 1 (first review) deliverable — literature survey
**Related deliverables:** `docs/STAGE1_FIRST_REVIEW_PPT.md`, `docs/STAGE1_PROPOSED_SYSTEM.md`, `docs/STAGE1_REFERENCES.md`

---

## Table of contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Theme-by-theme survey](#2-theme-by-theme-survey)
   - [2.1 Retrieval-augmented generation foundations](#21-retrieval-augmented-generation-foundations)
   - [2.2 Retrieval models and sparse/dense fusion](#22-retrieval-models-and-sparsedense-fusion)
   - [2.3 Document structure and chunking for domain corpora](#23-document-structure-and-chunking-for-domain-corpora)
   - [2.4 Knowledge-graph retrieval](#24-knowledge-graph-retrieval)
   - [2.5 Reranking](#25-reranking)
   - [2.6 Evaluation of RAG systems](#26-evaluation-of-rag-systems)
   - [2.7 The ISO 15118 standard as corpus](#27-the-iso-15118-standard-as-corpus)
   - [2.8 Summary table](#28-summary-table)
3. [Comparative discussion](#3-comparative-discussion)
4. [Identified research gaps](#4-identified-research-gaps)
5. [How this project addresses the gaps](#5-how-this-project-addresses-the-gaps)
6. [Conclusion](#6-conclusion)
7. [References](#7-references)
   - [7.1 Software and framework sources](#71-software-and-framework-sources)

---

## 1. Purpose and scope

This survey reviews the prior work that motivates the design of a retrieval-augmented generation
(RAG) system over the three parts of the ISO 15118 vehicle-to-grid communication interface
standard. It is not a general review of information retrieval; it is restricted to the works that
bear directly on the decisions this project has to make — how a layout-heavy, clause-numbered,
table-rich technical standard is parsed and chunked, how candidate clauses are retrieved, how
the candidate list is reordered, and how the answering quality of the resulting system is measured.

The twelve numbered references are grouped into seven themes:

| Theme | References | Why it matters here |
|---|---|---|
| (a) RAG foundations | [1] | Defines the retrieve-then-generate pattern the whole project implements. |
| (b) Retrieval models and sparse/dense fusion | [2], [3], [4], [11] | Supply the dense retriever, the lexical retriever and the fusion rule used in hybrid search. |
| (c) Document structure and chunking for domain corpora | [10] | Provides the layout-aware parsing on which all structure-aware chunking depends. |
| (d) Knowledge-graph retrieval | [9] | Motivates treating cross-references between clauses as graph edges. |
| (e) Reranking | [5], [6], [7] | Justify a second, finer-grained ranking pass over the retrieved candidates. |
| (f) Evaluation of RAG systems | [8] | Motivates an LLM-judge component alongside deterministic metrics. |
| (g) The ISO 15118 standard itself | [12] | Defines the target corpus, its clause hierarchy, annexes and normative tables. |

Each reference is treated under its theme and discussed in a consistent form: *citation*,
*research focus*, *methodology*, *key findings*, and *limitations / relevance to this project*.
Findings are reported qualitatively. No experimental numbers from the surveyed papers are
reproduced, since the survey is concerned with design rationale rather than with re-reporting of
results. The only quantities that appear in this document are this project's own design
parameters, taken from `config.py`.

---

## 2. Theme-by-theme survey

### 2.1 Retrieval-augmented generation foundations

#### [1] Lewis et al. — Retrieval-Augmented Generation for knowledge-intensive NLP tasks

- **Citation:** [1] P. Lewis et al., "Retrieval-augmented generation for knowledge-intensive NLP tasks," in Advances in Neural Information Processing Systems (NeurIPS), 2020.
- **Research focus:** The paper asks whether a sequence-to-sequence model can be made to use a retrieved, non-parametric corpus at generation time instead of relying only on the parameters learned during training.
- **Methodology:** A parametric encoder–decoder is augmented with a differentiable non-parametric retriever over a document index. The retriever produces a latent representation of the selected documents, and this representation is fed to the decoder alongside the token-level history, so the two components can be trained jointly end-to-end.
- **Key findings:** The authors report substantial gains over parametric-only baselines on knowledge-intensive tasks, and show that the retrieved document representation carries most of the factual burden of the answer. Retrieval quality is identified as a major driver of the final answer quality.
- **Limitations / relevance to this project:** The original setup assumes a large web-scale corpus and a large, heavily trained generator. It does not address document structure, chunking boundaries, tables, or clause-level provenance, and it gives no mechanism for citing the source of a statement. This project keeps the retrieve-then-generate idea but replaces the learned joint retriever with an explicit, inspectable hybrid pipeline, and adds a citation requirement that the original formulation does not have.

### 2.2 Retrieval models and sparse/dense fusion

#### [2] Karpukhin et al. — Dense Passage Retrieval for Open-Domain Question Answering

- **Citation:** [2] V. Karpukhin et al., "Dense passage retrieval for open-domain question answering," in Proc. Conf. Empirical Methods Natural Language Processing (EMNLP), 2020, pp. 6769–6781.
- **Research focus:** The work establishes dense bi-encoder retrieval of passages as a viable replacement for conventional sparse retrieval in open-domain question answering.
- **Methodology:** Question and passage are encoded independently into a single dense vector space using BERT encoders, with passages pre-encoded so that retrieval reduces to a nearest-neighbour vector search. The training objective uses question–passage pairs with in-batch negatives, optionally augmented by span-relevance supervision derived from Wikipedia hyperlinks.
- **Key findings:** Dense retrieval is reported to substantially outperform lexical baselines on open-domain QA when enough training data is available. The learned vector space captures semantic similarity that exact term matching misses.
- **Limitations / relevance to this project:** The training data is Wikipedia-scale and out-of-domain for technical standards, and the passage segments are fixed, structure-agnostic spans. The paper does not study hybrid fusion, reranking, or citation of the retrieved span. Here the dense component is used with a modern instruction-aware sentence embedder rather than a task-fine-tuned QA encoder, and is fused with a lexical retriever rather than used alone.

#### [3] Robertson and Zaragoza — The Probabilistic Relevance Framework: BM25 and Beyond

- **Citation:** [3] S. Robertson and H. Zaragoza, "The probabilistic relevance framework: BM25 and beyond," Foundations and Trends in Information Retrieval, vol. 3, no. 4, pp. 333–389, 2009.
- **Research focus:** The article gives a unified probabilistic account of lexical ranking functions, explaining why term-frequency saturation and inverse document frequency weighting work and how they generalise.
- **Methodology:** Relevance is modelled as a probability derived from a document–term generation process; classical TF-IDF is then re-derived as a special case, and BM25 emerges with document-length normalisation and term-frequency saturation built into the model. The framework is also used to motivate later variants that add semantic or latent term structure.
- **Key findings:** Lexical ranking is presented as a principled, robust and cheap component rather than a heuristic to be discarded. Saturation and length normalisation are shown to matter in practice, and the framework explains why lexical methods remain strong when the query vocabulary matches the document vocabulary.
- **Limitations / relevance to this project:** The framework assumes pre-segmented documents and treats each segment independently; it is blind to document hierarchy and to the fact that a "document" here is a numbered clause with a caption. It also has no notion of semantic similarity, so it alone cannot answer a paraphrased question. It therefore serves as the lexical half of this project's hybrid retriever, where exact clause numbers, signal names and parameter symbols are decisive.

#### [4] Cormack, Clarke and Buettcher — Reciprocal Rank Fusion

- **Citation:** [4] G. V. Cormack, C. L. A. Clarke, and S. Buettcher, "Reciprocal rank fusion outperforms Condorcet and individual rank learning methods," in Proc. Int. ACM SIGIR Conf. Research and Development in Information Retrieval (SIGIR), 2009, pp. 758–759.
- **Research focus:** The paper asks how several ranked result lists can be combined into one ranking when the individual scores are not comparable or calibratable.
- **Methodology:** Each document's fused score is the sum over rankers of a reciprocal function of its rank in that ranker's list, with a damping constant controlling the influence of low ranks. Because the combination operates purely on positions, it needs no score normalisation, no parameter estimation from data, and no training.
- **Key findings:** The authors report reciprocal rank fusion outperforming both Condorcet fusion and trained rank-learning methods across a range of document collections. Its main virtue is practical: it is a strong, parameter-light default when heterogeneous rankers are combined.
- **Limitations / relevance to this project:** The evaluation is over web document collections and considers ranking fusion in general, not the composition of a final context window for a generator. It says nothing about how much of the fused list a language model can use, and it offers no way to record *why* a chunk was retrieved. This project uses reciprocal rank fusion as the hybrid combination rule for dense and sparse retrievers, and follows the spirit of the paper in preferring a robust default over a tuned combiner.

#### [11] Reimers and Gurevych — Sentence-BERT

- **Citation:** [11] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-Transformer networks," in Proc. Conf. Empirical Methods Natural Language Processing (EMNLP), 2019, pp. 3982–3992.
- **Research focus:** The paper addresses the cost of computing BERT sentence embeddings on-the-fly for every candidate at query time.
- **Methodology:** The original cross-encoder setup, which feeds a sentence pair through the full transformer for each comparison, is replaced by a siamese architecture with shared weights plus a pooling and projection layer, so that sentences are embedded independently and compared by cosine similarity or dot product. Sentence embeddings are then fine-tuned with a siamese or triplet objective over sentence-similarity data.
- **Key findings:** Pre-computed embeddings make large-scale semantic search practical, and the resulting embeddings transfer well to semantic textual similarity and clustering tasks. This is what makes a local, offline vector index over a small corpus realistic.
- **Limitations / relevance to this project:** The study targets sentence- or passage-length English text and does not address long structured documents. In particular, a single embedding over an entire clause containing a normative table blurs the table's row and column structure, which is one motivation for treating tables as separate chunks and flattening tables before encoding.

### 2.3 Document structure and chunking for domain corpora

#### [10] Nigheti et al. — Docling technical report

- **Citation:** [10] M. Nigheti et al., "Docling technical report," arXiv preprint, 2024.
- **Research focus:** The report describes a document conversion pipeline designed to produce machine-readable, layout-aware output from PDFs while preserving reading order and document hierarchy.
- **Methodology:** A layout analysis model detects page elements and assigns labels to them; a structure model then orders those elements and predicts their hierarchical relationships. Tables are recognised as structured objects with rows, cells and headers rather than as flat text, and the pipeline offers several output formats so downstream consumers can choose the level of structure they need.
- **Key findings:** The authors report that layout-aware conversion is markedly more reliable than naive text extraction on complex, columned, table-bearing documents. Preserving reading order and table structure is presented as essential for any downstream task that requires document-level meaning.
- **Limitations / relevance to this project:** The report is a system description rather than an evaluation of retrieval or of chunking, and it does not propose chunking strategies for standards. Its output is a general structured document, whereas ISO 15118 has an additional, externally defined hierarchy of numbered clauses and non-numeric annexes. This project therefore derives a clause outline from the structured output in `ingestion/structure.py` rather than from the exported Markdown.

### 2.4 Knowledge-graph retrieval

#### [9] Edge et al. — From Local to Global: A Graph RAG Approach to Query-Focused Summarization

- **Citation:** [9] D. Edge et al., "From local to global: A graph RAG approach to query-focused summarization," arXiv preprint arXiv:2404.16130, 2024.
- **Research focus:** The paper studies the class of questions that ask about a whole corpus rather than about a single document, which conventional chunk-and-retrieve pipelines answer poorly.
- **Methodology:** An LLM is prompted to extract entities and the relations between them from each text unit; these are emitted as triples and assembled into a graph that is stored in a graph store. A local search retrieves entities and their immediate neighbourhoods for specific questions, while a community-detection step summarises clusters of related entities to answer global questions.
- **Key findings:** The authors report that graph-based retrieval produces more comprehensive answers for global, thematic questions than plain vector retrieval over the same corpus. The intermediate representation, an explicit entity-relation graph, is shown to be more useful than raw chunks for questions that require relating distant parts of a collection.
- **Limitations / relevance to this project:** The extracted graph is entity-centred and built by an LLM at ingestion time, which is expensive and depends on a generative model being available for the whole corpus. The paper targets community-level summarisation, not clause-level answers with verifiable citations, and it does not use the cross-references that a standard already states explicitly. This project instead builds a deterministic, citation-derived graph whose edges are the standard's own "see 4.2" and "Annex A" references, which is cheaper and auditable.

### 2.5 Reranking

#### [5] Khattab et al. — ColBERT

- **Citation:** [5] O. Khattab et al., "ColBERT: Efficient and effective passage search via contextualized late interaction over BERT," in Proc. Int. ACM SIGIR Conf. Research and Development in Information Retrieval (SIGIR), 2020, pp. 391–400.
- **Research focus:** The paper addresses the accuracy–efficiency trade-off between a full cross-encoder, which must score every query–document pair jointly, and a bi-encoder, which cannot represent fine-grained token-level matches.
- **Methodology:** Both query and document are encoded with BERT, but the document embedding is kept at token granularity and pre-computed. Retrieval is then a late-interaction step in which each query token is matched against all document tokens and the per-token similarities are aggregated, typically with a maximum over query tokens. This allows richer matching than pooled bi-encoder vectors at a fraction of the cost of full cross-encoding.
- **Key findings:** The authors report that ColBERT attains retrieval quality comparable to or better than a full cross-encoder while being much faster at query time, and that the token-level matching is the reason for the gain. The broader lesson is that the representation used for final ranking need not be the same as the one used for candidate generation.
- **Limitations / relevance to this project:** ColBERT's index is considerably larger than a vector index, which conflicts with this project's local-first, small-corpus constraint, and it still yields a bag-of-token interaction rather than a genuine pairwise joint encoding. It also gives no answer-level guarantees. This project therefore uses the cheaper, established alternative: a small cross-encoder applied as a second pass over only the top candidates.

#### [6] Gao et al. — Precise Zero-Shot Dense Retrieval without Relevance Labels

- **Citation:** [6] W. Gao et al., "Precise zero-shot dense retrieval without relevance labels," in Proc. Conf. Empirical Methods Natural Language Processing (EMNLP), 2023, pp. 7215–7228.
- **Research focus:** The paper tackles the situation in which a new retrieval domain has no relevance judgements, so a dense retriever cannot be fine-tuned for it and must work zero-shot.
- **Methodology:** A pre-trained embedding model is instruction-tuned using synthetic query–passage pairs generated by a large language model from unstructured passages, together with a consistency-filtering step that removes noisy synthetic supervision. The objective distils the LLM's notion of relevance into the bi-encoder, producing an embedder that follows retrieval instructions such as one for a query and an empty instruction for a document.
- **Key findings:** The authors report that the resulting instruction-following embedder improves zero-shot and few-shot dense retrieval over its base model, and that it also transfers to reranking. The practical takeaway is that a domain-specific corpus with no labels can still be served well by a general, instruction-aware embedder.
- **Limitations / relevance to this project:** The gains concern retrieval quality, not structure handling, and the instruction mechanism has no notion of a clause number or a table. In this project the embedder is used for the dense half of hybrid search, with the query-side instruction applied to questions and the document side left instruction-free; the stored chunk text is left untouched, with only the encoder input normalised.

#### [7] Chen et al. — BAAI bge-reranker-v2-m3

- **Citation:** [7] J. Chen et al., "BAAI bge-reranker-v2-m3: A multilingual cross-encoder model for text ranking," BAAI model card, 2024.
- **Research focus:** This artefact provides a ready-to-use cross-encoder for multilingual passage reranking, usable without task-specific training by the downstream project.
- **Methodology:** The model is a cross-encoder that takes a query–passage pair and outputs a relevance score; the pair is encoded jointly, so query and passage tokens attend to one another. It is trained over a mixture of ranking data covering multiple languages and long inputs, and is distributed with the settings needed to normalise and truncate long passages.
- **Key findings:** The model card reports strong ranking quality on standard retrieval benchmarks while remaining usable locally as a comparatively small downloadable checkpoint. Its long input tolerance is what makes it applicable to clause-sized and table-sized chunks.
- **Limitations / relevance to this project:** It is a reranker, not a retriever: it must be run over a candidate list and its cost grows with the list length, so it cannot replace the first-stage retrieval. Being an off-the-shelf model, it was not trained on normative technical text and its ordering on standards clauses is an open question that this project treats as an experimental variable rather than an assumption.

### 2.6 Evaluation of RAG systems

#### [8] Es et al. — RAGAS

- **Citation:** [8] S. Es et al., "RAGAS: Automated evaluation of retrieval augmented generation," in Proc. Demonstrations Eur. Conf. Adv. Methods Natural Language Processing (EACL), 2024, pp. 809–816.
- **Research focus:** The paper addresses the absence of standard, reference-free ways to measure whether a RAG system's answer is actually supported by the retrieved text.
- **Methodology:** A set of metrics is defined over a RAG pipeline's own inputs and outputs — the generated answer, the retrieved contexts and the original question — each scored by an LLM prompted with a fixed rubric. Faithfulness is judged by asking whether each claim in the answer is entailed by the supplied contexts, and answer relevance by asking whether the answer addresses the question.
- **Key findings:** The authors report that these reference-free LLM-judged metrics correlate with human judgement on faithfulness and relevance, and that they make RAG systems with different architectures comparable without building a ground-truth answer set.
- **Limitations / relevance to this project:** The metrics are non-deterministic, depend on judge model and prompt, and the paper does not address retrieval quality itself or clause-level provenance. Small local judges are known to be unreliable for such rubrics, so this project implements a custom, more constrained judge in `eval/judge.py` and records uncomputable scores as flagged `None` rather than dropping them, while keeping the deterministic retrieval metrics of `eval/metrics.py` as the primary, reproducible signal.

### 2.7 The ISO 15118 standard as corpus

#### [12] ISO 15118-1, ISO 15118-2 and ISO 15118-3

- **Citation:** [12] International Organization for Standardization, ISO 15118-1, ISO 15118-2, and ISO 15118-3, Road vehicles — Vehicle to grid communication interface — Part 1/2/3, Geneva, Switzerland, 2024.
- **Research focus:** This is the target corpus rather than a research contribution: it defines the physical and data link communication between an electric vehicle and an external charging station.
- **Methodology:** The specification is organised hierarchically. Part 1 covers the overall information and normative references; the later parts specify layers of the communication protocol, presented as numbered clauses and subclauses, accompanied by normative tables of data identifiers, timing parameters and value ranges, plus informative and normative annexes that are labelled with letters rather than numbers. Each requirement is written in a recognisable normative form and is normally cross-referenced to the clause or annex that defines the terms it uses.
- **Key findings:** For a retrieval system the important observations are structural rather than empirical: chunk boundaries can and should be aligned with clause and annex boundaries, tables carry much of the answerable detail and must not be fragmented, and the explicit cross-references form a genuine link structure that generic text pipelines discard.
- **Limitations / relevance to this project:** The corpus is small, fixed, largely non-public-domain and dominated by normative rather than narrative prose, so retrieval research published on web-scale or narrative corpora does not transfer automatically. There is no labelled query set for these documents, which is why this project builds its own gold question set and evaluation harness.

### 2.8 Summary table

| Ref | Year | Focus | Method | Main contribution | Limitation for this project |
|---|---|---|---|---|---|
| [1] | 2020 | RAG foundations | Jointly trained parametric generator + differentiable non-parametric retriever | Establishes retrieve-then-generate as a general architecture | No structure, table, chunking or citation handling; web-scale assumption |
| [2] | 2020 | Dense retrieval | BERT bi-encoder passages into a single vector space, nearest-neighbour search | Semantic passage retrieval that beats lexical baselines on open-domain QA | Structure-agnostic fixed segments; large labelled training domain; no fusion or citation |
| [3] | 2009 | Lexical retrieval | Probabilistic relevance model yielding BM25 with saturation and length normalisation | Principled, cheap lexical ranking | Treats segments independently; blind to hierarchy, annexes and tables |
| [4] | 2009 | Rank fusion | Sum of reciprocal rank functions over rankers' lists | Robust, training-free fusion of incomparable ranked lists | Evaluated on web collections; silent about how much fused context a generator can use |
| [5] | 2020 | Late interaction | Token-level query–document matching over pre-computed BERT embeddings | Rich matching near cross-encoder quality at lower query cost | Large index, conflicts with local-first budget; still not a true cross-encoder |
| [6] | 2023 | Zero-shot dense retrieval | LLM distillation and synthetic pairs to instruction-tune a bi-encoder | Usable dense retrieval for unlabelled domains | Improves retrieval only; no notion of clause, table or provenance |
| [7] | 2024 | Reranking | Multilingual cross-encoder scoring query–passage pairs jointly | Strong off-the-shelf second-stage ranker with long-input support | Reranker only, cost grows with candidate count; untrained on normative text |
| [8] | 2024 | RAG evaluation | Reference-free LLM-judged faithfulness and relevance metrics | Makes RAG architectures comparable without gold answers | Non-deterministic; judge-dependent; no retrieval-quality or provenance metric |
| [9] | 2024 | Graph retrieval | LLM-extracted entity–relation graph, local and community-level search | Better global answers than plain vector retrieval | Expensive at ingestion; entity-centred; targets summarisation, not cited clause answers |
| [10] | 2024 | Document parsing | Layout labelling plus structure model, tables as first-class objects | Layout-aware conversion preserves reading order and table structure | Not a chunking or retrieval study; no standards-specific hierarchy |
| [11] | 2019 | Sentence embeddings | Siamese shared-weight BERT with pooling, compared by cosine similarity | Makes pre-computed semantic search practical | Sentence-scale English text; a whole clause with a table blurs into one vector |
| [12] | 2024 | Target corpus | Numbered clauses, non-numeric annexes, normative tables, explicit cross-references | Supplies the answerable content and the link structure | Small, fixed, non-public-domain; no labelled queries; no published RAG results |

---

## 3. Comparative discussion

### 3.1 Mapping onto the three pairings

The project implements three interchangeable chunking × retrieval pairings over one fixed corpus, so
that the contribution of each design decision can be measured rather than assumed. The surveyed
work maps onto them as follows.

| Pairing | Chunking basis | Retrieval basis | Prior work followed |
|---|---|---|---|
| **1 — Baseline** | Generic recursive character splitting (child 256, overlap 25) with larger parents (1024) | Parent-document retrieval: match small children, return the enclosing parent | The general small-chunk / large-context idea of parent–child retrieval; sentence-level embeddings from [11] for the index |
| **2 — Structure-aware** | Docling-derived section chunks (512, overlap 50), tables atomic, clause metadata attached | Metadata-filtered hybrid search: Chroma dense + BM25, combined by reciprocal rank fusion | Dense retrieval [2] and sentence embeddings [11]; BM25 [3]; rank fusion [4]; structure-aware parsing [10] |
| **3 — Advanced** | Table-first standalone table chunks plus clause-level text chunks, with cross-references extracted into metadata | Self-query candidate generation, hybrid fusion, breadth-first expansion over the cross-reference graph, then cross-encoder reranking of the top candidates | Reranking [5]–[7]; graph-based retrieval [9]; instruction-aware dense retrieval [6] |

### 3.2 Where the project follows prior work

- **Hybrid fusion through reciprocal rank fusion.** Following [3] and [4], the lexical and dense
  retrievers produce independent ranked lists that are combined by reciprocal rank fusion with a
  fixed weighting of the vector and BM25 components (0.6 / 0.4 in `config.py`). This is the
  parameter-light default endorsed by [4], and it avoids the score-calibration problem that makes
  weighted score fusion fragile.
- **Parent/child retrieval.** As in Pairing 1, retrieval operates on small units for precision
  while the language model receives a larger enclosing unit for context, so that a matched fact is
  not answered out of its surrounding explanation.
- **Cross-encoder reranking.** Consistent with [5], [6] and [7], the first stage is cheap and
  approximate and the second stage is more expensive and more accurate; only the top candidates
  (rerank over the top 20, returning the top 5) are reranked, which bounds the cost.
- **LLM-judge evaluation.** Following [8], qualitative properties of the answer — faithfulness to
  the retrieved context and relevance to the question — are scored by a judge model, alongside
  deterministic retrieval metrics that do not depend on any model.
- **Local, open component models.** The use of general-purpose, openly available embedding and
  reranking checkpoints ([6], [7], [11]) is what makes a fully local system feasible without
  task-specific training data.

### 3.3 Where the project extends prior work

- **Clause outline derived from a layout parser.** [10] provides the structured document; this
  project goes further in `ingestion/structure.py` and turns that structure into a clause-level
  outline with headings, annexes, tables and per-block page provenance, because the externally
  defined hierarchy of the standard is not present in the parser's generic labels.
- **Annex-aware heading normalisation.** A Markdown-level heading regex misses annex headings and
  non-numeric headings and collapses large regions of the document into single oversized sections.
  Working from the parser's structured labels instead keeps annexes and lettered headings as
  first-class sections.
- **Table atomicity as a design constraint.** None of the surveyed chunking work treats a
  normative table as an indivisible unit. Here, `ingestion/chunk_utils.py` packs text at block
  boundaries and splits only oversized tables on row boundaries with the header repeated, so a
  table is never fragmented mid-row and every table chunk remains interpretable on its own.
- **Cross-references as a first-class graph.** [9] builds an entity graph by prompting an LLM at
  ingestion time. Here the links are already written in the document: references of the form
  "see 4.2" or "refer to Annex A" are extracted deterministically into graph edges in
  `graph/builder.py`, with node identifiers qualified by document part so that identically numbered
  clauses in different parts do not collide. Multi-hop expansion then pulls in clauses that a
  matched clause explicitly depends on.
- **Citation-level verifiability as a metric.** The surveyed evaluation work measures faithfulness
  of the answer text. This project additionally requires the answer to name a clause that was
  actually retrieved, and scores that deterministically, so a fluent but ungrounded answer fails.

---

## 4. Identified research gaps

1. **Generic text splitters degrade on standards.** Chunking methods that assume prose with regular
   numeric headings are not adequately characterised on documents with annexes, non-numeric and
   lettered headings, and large normative tables. Fragmenting such a table mid-row produces a
   chunk that is meaningless alone, and merging an annex into its parent section destroys the
   boundary a reader would use.
2. **Parent/child and reranking evidence comes from the wrong text distribution.** Studies of
   parent–child retrieval and cross-encoder reranking overwhelmingly use web pages, news and
   open-domain QA passages, which are short and largely self-contained. How these techniques behave
   on long normative clauses, where a single requirement is entangled with definitions and
   cross-references, has not been adequately addressed.
3. **Cross-reference and multi-hop reasoning in standards is rarely modelled as a graph.** Standards
   state their own link structure in plain text. Retrieval systems that treat such references as
   ordinary tokens lose the multi-hop path a human reader would follow, and there is little published
   treatment of standards cross-references as an explicit, traversable graph.
4. **Structure-aware chunking is evaluated on answer accuracy alone.** Whether a chunking strategy
   produces coherent boundaries or dense, relevant units is usually inferred indirectly from
   downstream answer quality. Chunk-level quality measures such as boundary coherence, table
   integrity and relevance density are not adequately reported.
5. **Structure, hybridity, graph expansion and reranking are rarely combined and ablated together.**
   The techniques above are usually studied in isolation on different corpora. Measuring their
   marginal contribution on one fixed corpus with one shared harness has not been adequately
   addressed, which makes the engineering trade-off between them hard to judge.
6. **Reproducible, open, local-first evaluation on a small fixed corpus is uncommon.** Much of the
   reported work depends on hosted models, proprietary corpora or large benchmark collections. A
   harness in which the corpus, the index, the embedder, the reranker and the judge are all local
   and the seed questions are fixed has not been adequately addressed for domain corpora of this
   size.
7. **Citation-level verifiability is rarely measured.** Answer faithfulness is usually judged as
   text entailment. Whether the specific clause the answer attributes its claim to is among the
   clauses actually retrieved — the property a standards user most needs — is rarely measured at all.

---

## 5. How this project addresses the gaps

| Gap | Project mechanism |
|---|---|
| 1. Generic splitters degrade on standards | `ingestion/structure.py` recovers the clause outline from the parser's structured labels instead of a Markdown regex; `ingestion/chunk_utils.py` keeps tables atomic and bounds oversized tables by row-boundary splitting with a repeated header. |
| 2. Evidence from the wrong text distribution | The three pairings are all run over the same fixed ISO 15118 corpus, so hybridity and reranking are characterised on normative clause text rather than assumed from web-text results. |
| 3. Cross-references not modelled as a graph | `graph/builder.py` builds a directed cross-reference graph from extracted clause and annex references, and `retrieval/pair3_retriever.py` expands candidates over it before ranking. |
| 4. Chunking evaluated on answer accuracy only | `eval/metrics.py` reports chunk-level statistics alongside retrieval metrics, and `scripts/inspect_chunks.py` provides a dedicated diagnostic for boundary and table behaviour. |
| 5. Techniques rarely combined and ablated | The three pairings share one ingestion path, one answer chain and one harness, so the marginal value of structure, hybrid fusion, graph expansion and reranking is measured rather than asserted. |
| 6. Reproducible local-first evaluation | The evaluation entry points under `eval/` run end-to-end with local models, a fixed gold question set and no hosted dependency; uncomputable judge scores are recorded as flagged `None` instead of being silently dropped. |
| 7. Citation-level verifiability rarely measured | `eval/judge.py` computes citation accuracy deterministically, by checking that the clause numbers cited in the answer were among the clauses actually retrieved. |

---

## 6. Conclusion

The surveyed literature establishes each individual component this project needs: a
retrieve-then-generate architecture [1], a dense retriever [2, 11], a principled lexical retriever
[3], a robust fusion rule [4], a second-stage reranking strategy [5–7], a layout-aware parser
[10], a graph-based retrieval idea [9], and an approach to reference-free RAG evaluation [8]. What
it does not establish is how these components behave when combined on a small, fixed, table-heavy
technical standard such as ISO 15118 [12]. The evidence base is dominated by web, news and
open-domain QA text, where passages are short and self-contained and where document hierarchy is
largely absent. The identified gaps follow directly from that mismatch: structure-aware chunking,
table integrity, standards cross-references, chunk-level quality metrics, joint ablation,
reproducible local evaluation and citation-level verification are each under-addressed. This
project responds by fixing one corpus, one ingestion path and one evaluation harness, and by
varying only the chunking and retrieval design across three pairings, so that the contribution of
each idea is measured on the target domain rather than inherited from general-domain results.

---

## 7. References

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

### 7.1 Software and framework sources

The following are software and framework documentation, not academic publications. They are listed
separately and are deliberately excluded from the numbered reference list above.

- LangChain framework documentation — retrieval modules, in particular `ParentDocumentRetriever`
  (small-chunk matching with larger parent context) and `SelfQueryRetriever` (LLM-constructed
  metadata filters), together with the `EnsembleRetriever` used for reciprocal rank fusion.
- LangChain Chroma integration documentation — persistent vector store and metadata-filtered
  queries, used for the clause-number lookup path.
- LangChain Community BM25 retriever documentation — the sparse half of hybrid search.
- BAAI `bge-mixtral` / Qwen embedding and reranker model documentation as distributed on the
  Hugging Face Model Hub, covering intended input formatting and maximum sequence lengths.
- Ollama model runner documentation for the local chat model, and the RAGAS project documentation
  for the reference-free metric definitions referenced by `eval/`.
