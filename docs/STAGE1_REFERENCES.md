# Stage 1 References — RAG over ISO 15118

**Project:** Retrieval-Augmented Generation over ISO 15118 EV Charging Communication Standards
**Document type:** Academic Stage 1 (first review) deliverable — master bibliography

---

## 1. Purpose

This file is the master bibliography for the Stage 1 review of the retrieval-augmented generation
system built over the ISO 15118 vehicle-to-grid communication interface standard. It holds the
canonical numbered reference list, a map from each reference to the place it is cited in this
project, and a separate list of software sources used for reproducibility.

The same twelve numbered entries appear, word for word and in the same order, in
`docs/STAGE1_FIRST_REVIEW_PPT.md`, `docs/STAGE1_LITERATURE_SURVEY.md`, and
`docs/STAGE1_PROPOSED_SYSTEM.md`. This file is the version of record: if the four documents ever
disagree, this one defines the numbering and the wording.

---

## 2. Numbered references

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

**Note on entry [9].** The title of entry [9] is rendered with the US spelling "summarization"
consistently across the four Stage 1 documents. The entry itself is not otherwise altered.

---

## 3. In-text citation map

| Ref | Where cited in this project | Purpose of the citation |
|---|---|---|
| [1] | `chain/rag_chain.py`, system architecture section of the Stage 1 documents | Grounding for the retrieve-then-generate paradigm the whole pipeline implements. |
| [2] | `retrieval/pair1_retriever.py`, `retrieval/pair2_retriever.py`, `retrieval/pair3_retriever.py` | Dense bi-encoder retrieval as the semantic half of the retriever. |
| [3] | `retrieval/pair2_retriever.py`, `retrieval/pair3_retriever.py` | BM25 sparse retrieval in Pairing 2 and Pairing 3, decisive for clause numbers and signal names. |
| [4] | `retrieval/pair2_retriever.py`, `retrieval/pair3_retriever.py` | Reciprocal Rank Fusion as the training-free combination rule for hybrid retrieval. |
| [5] | Literature survey §2.5, proposed-system reranking discussion | Late-interaction retrieval as the considered alternative to cross-encoder reranking, and the reason it was not adopted. |
| [6] | `models/embeddings.py`, query-side instruction handling | Zero-shot dense retrieval and the instruction-following embedder used for the unlabelled ISO corpus. |
| [7] | `models/reranker.py`, `retrieval/pair3_retriever.py` | Cross-encoder reranking in Pairing 3. |
| [8] | `eval/evaluator.py`, `eval/judge.py`, `eval/metrics.py` | Reference-free evaluation practice and the definition of the faithfulness and relevance rubrics. |
| [9] | `graph/builder.py`, `retrieval/pair3_retriever.py` | Graph-based retrieval as the motivation for the cross-reference graph, and the contrast with the deterministic edge extraction used here. |
| [10] | `ingestion/loader.py`, `ingestion/structure.py`, `ingestion/chunk_utils.py` | Docling as the PDF parser whose structured labels drive section and table handling. |
| [11] | `models/embeddings.py`, `vectorstore/store.py` | Sentence-embedding foundation for pre-computing and storing chunk vectors. |
| [12] | `ingestion/loader.py`, `graph/builder.py`, all three pairings | The corpus documents themselves: clause hierarchy, annexes, normative tables and cross-references. |

---

## 4. Software and framework sources

The entries below are software and framework documentation, not academic publications. They are
listed unnumbered and are deliberately excluded from the numbered list in section 2, so that they
cannot be confused with the twelve academic references. They are cited for reproducibility — that
is, to record the exact components the pipeline depends on — and not as research contributions.
Version numbers are deliberately omitted here; the pinned versions are recorded in
`requirements.txt` and `requirements.lock`.

- Hugging Face Hub model documentation for `Qwen/Qwen3-Embedding-0.6B` — intended input formatting
  and the built-in query and document instructions used by `models/embeddings.py`.
- LangChain documentation — retrieval modules, in particular `ParentDocumentRetriever`
  (small-chunk matching with larger parent context), `SelfQueryRetriever` (LLM-constructed metadata
  filters), and `EnsembleRetriever` (reciprocal rank fusion).
- NetworkX documentation — the graph data structures and traversal operations used by
  `graph/builder.py` for the cross-reference graph.
- Ollama documentation — the local runner serving the chat model.
- pytest documentation — the test harness under `tests/`.
- RAGAS project documentation — the reference-free metric definitions that motivate `eval/`.
- `langchain-chroma` documentation — the persistent Chroma vector store and metadata-filtered
  queries used by `vectorstore/store.py`.
- `rank_bm25` documentation — the BM25 implementation used for the sparse retriever.
- sentence-transformers documentation — the encoder interface and the reranker wrapper used in
  `models/embeddings.py` and `models/reranker.py`.
- Streamlit documentation — the user interface under `ui/`.

---

## 5. Verification guidance

Each source can be checked independently by the reviewer:

- **Peer-reviewed entries [1]–[6], [8], [11]** — search the title in the proceedings of the named
  venue (NeurIPS 2020, EMNLP 2019/2020/2023, SIGIR 2009/2020, EACL 2024) or in the ACM Digital
  Library. The proceedings pagination should match the page ranges given.
- **Preprint entries [9] and [10]** — locate them by the arXiv identifier given for [9]
  (arXiv:2404.16130); for [10] search the title on arXiv. Preprints are not peer reviewed, and are
  cited as such.
- **Model artefact [7]** — verify against the official BAAI model card published on the Hugging
  Face Model Hub, which is the authoritative description of the reranker.
- **Standard [12]** — verify the part numbers, titles and edition years in the ISO catalogue; the
  documents themselves are the files under `Iso/`.

Before final submission, every entry should be re-checked against the publisher of record, since
venue names, edition years and pagination are the details most easily transcribed incorrectly. No
source in this list is fabricated, invented or auto-generated; where a detail such as a volume,
issue, page range or DOI could not be confirmed, it has been omitted rather than supplied
approximately.

---

## 6. Cross-document consistency note

The numbering in this file is the master numbering for the Stage 1 report set. The entries in
`docs/STAGE1_FIRST_REVIEW_PPT.md`, `docs/STAGE1_LITERATURE_SURVEY.md`, and
`docs/STAGE1_PROPOSED_SYSTEM.md` must carry identical numbering, identical wording, and identical
order. If a reference is added, removed, or reworded anywhere, the change is made here first and
then propagated to the other three documents, so that every in-text citation such as `[4]` or
`[5–7]` resolves to the same work in all four files.
