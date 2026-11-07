"""Build the walkthrough notebook ``Grounded_RAG_QA_ISO15118.ipynb``.

Constructs a Jupyter notebook that runs the ISO 15118 RAG project end-to-end from the
repository root. The notebook is written as JSON directly (no ``nbformat`` dependency) so
it is valid anywhere.

Run from the repo root:
    python scripts/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path("Grounded_RAG_QA_ISO15118.ipynb")


def _source(text: str) -> list[str]:
    """Convert a code/markdown block into an ipynb ``source`` list."""
    lines = text.splitlines(keepends=True)
    # Jupyter tolerates a final line without a trailing newline.
    return lines


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _source(text),
    }


CELLS = []

# --------------------------------------------------------------------------- title
CELLS.append(_md(
    """\
# Grounded RAG QA System over ISO 15118

**Structure-Aware Retrieval for ISO Standard Documents: A Comparative Study of Chunking and Retrieval Strategies**

An end-to-end walkthrough of the retrieval-augmented generation pipeline:
**ingestion -> structure recovery -> chunking (3 pairings) -> indexing -> retrieval -> generation -> evaluation**.

> **Run this notebook from the repository root.** The code imports the project's own packages
> (`config`, `ingestion`, `models`, `retrieval`, `eval`), which live one level above the default
> notebook directory. The `DEVICE` environment variable can be set to `cpu` to force CPU.
"""
))

# --------------------------------------------------------------------------- setup
CELLS.append(_code(
    """\
# Make the repository root importable (Jupyter may start in the notebook's directory).
import os, sys
from pathlib import Path
ROOT = Path(os.getcwd())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import project config FIRST: it sets PyTorch env vars before torch is loaded.
import config

print("Repo root:", ROOT)
print("ISO_PDF_DIR:", config.ISO_PDF_DIR)
print("EMBEDDING_MODEL:", config.EMBEDDING_MODEL)
print("RERANKER_MODEL:", config.RERANKER_MODEL)
print("LLM_PROVIDER / LLM_MODEL:", config.LLM_PROVIDER, "/", config.LLM_MODEL)
print("TOP_K:", config.TOP_K, "| RERANK_TOP_N:", config.RERANK_TOP_N)
print("TABLE_MAX_CHARS:", config.TABLE_MAX_CHARS)
"""
))

CELLS.append(_code(
    """\
from models.device import describe_device
print("Compute device:", describe_device())
"""
))

# --------------------------------------------------------------------------- ingestion
CELLS.append(_md(
    """\
## 1. Ingestion (Docling)

Each PDF is converted **once** with Docling (layout + table structure, OCR off) and cached as JSON
under `data/parsed/`. `load_iso_structures` returns one `StructuredDocument` per PDF: the Markdown
`document` (with page-break metadata) plus the structured `sections` outline that drives chunking.

> The first call may be slow if the Docling JSON cache is cold; later calls reuse the cache.
"""
))

CELLS.append(_code(
    """\
from ingestion.loader import load_iso_structures

documents = load_iso_structures(config.ISO_PDF_DIR, use_ocr=False)
print("Ingested", len(documents), "structured document(s):")
for sd in documents:
    md = sd.document.metadata
    print("  ", md["document_id"], "| sections:", len(sd.sections),
          "| chars:", len(sd.document.page_content))
"""
))

CELLS.append(_code(
    """\
# Preview the recovered section outline for the first document.
sd = documents[0]
for sec in sd.sections[:6]:
    print(f"{sec.number:<8} | {sec.title:<32} | page {sec.page} | blocks: {len(sec.blocks)}")
"""
))

# --------------------------------------------------------------------------- chunking
CELLS.append(_md(
    """\
## 2. Chunking — the three pairings

All three chunkers consume the **same** `list[StructuredDocument]`, so any measured difference is
attributable to chunking/retrieval, not parsing.

* **Pairing 1 (baseline)** — recursive parent/child chunks; structure-agnostic, no clause metadata.
* **Pairing 2** — section-bounded chunks with clause metadata; tables kept atomic.
* **Pairing 3** — standalone table chunks + clause text chunks, with extracted cross-references.
"""
))

CELLS.append(_code(
    """\
from ingestion.chunking_pair1 import create_parent_child_chunks

parents, children = create_parent_child_chunks(documents)
print("Pairing 1 — parent chunks:", len(parents), "| child chunks:", len(children))
print("Child metadata sample:", {k: v for k, v in children[0].metadata.items() if k != "source_file"})
"""
))

CELLS.append(_code(
    """\
from ingestion.chunking_pair2 import create_section_based_chunks

p2_chunks = create_section_based_chunks(documents)
print("Pairing 2 — section chunks:", len(p2_chunks))
print("Sample metadata:", {k: v for k, v in p2_chunks[0].metadata.items() if k != "source_file"})
"""
))

CELLS.append(_code(
    """\
from ingestion.chunking_pair3 import create_advanced_iso_chunks

p3_chunks = create_advanced_iso_chunks(documents)
tables = [c for c in p3_chunks if c.metadata.get("content_type") == "table"]
with_refs = [c for c in p3_chunks if c.metadata.get("section_references")]
print("Pairing 3 — chunks:", len(p3_chunks), "| standalone table chunks:", len(tables))
print("Chunks with section references:", len(with_refs))
if with_refs:
    m = with_refs[0].metadata
    print("  section:", m["section_number"], "| refs:", m["section_references"])
"""
))

CELLS.append(_code(
    """\
from graph.builder import build_cross_reference_graph

graph = build_cross_reference_graph(p3_chunks)
print("Cross-reference graph — nodes:", graph.number_of_nodes(), "| edges:", graph.number_of_edges())
"""
))

# --------------------------------------------------------------------------- indexing + retrieval
CELLS.append(_md(
    """\
## 3. Indexing and retrieval

The next cell builds the **Pairing 2** hybrid retriever: a local dense Chroma index plus a BM25
sparse index, fused by Reciprocal Rank Fusion. This loads the local embedding model
(`Qwen/Qwen3-Embedding-0.6B`) — the first run downloads the weights.
"""
))

CELLS.append(_code(
    """\
from retrieval.pair2_retriever import build_metadata_filtered_hybrid_retriever

p2_retriever, p2_vectorstore = build_metadata_filtered_hybrid_retriever(p2_chunks)
print("Built Pairing 2 hybrid (dense + BM25 + RRF) retriever.")
"""
))

CELLS.append(_code(
    """\
query = "What is the nominal AC charging voltage?"
docs = p2_retriever.invoke(query)
print(f"Retrieved {len(docs)} chunk(s):")
for d in docs[:5]:
    m = d.metadata
    print(f"  [{m.get('document_id')} | {m.get('section_number')} | page {m.get('page_number')}] {d.page_content[:70]!r}")
"""
))

# --------------------------------------------------------------------------- generation
CELLS.append(_md(
    """\
## 4. Generation (citation-grounded answer)

Uses the local chat LLM (default `llama3.2:3b-instruct-q4_K_M` via Ollama). If Ollama is **not
running**, the generation cell will raise a connection error — start it with `ollama serve` first.
Retrieval is done once and the same context is passed to the chain and shown as sources.
"""
))

CELLS.append(_code(
    """\
from chain.rag_chain import build_conversational_chain, format_docs

conv_chain = build_conversational_chain()          # requires the chat LLM
docs = p2_retriever.invoke(query)
context = format_docs(docs)
answer = conv_chain.invoke({"context": context, "question": query, "history": []})

print("Q:", query)
print("A:", answer)
print("\\nSources cited:")
print(context[:600])
"""
))

# --------------------------------------------------------------------------- evaluation
CELLS.append(_md(
    """\
## 5. Evaluation

`eval` provides deterministic retrieval metrics (`Recall@k`, `Precision@k`, `nDCG@k`, `MRR`) and
LLM-as-judge generation metrics. First we compute **deterministic retrieval metrics** on a small
subset (no model calls). Then we optionally run the full `evaluate_pairing` harness (requires the
chat LLM). The complete 32-query, single-run results are precomputed in `eval/result_pairN.json`
and shown in the next section.
"""
))

CELLS.append(_code(
    """\
import json
from eval import metrics as m

gold = json.loads(Path("eval/gold_set.json").read_text(encoding="utf-8"))
print("Gold set size:", len(gold))

subset = gold[:5]
flags_per_q = []
for entry in subset:
    docs = p2_retriever.invoke(entry["query"])
    flags_per_q.append(m.relevance_flags(docs, entry))

agg = m.aggregate_retrieval(flags_per_q, k_list=(1, 3, 5))
print(f"Deterministic retrieval metrics on {len(subset)} queries (Pairing 2):")
for k in (1, 3, 5):
    print(f"  recall@{k}={agg[f'recall@{k}']:.3f}  precision@{k}={agg[f'precision@{k}']:.3f}")
print(f"  mrr={agg['mrr']:.3f}")
"""
))

CELLS.append(_code(
    """\
# Optional: full judge-based evaluation (requires the chat LLM). Uses build_rag_chain so the
# retriever is run inside the chain and the chain accepts a plain query string.
try:
    from chain.rag_chain import build_rag_chain
    from eval.runner import evaluate_pairing

    eval_chain = build_rag_chain(p2_retriever)
    result = evaluate_pairing(
        "Pairing 2: Section + Hybrid (notebook)",
        {"retriever": p2_retriever.invoke, "chain": eval_chain},
        gold[:3],
        corpus_chunks=p2_chunks,
        use_judge=False,
    )
    print("generation (subset):", {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in result["generation"].items()})
except Exception as e:  # noqa: BLE001
    print("Skipped full evaluation (requires a running chat LLM):", type(e).__name__, "-", e)
"""
))

# --------------------------------------------------------------------------- precomputed results
CELLS.append(_md(
    """\
## 6. Precomputed results (single run, 32 queries)

The full evaluation was run once per pairing and persisted to `eval/result_pairN.json`. We load
and compare them here — this reproduces the comparison table from `eval/RAG_EVAL_REPORT.md`.
"""
))

CELLS.append(_code(
    """\
import pandas as pd

rows = []
for n in (1, 2, 3):
    res = json.loads(Path(f"eval/result_pair{n}.json").read_text(encoding="utf-8"))
    r, g, e = res["retrieval"], res["generation"], res["efficiency"]
    rows.append({
        "Pairing": res["pairing"],
        "Recall@5": r["recall@5"],
        "nDCG@5": round(r["ndcg@5"], 4),
        "MRR": round(r["mrr"], 4),
        "Faithfulness": round(g["faithfulness"], 4),
        "Ans. correctness": round(g["answer_correctness"], 4),
        "Citation acc.": round(g["citation_accuracy"], 4),
        "E2E (ms)": round(e["avg_e2e_ms"], 1),
    })

pd.DataFrame(rows)
"""
))

# --------------------------------------------------------------------------- conclusion
CELLS.append(_md(
    """\
## Summary

* **Pairing 3** wins **retrieval** (Recall@5 = 1.0) via cross-reference graph expansion + cross-encoder reranking.
* **Pairing 2** wins **generation quality and latency** (citation accuracy 0.96, end-to-end 2660 ms).
* Structure-aware chunking is what makes **clause-level citation** possible — the baseline cannot cite clauses.

The key methodological lesson: **retrieval quality and generation quality are separable** — the
pairing that is best at one is not automatically best at the other.

**Next steps**
* Full judged evaluation: `python -m eval.run_pairing --pairing 2 --limit 32`
* Cross-pairing comparison report: `python -m eval.comparison_report`
* Interactive UI (from the repo root): `streamlit run ui/app.py`
"""
))


NOTEBOOK = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(NOTEBOOK, indent=1), encoding="utf-8")
print(f"Wrote {OUT} with {len(CELLS)} cells")
