"""Build and cache the RAG pipelines for the Streamlit UI.

Memory note: building all three pipelines at once was OOM-killed on an 8 GB machine
(see docs/MILESTONE_7.md). Single-pipeline mode caches only the selected pipeline;
compare mode builds each pipeline in turn and releases it.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the repo root importable even if this module is loaded directly (see ui/app.py).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gc

import streamlit as st

from config import CHROMA_PERSIST_DIR, CLOUD_MODE, ISO_PDF_DIR

PIPELINE_NAMES = (
    "Pairing 1: Recursive + Parent-Child",
    "Pairing 2: Section + Hybrid",
    "Pairing 3: Advanced ISO + Graph",
)


@st.cache_resource(show_spinner=False)
def load_structures() -> list:
    """Load and cache the corpus documents + structured outlines (Docling JSON cache)."""
    from ingestion.loader import load_iso_structures

    return load_iso_structures(ISO_PDF_DIR, use_ocr=False)


def _load_chunks_from_store(persist_directory: str, collection_name: str) -> list:
    """Load all chunks from a persisted Chroma collection (cloud mode).

    Reconstructs ``Document`` objects (content + metadata) from the store so BM25 and
    the graph can be rebuilt without re-running Docling chunking. Requires the index
    to have been built and shipped with the deployment.
    """
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    from models.embeddings import get_embeddings

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )
    data = vectorstore._collection.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=content, metadata=meta or {})
        for content, meta in zip(data.get("documents") or [], data.get("metadatas") or [])
        if content
    ]
    if not docs:
        raise RuntimeError(
            f"Cloud mode: collection {collection_name!r} in {persist_directory!r} is empty. "
            "Build and ship the index locally before deploying (see AGENTS.md)."
        )
    return docs


def build_pipeline(name: str) -> dict:
    """Construct a single pipeline. Returns a dict with retriever/chain/vectorstore/graph."""
    from chain.rag_chain import build_condense_chain, build_conversational_chain

    if CLOUD_MODE and name.startswith("Pairing 1"):
        raise RuntimeError(
            "Pairing 1 is not available in cloud mode (it has no persisted vector store). "
            "Use Pairing 2 or 3."
        )

    # Cloud mode skips Docling entirely: chunks are recovered from the shipped index.
    # The eval CLI builds indexes into ./data/indexes/pairN (see eval/run_pairing.py),
    # so cloud mode reads from there — NOT CHROMA_PERSIST_DIR.
    cloud_persist_dir: str | None = None
    if CLOUD_MODE:
        if name.startswith("Pairing 2"):
            cloud_persist_dir = "./data/indexes/pair2"
            documents = _load_chunks_from_store(cloud_persist_dir, "pair2_section_hybrid")
        else:
            cloud_persist_dir = "./data/indexes/pair3"
            documents = _load_chunks_from_store(cloud_persist_dir, "pair3_graph_hybrid")
    else:
        documents = load_structures()

    if name.startswith("Pairing 1"):
        from ingestion.chunking_pair1 import create_parent_child_chunks
        from retrieval.pair1_retriever import build_parent_child_retriever

        parents, children = create_parent_child_chunks(documents)
        retriever = build_parent_child_retriever(children, parents)
        return {
            "retriever": retriever.invoke,
            "chain": build_conversational_chain(),
            "condense": build_condense_chain(),
            "vectorstore": None,
            "graph": None,
        }

    if name.startswith("Pairing 2"):
        from ingestion.chunking_pair2 import create_section_based_chunks
        from retrieval.pair2_retriever import build_metadata_filtered_hybrid_retriever

        chunks = create_section_based_chunks(documents) if cloud_persist_dir is None else documents
        retriever, vectorstore = build_metadata_filtered_hybrid_retriever(
            chunks,
            persist_directory=cloud_persist_dir or CHROMA_PERSIST_DIR,
        )
        return {
            "retriever": retriever.invoke,
            "chain": build_conversational_chain(),
            "condense": build_condense_chain(),
            "vectorstore": vectorstore,
            "graph": None,
        }

    if name.startswith("Pairing 3"):
        from graph.builder import build_cross_reference_graph
        from ingestion.chunking_pair3 import create_advanced_iso_chunks
        from retrieval.pair3_retriever import (
            build_graph_enhanced_retriever,
            graph_expanded_retrieve,
        )

        chunks = create_advanced_iso_chunks(documents) if cloud_persist_dir is None else documents
        graph = build_cross_reference_graph(chunks)
        retriever, vectorstore, graph = build_graph_enhanced_retriever(
            chunks,
            graph,
            persist_directory=cloud_persist_dir or CHROMA_PERSIST_DIR,
        )

        def retrieve(query: str) -> list:
            return graph_expanded_retrieve(query, retriever, graph, vectorstore=vectorstore)

        return {
            "retriever": retrieve,
            "chain": build_conversational_chain(),
            "condense": build_condense_chain(),
            "vectorstore": vectorstore,
            "graph": graph,
        }

    raise ValueError(f"Unknown pipeline: {name}")


@st.cache_resource(show_spinner="Building pipeline...")
def get_pipeline(name: str, settings_key: tuple = ()) -> dict:
    """Cached single-pipeline accessor (holds at most one pipeline in memory).

    ``settings_key`` (the runtime model-settings signature) is part of the cache key,
    so changing the model/API-key settings rebuilds the pipeline with the new models.
    """
    return build_pipeline(name)


def run_query(
    pipeline: dict,
    query: str,
    history: list | None = None,
    section_filter: str | None = None,
) -> tuple[str, list]:
    """Run retrieval + generation for a single pipeline, with optional chat history.

    When ``history`` is provided, the follow-up is first rewritten into a standalone question
    (so retrieval and the answer understand references like "tell me more"), and the history
    is passed to the LLM for conversational memory.
    """
    from chain.rag_chain import format_docs

    history = history or []
    standalone = query
    condense = pipeline.get("condense")
    if history and condense is not None:
        standalone = condense(history, query)

    if section_filter and pipeline.get("vectorstore") is not None:
        from retrieval.pair2_retriever import filtered_retrieve

        sources = filtered_retrieve(pipeline["vectorstore"], standalone, section_filter)
    else:
        sources = pipeline["retriever"](standalone)

    context = format_docs(sources)
    answer = pipeline["chain"].invoke(
        {"context": context, "question": standalone, "history": history}
    )
    return answer, sources


def compare_all(
    query: str,
    history: list | None = None,
    section_filter: str | None = None,
) -> dict[str, tuple[str, list]]:
    """Run the same query through all pipelines, building/releasing each in turn."""
    results: dict[str, tuple[str, list]] = {}
    for name in PIPELINE_NAMES:
        pipeline = build_pipeline(name)  # uncached -> memory-safe for compare mode
        try:
            results[name] = run_query(pipeline, query, history, section_filter)
        finally:
            del pipeline
            gc.collect()
    return results
