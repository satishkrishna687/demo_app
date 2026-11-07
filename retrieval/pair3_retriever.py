"""Pairing 3 retrieval: four-stage graph-enhanced hybrid retrieval.

Stages:
1. Metadata-aware candidate generation (``SelfQueryRetriever``, with a vector fallback).
2. Hybrid search (self-query + BM25) fused by ``EnsembleRetriever``.
3. Cross-reference graph expansion (BFS over the NetworkX graph).
4. Cross-encoder reranking of the expanded candidate set.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import networkx as nx
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers import EnsembleRetriever, SelfQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from config import (
    CHROMA_PERSIST_DIR,
    HYBRID_BM25_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    RERANK_TOP_N,
    TOP_K,
)
from graph.builder import chunk_id
from models.embeddings import get_embeddings
from models.llm import get_llm
from models.reranker import rerank
from vectorstore.store import ensure_indexed

logger = logging.getLogger(__name__)

_METADATA_FIELDS = [
    AttributeInfo(
        name="section_number",
        description="The ISO clause number (e.g. '4.1.2')",
        type="string",
    ),
    AttributeInfo(
        name="content_type",
        description="Content type: 'text' or 'table'",
        type="string",
    ),
    AttributeInfo(
        name="document_id",
        description="The ISO 15118 part (e.g. 'ISO-15118-2')",
        type="string",
    ),
    AttributeInfo(
        name="source_file",
        description="The source PDF file name",
        type="string",
    ),
]


def build_graph_enhanced_retriever(
    chunks: list[Document],
    graph: nx.DiGraph,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = "pair3_graph_hybrid",
    vector_weight: float = HYBRID_VECTOR_WEIGHT,
    bm25_weight: float = HYBRID_BM25_WEIGHT,
    k: int = TOP_K,
) -> tuple[EnsembleRetriever, Chroma, nx.DiGraph]:
    """Build the hybrid retriever for Pairing 3 and return it with the store and graph."""
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    ensure_indexed(vectorstore, chunks, ids=[chunk_id(c) for c in chunks])

    # Stage 1: metadata-aware retrieval. SelfQueryRetriever depends on the LLM's structured
    # output support; fall back to a plain vector retriever if it cannot be built.
    try:
        vector_component: Any = SelfQueryRetriever.from_llm(
            llm=get_llm(),
            vectorstore=vectorstore,
            document_contents="ISO 15118 engineering specification",
            metadata_field_info=_METADATA_FIELDS,
            search_kwargs={"k": k},
        )
    except Exception as exc:
        # langchain-community 0.4.2's self-query translators import
        # `DatabricksVectorSearch`, which is absent -> from_llm raises ImportError.
        # Fall back to a plain vector retriever (see docs/MILESTONE_6.md).
        logger.warning(
            "SelfQueryRetriever unavailable (%s: %s); using vector retriever fallback.",
            type(exc).__name__,
            exc,
        )
        vector_component = vectorstore.as_retriever(search_kwargs={"k": k})

    # Stage 2: BM25 + ensemble fusion.
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    ensemble = EnsembleRetriever(
        retrievers=[vector_component, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )
    return ensemble, vectorstore, graph


def graph_expanded_retrieve(
    query: str,
    hybrid_retriever: EnsembleRetriever,
    graph: nx.DiGraph,
    vectorstore: Chroma | None = None,
    max_depth: int = 1,
    top_k: int = TOP_K,
    rerank_top_n: int = RERANK_TOP_N,
    use_rerank: bool = True,
) -> list[Document]:
    """Run hybrid search, expand via the graph, and rerank the candidates."""
    # Stages 1-2: hybrid search (fall back to the vector store if the ensemble fails).
    try:
        initial_results = hybrid_retriever.invoke(query)
    except Exception:
        initial_results = vectorstore.similarity_search(query, k=top_k) if vectorstore else []

    # Stage 3: graph expansion (BFS up to max_depth).
    expanded: list[Document] = list(initial_results)
    seen = {chunk_id(d) for d in initial_results}

    frontier = [chunk_id(d) for d in initial_results if chunk_id(d) in graph]
    for _ in range(max_depth):
        next_frontier: list[str] = []
        for node_id in frontier:
            neighbours = list(graph.successors(node_id)) + list(graph.predecessors(node_id))
            for neighbour_id in neighbours:
                if neighbour_id in seen or neighbour_id not in graph:
                    continue
                neighbour_chunk = graph.nodes[neighbour_id].get("chunk")
                if neighbour_chunk is not None:
                    expanded.append(neighbour_chunk)
                    seen.add(neighbour_id)
                    next_frontier.append(neighbour_id)
        frontier = next_frontier

    # Stage 4: rerank the (bounded) candidate set.
    if use_rerank and len(expanded) > 1:
        candidates = expanded[: max(rerank_top_n, top_k)]
        try:
            return rerank(query, candidates, top_k)
        except Exception:
            pass  # fall back to unranked results

    return expanded[: top_k * 2]