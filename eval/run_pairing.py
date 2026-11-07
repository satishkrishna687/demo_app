"""Build and thoroughly evaluate a single pairing.

Usage:
    python -m eval.run_pairing --pairing 1 [--limit N] [--runs 1] [--no-judge] [--max-judge N]

Produces:
    eval/report_pair<N>.md   (human-readable, all four RAG_EVAL_METRICS categories)
    eval/result_pair<N>.json (raw metrics, consumed by the Milestone 13 comparison report)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from eval.metrics import dir_size_bytes
from eval.pair_report import generate_pair_report
from eval.runner import evaluate_pairing

logger = logging.getLogger(__name__)

PAIRINGS = {
    1: "Pairing 1: Recursive + Parent-Child",
    2: "Pairing 2: Section + Hybrid",
    3: "Pairing 3: Advanced ISO + Graph",
}


def build_pairing(number: int, documents: list, persist_dir: str) -> dict:
    """Build one pairing, measuring indexing time; return pipeline + corpus chunks."""
    from chain.rag_chain import build_rag_chain

    if number == 1:
        from ingestion.chunking_pair1 import create_parent_child_chunks
        from retrieval.pair1_retriever import build_parent_child_retriever

        t0 = time.perf_counter()
        parents, children = create_parent_child_chunks(documents)
        chunk_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        retriever = build_parent_child_retriever(
            children, parents, persist_directory=persist_dir, collection_name="pair1_parent_child"
        )
        embed_ms = (time.perf_counter() - t0) * 1000

        return {
            "retriever": retriever.invoke,
            "chain": build_rag_chain(retriever),
            "corpus_chunks": parents,
            "chunking_ms": chunk_ms,
            "embedding_ms": embed_ms,
            "notes": "corpus_chunks = parent chunks (children are internal)",
        }

    if number == 2:
        from ingestion.chunking_pair2 import create_section_based_chunks
        from retrieval.pair2_retriever import build_metadata_filtered_hybrid_retriever

        t0 = time.perf_counter()
        chunks = create_section_based_chunks(documents)
        chunk_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        retriever, _vectorstore = build_metadata_filtered_hybrid_retriever(
            chunks, persist_directory=persist_dir, collection_name="pair2_section_hybrid"
        )
        embed_ms = (time.perf_counter() - t0) * 1000

        return {
            "retriever": retriever.invoke,
            "chain": build_rag_chain(retriever),
            "corpus_chunks": chunks,
            "chunking_ms": chunk_ms,
            "embedding_ms": embed_ms,
            "notes": "hybrid vector+BM25",
        }

    if number == 3:
        from graph.builder import build_cross_reference_graph
        from ingestion.chunking_pair3 import create_advanced_iso_chunks
        from retrieval.pair3_retriever import (
            build_graph_enhanced_retriever,
            graph_expanded_retrieve,
        )

        t0 = time.perf_counter()
        chunks = create_advanced_iso_chunks(documents)
        graph = build_cross_reference_graph(chunks)
        chunk_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        retriever, vectorstore, graph = build_graph_enhanced_retriever(
            chunks, graph, persist_directory=persist_dir, collection_name="pair3_graph_hybrid"
        )
        embed_ms = (time.perf_counter() - t0) * 1000

        def retrieve(query: str) -> list:
            return graph_expanded_retrieve(query, retriever, graph, vectorstore=vectorstore)

        return {
            "retriever": retrieve,
            "chain": build_rag_chain(retrieve),
            "corpus_chunks": chunks,
            "chunking_ms": chunk_ms,
            "embedding_ms": embed_ms,
            "notes": "graph-enhanced hybrid + rerank",
        }

    raise ValueError(f"Unknown pairing number: {number}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate one pairing thoroughly.")
    parser.add_argument("--pairing", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--persist-dir", default=None,
                        help="Chroma dir for this pairing (defaults to ./data/indexes/pairN)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--max-judge", type=int, default=None,
                        help="Judge only the first N queries (cost control).")
    parser.add_argument("--use-ocr", action="store_true")
    args = parser.parse_args(argv)

    from config import ISO_PDF_DIR, LLM_MODEL, LLM_PROVIDER
    from ingestion.loader import load_iso_structures
    from models.device import describe_device

    persist_dir = args.persist_dir or f"./data/indexes/pair{args.pairing}"
    name = PAIRINGS[args.pairing]

    gold = json.loads(Path("eval/gold_set.json").read_text(encoding="utf-8"))
    if args.limit:
        gold = gold[: args.limit]

    documents = load_iso_structures(ISO_PDF_DIR, use_ocr=args.use_ocr)

    logger.info("Building %s (persist=%s) ...", name, persist_dir)
    t0 = time.perf_counter()
    built = build_pairing(args.pairing, documents, persist_dir)
    build_ms = (time.perf_counter() - t0) * 1000

    indexing_time_ms = built["chunking_ms"] + built["embedding_ms"]
    index_size = dir_size_bytes(persist_dir)

    result = evaluate_pairing(
        name,
        {"retriever": built["retriever"], "chain": built["chain"]},
        gold,
        corpus_chunks=built["corpus_chunks"],
        indexing_time_ms=indexing_time_ms,
        index_size_bytes=index_size,
        runs=args.runs,
        use_judge=not args.no_judge,
        max_judge_queries=args.max_judge,
    )
    result["indexing_breakdown_ms"] = {
        "chunking": built["chunking_ms"],
        "embedding": built["embedding_ms"],
        "build_total": build_ms,
    }
    result["notes"] = built.get("notes")

    config_lines = {
        "LLM provider": LLM_PROVIDER,
        "LLM model": LLM_MODEL,
        "Device": describe_device(),
        "Chunk stats source": built.get("notes", ""),
        "Index dir": persist_dir,
    }

    report_path = f"eval/report_pair{args.pairing}.md"
    generate_pair_report(result, report_path, config_lines=config_lines)

    result_path = f"eval/result_pair{args.pairing}.json"
    Path(result_path).write_text(json.dumps(result, indent=2), encoding="utf-8")

    logger.info("Wrote %s and %s", report_path, result_path)
    print(f"Report: {report_path}\nRaw:    {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
