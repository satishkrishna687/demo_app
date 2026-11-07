"""Evaluation harness comparing the three chunking/retrieval pipelines.

Computes custom retrieval metrics (hit rate, citation accuracy, latency, chunk counts) for
every pipeline, and — when enabled — RAGAS metrics using the configured LLM and embeddings.

The RAGAS import is broken in the pinned dependency set (ragas 0.3.1 imports a chat-model
module removed from langchain-community 0.4.2). :func:`install_ragas_shim` injects a minimal
placeholder module so RAGAS can import; the metrics still work for non-Vertex backends.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_QUERIES_PATH = "eval/test_queries.json"
DEFAULT_OUTPUT_PATH = "eval/report.md"


def install_ragas_shim() -> None:
    """Make ``ragas`` importable under langchain-community 0.4.2."""
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
        return
    except Exception:
        pass

    import langchain_community.chat_models as cm

    shim = types.ModuleType("langchain_community.chat_models.vertexai")

    class ChatVertexAI:  # pragma: no cover - placeholder, unused for Ollama/OpenAI
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise NotImplementedError("ChatVertexAI is a shim; Vertex is not supported here.")

    shim.ChatVertexAI = ChatVertexAI  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = shim
    cm.vertexai = shim  # type: ignore[attr-defined]
    logger.info("Installed ragas vertexai import shim.")


def load_test_queries(path: str = DEFAULT_QUERIES_PATH) -> list[dict]:
    """Load the evaluation query set."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _call(obj: Any, query: str) -> Any:
    """Invoke a retriever/chain that may be a Runnable (use ``.invoke``) or a callable.

    LangChain 1.x ``RunnableSequence`` objects are not directly callable, so we must use
    ``.invoke()`` when available.
    """
    invoke = getattr(obj, "invoke", None)
    if callable(invoke):
        return invoke(query)
    return obj(query)


def _expected_sections(item: dict) -> list[str]:
    expected = item.get("expected_section") or item.get("expected_sections") or []
    if isinstance(expected, str):
        return [expected] if expected else []
    return list(expected)


def _citation_ok(answer: str, retrieved_sections: list[str]) -> bool:
    """Heuristic: does the answer cite a section number that was actually retrieved?"""
    import re

    cited = re.findall(r"(?:Section|Clause)\s+(\d+(?:\.\d+)*)", answer, re.IGNORECASE)
    if not cited:
        return False
    return any(any(c == s or s.startswith(c + ".") for s in retrieved_sections) for c in cited)


def run_pipeline_evaluation(
    pipeline_name: str,
    retriever: Callable[[str], list],
    rag_chain: Callable[[str], str],
    queries: list[dict],
    use_ragas: bool = True,
) -> dict:
    """Evaluate one pipeline over the query set and return rows + aggregate metrics."""
    rows: list[dict] = []

    for item in queries:
        query = item["query"]

        t0 = time.perf_counter()
        try:
            docs = _call(retriever, query)
        except Exception as exc:
            logger.warning("[%s] retrieval failed for %s: %s", pipeline_name, item["id"], exc)
            docs = []
        retrieval_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        try:
            answer = _call(rag_chain, query)
        except Exception as exc:
            logger.warning("[%s] generation failed for %s: %s", pipeline_name, item["id"], exc)
            answer = ""
        e2e_ms = (time.perf_counter() - t0) * 1000

        retrieved_sections = [d.metadata.get("section_number", "") for d in docs]
        expected = _expected_sections(item)
        hit = any(
            any(exp == sec or sec.startswith(exp + ".") for sec in retrieved_sections)
            for exp in expected
        ) if expected else None

        rows.append(
            {
                "query_id": item["id"],
                "query": query,
                "category": item.get("category", "uncategorized"),
                "answer": answer,
                "contexts": [d.page_content for d in docs],
                "ground_truth": item.get("gold_answer", "") or "",
                "expected_sections": expected,
                "retrieved_sections": retrieved_sections,
                "retrieval_hit": hit,
                "citation_ok": _citation_ok(answer, retrieved_sections),
                "retrieval_latency_ms": retrieval_ms,
                "e2e_latency_ms": e2e_ms,
                "num_chunks_retrieved": len(docs),
            }
        )

    aggregate: dict[str, Any] = {
        "pipeline": pipeline_name,
        "num_queries": len(rows),
        "hit_rate": _mean([r["retrieval_hit"] for r in rows if r["retrieval_hit"] is not None]),
        "citation_accuracy": _mean([r["citation_ok"] for r in rows]),
        "avg_retrieval_latency_ms": _mean([r["retrieval_latency_ms"] for r in rows]),
        "avg_e2e_latency_ms": _mean([r["e2e_latency_ms"] for r in rows]),
        "avg_chunks_retrieved": _mean([r["num_chunks_retrieved"] for r in rows]),
        "ragas": {},
        "ragas_error": None,
    }

    if use_ragas:
        aggregate["ragas"], aggregate["ragas_error"] = _run_ragas(rows)

    return {"rows": rows, "aggregate": aggregate}


def _mean(values: list[Any]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _run_ragas(rows: list[dict]) -> tuple[dict, str | None]:
    """Run RAGAS metrics; return (metrics_dict, error_message_or_None)."""
    try:
        install_ragas_shim()
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import answer_relevancy, faithfulness

        metrics: list[Any] = [faithfulness, answer_relevancy]

        # Ground-truth-dependent metrics only when gold answers exist.
        if any(r["ground_truth"] for r in rows):
            from ragas.metrics import answer_correctness, context_recall

            metrics += [context_recall, answer_correctness]

        dataset = Dataset.from_list(
            [
                {
                    "question": r["query"],
                    "answer": r["answer"] or " ",
                    "contexts": r["contexts"] or [" "],
                    "ground_truth": r["ground_truth"] or " ",
                }
                for r in rows
            ]
        )

        from models.embeddings import get_embeddings
        from models.llm import get_llm

        result = evaluate(
            dataset,
            metrics=metrics,
            llm=LangchainLLMWrapper(get_llm()),
            embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
            raise_exceptions=False,
            show_progress=False,
        )
        df = result.to_pandas()
        skip = {"question", "answer", "contexts", "ground_truth"}
        return {
            col: round(float(df[col].mean()), 4)
            for col in df.columns
            if col not in skip and df[col].dtype.kind in "fi"
        }, None
    except Exception as exc:  # RAGAS is best-effort; custom metrics still stand
        logger.warning("RAGAS evaluation unavailable: %s: %s", type(exc).__name__, exc)
        return {}, f"{type(exc).__name__}: {exc}"


_PIPELINE_NAMES = (
    "Pairing 1: Recursive + Parent-Child",
    "Pairing 2: Section + Hybrid",
    "Pairing 3: Advanced ISO + Graph",
)


def build_pipeline(
    name: str, documents: list
) -> tuple[Callable[[str], list], Callable[[str], str]]:
    """Build a single pipeline from already-loaded documents.

    Building one pipeline at a time keeps peak memory low: the three vector stores, the
    two BM25 indexes, and the embedding model do not comfortably co-reside in ~8 GB.
    """
    from chain.rag_chain import build_rag_chain

    if name.startswith("Pairing 1"):
        from ingestion.chunking_pair1 import create_parent_child_chunks
        from retrieval.pair1_retriever import build_parent_child_retriever

        parents, children = create_parent_child_chunks(documents)
        retriever = build_parent_child_retriever(children, parents)
        return retriever.invoke, build_rag_chain(retriever)

    if name.startswith("Pairing 2"):
        from ingestion.chunking_pair2 import create_section_based_chunks
        from retrieval.pair2_retriever import build_metadata_filtered_hybrid_retriever

        chunks = create_section_based_chunks(documents)
        retriever, _ = build_metadata_filtered_hybrid_retriever(chunks)
        return retriever.invoke, build_rag_chain(retriever)

    if name.startswith("Pairing 3"):
        from graph.builder import build_cross_reference_graph
        from ingestion.chunking_pair3 import create_advanced_iso_chunks
        from retrieval.pair3_retriever import (
            build_graph_enhanced_retriever,
            graph_expanded_retrieve,
        )

        chunks = create_advanced_iso_chunks(documents)
        graph = build_cross_reference_graph(chunks)
        retriever, vectorstore, graph = build_graph_enhanced_retriever(chunks, graph)

        def retrieve(query: str) -> list:
            return graph_expanded_retrieve(query, retriever, graph, vectorstore=vectorstore)

        return retrieve, build_rag_chain(retrieve)

    raise ValueError(f"Unknown pipeline: {name}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Evaluate the three RAG pipelines.")
    parser.add_argument("--queries", default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N queries.")
    parser.add_argument("--skip-ragas", action="store_true", help="Compute only custom metrics.")
    parser.add_argument("--use-ocr", action="store_true")
    args = parser.parse_args(argv)

    from eval.report_generator import generate_comparison_report

    queries = load_test_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    logger.info("Evaluating %d queries across 3 pipelines (ragas=%s)", len(queries), not args.skip_ragas)

    from config import ISO_PDF_DIR
    from ingestion.loader import load_iso_structures

    documents = load_iso_structures(ISO_PDF_DIR, use_ocr=args.use_ocr)

    results = []
    for name in _PIPELINE_NAMES:
        logger.info("Building and evaluating %s ...", name)
        retriever, chain = build_pipeline(name, documents)
        results.append(
            run_pipeline_evaluation(name, retriever, chain, queries, use_ragas=not args.skip_ragas)
        )
        # Release this pipeline before building the next (memory is tight).
        del retriever, chain
        gc.collect()

    path = generate_comparison_report(results, output_path=args.output)
    logger.info("Wrote comparison report to %s", path)
    print(f"Report written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())