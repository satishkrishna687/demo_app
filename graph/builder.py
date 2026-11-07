"""Cross-reference knowledge graph construction.

Builds a directed graph whose nodes are chunks and whose edges are cross-references between
them. Traversal during retrieval lets the pipeline pull in clauses that a matched clause refers
to (multi-hop reasoning), which is important for ISO documents.

Node ids include ``document_id`` so identical clause numbers across the three ISO parts do not
collide (e.g. ``ISO-15118-2:4.1.2_ab12cd34``).
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import networkx as nx
from langchain_core.documents import Document


def chunk_id(chunk: Document) -> str:
    """Stable id for a chunk: ``{document_id}:{section}_{md5(content)[:8]}``."""
    document_id = chunk.metadata.get("document_id", "doc")
    section = chunk.metadata.get("section_number", "unknown")
    content_hash = hashlib.md5(chunk.page_content.encode("utf-8")).hexdigest()[:8]
    return f"{document_id}:{section}_{content_hash}"


def build_cross_reference_graph(chunks: list[Document]) -> nx.DiGraph:
    """Build a directed graph from chunk metadata.

    Two edge types are added, both scoped to the same ``document_id`` (references are local
    to a part):

    - ``cross_reference``: ``target_section == ref`` or ``target_section.startswith(ref + ".")``.
    - ``hierarchy``: a section links to its parent section (e.g. ``8.6.2.1`` -> ``8.6.2``),
      which keeps the graph connected even when a clause has no explicit references.
    """
    graph = nx.DiGraph()

    # Index chunks by (document_id, section_number) for efficient target lookup.
    by_section: dict[tuple[str, str], list[Document]] = defaultdict(list)

    for chunk in chunks:
        node_id = chunk_id(chunk)
        graph.add_node(
            node_id,
            chunk=chunk,
            section=chunk.metadata.get("section_number"),
            document_id=chunk.metadata.get("document_id"),
        )
        by_section[
            (chunk.metadata.get("document_id"), chunk.metadata.get("section_number", "unknown"))
        ].append(chunk)

    # Bound fan-out: a short ref like "1" can otherwise match every "1.x" clause.
    max_targets_per_ref = 20

    for chunk in chunks:
        source_id = chunk_id(chunk)
        source_doc = chunk.metadata.get("document_id")
        # Prefer clause-only references; fall back to all references if absent.
        refs = (
            chunk.metadata.get("section_references")
            or chunk.metadata.get("cross_references")
            or []
        )

        for ref in refs:
            ref = str(ref)
            exact: list[Document] = []
            descendants: list[Document] = []
            for (doc_id, section), target_chunks in by_section.items():
                if doc_id != source_doc:
                    continue
                if section == ref:
                    exact.extend(target_chunks)
                elif section.startswith(ref + "."):
                    descendants.extend(target_chunks)

            targets = exact if exact else descendants[:max_targets_per_ref]
            for target in targets:
                graph.add_edge(
                    source_id,
                    chunk_id(target),
                    type="cross_reference",
                    ref=ref,
                )

    # Hierarchy edges: link each section to its parent section (e.g. 8.6.2.1 -> 8.6.2).
    for chunk in chunks:
        section = chunk.metadata.get("section_number")
        if not section or section == "unknown" or "." not in section:
            continue
        parent = section.rsplit(".", 1)[0]
        source_id = chunk_id(chunk)
        for target in by_section.get((chunk.metadata.get("document_id"), parent), []):
            graph.add_edge(source_id, chunk_id(target), type="hierarchy")

    return graph