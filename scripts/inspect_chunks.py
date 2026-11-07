"""Diagnostic: compare actual chunk output against the intended pairing design.

Loads the cached Docling documents (no PDF parsing, no embeddings, no model loads) and
reports, per pairing, the statistics needed to judge whether the chunks reflect the design:

- chunk counts and size distribution
- section coverage and distinct-section counts
- the most over-loaded sections (a sign of missed heading boundaries)
- content-type breakdown (text vs table)
- cross-reference / graph coverage (Pairing 3)
- parent-child link integrity (Pairing 1)
- page-number coverage
- page-furniture leakage (headers/footers that survived cleaning)

Run from the repo root:

    .venv/bin/python scripts/inspect_chunks.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import ISO_PDF_DIR  # noqa: E402
from ingestion.loader import load_iso_structures  # noqa: E402

# Page furniture that should NOT appear in chunks (survived header/footer stripping).
_FURNITURE_PATTERNS = [
    re.compile(r"iTeh\s+STANDARD\s+PREVIEW", re.I),
    re.compile(r"standards\.iteh\.ai", re.I),
    re.compile(r"COPYRIGHT\s+PROTECTED\s+DOCUMENT", re.I),
    re.compile(r"©\s*ISO\s*\d{4}", re.I),
    re.compile(r"^\s*ISO\s+15118-\d:\d{4}\(E\)\s*$", re.I | re.M),
    re.compile(r"^\s*Contents\s*$", re.I | re.M),
    re.compile(r"^\s*Foreword\s*$", re.I | re.M),
    re.compile(r"^\s*Introduction\s*$", re.I | re.M),
]


def _is_furniture(text: str) -> bool:
    return any(p.search(text) for p in _FURNITURE_PATTERNS)


def _size_stats(sizes: list[int]) -> str:
    if not sizes:
        return "n/a"
    sizes = sorted(sizes)
    n = len(sizes)
    return (
        f"min={sizes[0]} p50={sizes[n // 2]} p90={sizes[int(n * 0.9)]} "
        f"max={sizes[-1]} avg={sum(sizes) / n:.0f}"
    )


def _section_report(chunks: list, label: str, top: int = 15) -> None:
    print(f"\n--- {label} ---")
    print(f"chunks: {len(chunks)}")
    if not chunks:
        return

    sizes = [len(c.page_content) for c in chunks]
    print(f"size: {_size_stats(sizes)}")

    sections = [c.metadata.get("section_number") for c in chunks]
    present = [s for s in sections if s and s != "unknown"]
    print(
        f"section_number present: {len(present)}/{len(chunks)} "
        f"({100.0 * len(present) / len(chunks):.1f}%) | distinct: {len(set(present))}"
    )

    heading_paths = [c.metadata.get("heading_path") for c in chunks]
    hp = [h for h in heading_paths if h]
    print(f"heading_path present: {len(hp)}/{len(chunks)} ({100.0 * len(hp) / len(chunks):.1f}%)")

    pages = [c.metadata.get("page_number") for c in chunks]
    pg = [p for p in pages if p is not None]
    print(f"page_number present: {len(pg)}/{len(chunks)} ({100.0 * len(pg) / len(chunks):.1f}%)")

    types = Counter(c.metadata.get("content_type", "text") for c in chunks)
    print(f"content_type: {dict(types)}")

    furniture = sum(1 for c in chunks if _is_furniture(c.page_content))
    print(f"chunks containing page furniture: {furniture} ({100.0 * furniture / len(chunks):.1f}%)")

    counts = Counter(present)
    print(f"top {top} sections by chunk count:")
    for sec, n in counts.most_common(top):
        print(f"    {sec:>16}  {n}")

    giant = [(s, n) for s, n in counts.items() if n >= 100]
    print(f"sections with >=100 chunks: {len(giant)} -> {sorted(giant, key=lambda x: -x[1])[:10]}")


def main() -> None:
    documents = load_iso_structures(ISO_PDF_DIR, use_ocr=False)
    print(f"loaded {len(documents)} documents from cache")
    for sd in documents:
        d = sd.document
        print(
            f"  {d.metadata.get('document_id')}: {len(d.page_content)} chars, "
            f"{d.metadata.get('num_pages')} pages, {len(sd.sections)} sections"
        )

    # ---------------- Pairing 1 ----------------
    from ingestion.chunking_pair1 import create_parent_child_chunks

    parents, children = create_parent_child_chunks(documents)
    print("\n" + "=" * 72)
    print("PAIRING 1: Recursive + Parent-Child")
    print("=" * 72)
    print(f"parents: {len(parents)} | children: {len(children)}")
    print(f"parent size: {_size_stats([len(p.page_content) for p in parents])}")
    print(f"child size:  {_size_stats([len(c.page_content) for c in children])}")
    parent_ids = {p.metadata.get("parent_id") for p in parents}
    orphans = [c for c in children if c.metadata.get("parent_id") not in parent_ids]
    print(f"orphan children (parent_id not in parents): {len(orphans)}")
    _section_report(children, "Pairing 1 children")

    # ---------------- Pairing 2 ----------------
    from ingestion.chunking_pair2 import create_section_based_chunks

    chunks2 = create_section_based_chunks(documents)
    print("\n" + "=" * 72)
    print("PAIRING 2: Section + Hybrid")
    print("=" * 72)
    _section_report(chunks2, "Pairing 2 chunks")

    # ---------------- Pairing 3 ----------------
    from graph.builder import build_cross_reference_graph
    from ingestion.chunking_pair3 import create_advanced_iso_chunks

    chunks3 = create_advanced_iso_chunks(documents)
    print("\n" + "=" * 72)
    print("PAIRING 3: Advanced ISO + Graph")
    print("=" * 72)
    _section_report(chunks3, "Pairing 3 chunks")

    tables = [c for c in chunks3 if c.metadata.get("content_type") == "table"]
    with_caption = [c for c in tables if c.metadata.get("table_caption")]
    print(f"\ntable chunks: {len(tables)} | with caption: {len(with_caption)}")
    xref = [c for c in chunks3 if c.metadata.get("cross_references")]
    sref = [c for c in chunks3 if c.metadata.get("section_references")]
    print(f"chunks with cross_references: {len(xref)} | with section_references: {len(sref)}")

    graph = build_cross_reference_graph(chunks3)
    isolated = [n for n in graph.nodes if graph.degree(n) == 0]
    print(
        f"graph: nodes={graph.number_of_nodes()} edges={graph.number_of_edges()} "
        f"isolated={len(isolated)}"
    )


if __name__ == "__main__":
    main()

