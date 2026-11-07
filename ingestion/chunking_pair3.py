"""Pairing 3 chunking: advanced ISO chunking (structure-aware).

Combines four techniques, all driven by Docling's structured outline
(:mod:`ingestion.structure`) rather than a Markdown regex:

1. **Table-first**: ``TableItem`` blocks become standalone chunks (``content_type="table"``)
   with their caption; oversized tables are split on row boundaries with the header repeated.
2. **Clause splitting**: the remaining text blocks are packed at block boundaries.
3. **Rich metadata**: section number/title, heading path, document id, per-chunk page.
4. **Cross-references**: ``see 4.2``, ``refer to Annex A`` etc. are extracted into
   ``metadata["cross_references"]`` (display) and ``metadata["section_references"]`` (graph).
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PAIR3_CHUNK_OVERLAP, PAIR3_CHUNK_SIZE, TABLE_MAX_CHARS
from ingestion.chunk_utils import pack_units, split_table_markdown
from ingestion.structure import Section, StructuredDocument

_SEPARATORS = ["\n\n", "\n", ". ", " "]

# Clause/annex references — matched against section numbers in the graph.
_SECTION_REF_PATTERNS = [
    r"see\s+(\d+(?:\.\d+)*)",
    r"refer\s+to\s+(\d+(?:\.\d+)*)",
    r"in\s+(?:clause|section|subclause)\s+(\d+(?:\.\d+)*)",
    r"as\s+(?:defined|specified|described)\s+in\s+(\d+(?:\.\d+)*)",
    r"according\s+to\s+(\d+(?:\.\d+)*)",
    r"(?:defined|specified|described)\s+in\s+(\d+(?:\.\d+)*)",
    r"clause\s+(\d+(?:\.\d+)*)",
    r"Annex\s+([A-Z])\b",
]
# Table/figure references — kept for display but NOT used to build graph edges (they are
# not clause numbers and would match unrelated sections).
_OTHER_REF_PATTERNS = [
    r"Table\s+(\d+)",
    r"Figure\s+(\d+)",
]


def _collect(text: str, patterns: list[str]) -> list[str]:
    refs: set[str] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            refs.add(match.group(1))
    return sorted(refs)


def extract_cross_references(text: str) -> list[str]:
    """Extract all clause/annex/table/figure references from text (for display)."""
    return _collect(text, _SECTION_REF_PATTERNS + _OTHER_REF_PATTERNS)


def extract_section_references(text: str) -> list[str]:
    """Extract only clause/annex references, used to build graph edges."""
    return _collect(text, _SECTION_REF_PATTERNS)


def _dedupe(chunks: list[Document]) -> list[Document]:
    """Drop exact-duplicate chunks (same document, section, type, and content)."""
    seen: set[tuple] = set()
    unique: list[Document] = []
    for chunk in chunks:
        key = (
            chunk.metadata.get("document_id"),
            chunk.metadata.get("section_number"),
            chunk.metadata.get("content_type"),
            chunk.page_content,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def _base_meta(structured: StructuredDocument, section: Section) -> dict:
    doc_meta = structured.document.metadata
    return {
        "document_id": doc_meta.get("document_id", "doc"),
        "source_file": doc_meta.get("source_file"),
        "section_number": section.number,
        "section_title": section.title,
        "heading_path": section.heading_path,
    }


def create_advanced_iso_chunks(
    structures: list[StructuredDocument],
    chunk_size: int = PAIR3_CHUNK_SIZE,
    chunk_overlap: int = PAIR3_CHUNK_OVERLAP,
    table_max_chars: int = TABLE_MAX_CHARS,
) -> list[Document]:
    """Produce table and clause chunks with rich metadata and cross-references."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=_SEPARATORS
    )

    all_chunks: list[Document] = []

    for structured in structures:
        for section in structured.sections:
            if not section.blocks:
                continue

            base = _base_meta(structured, section)
            heading = section.heading_line
            text_units: list[tuple[str, int | None, bool]] = []

            for block in section.blocks:
                if block.kind == "table":
                    content = (
                        f"**{block.caption}**\n\n{block.text}"
                        if block.caption and block.caption not in block.text
                        else block.text
                    )
                    for part in split_table_markdown(content, table_max_chars):
                        all_chunks.append(
                            Document(
                                page_content=part,
                                metadata={
                                    **base,
                                    "content_type": "table",
                                    "table_caption": block.caption or "",
                                    "page_number": block.page if block.page is not None else section.page,
                                    "cross_references": extract_cross_references(part),
                                    "section_references": extract_section_references(part),
                                },
                            )
                        )
                else:
                    text_units.append((block.text, block.page, False))

            for text, page in pack_units(text_units, chunk_size, splitter.split_text):
                content = f"{heading}\n\n{text}" if heading else text
                all_chunks.append(
                    Document(
                        page_content=content,
                        metadata={
                            **base,
                            "content_type": "text",
                            "page_number": page if page is not None else section.page,
                            "cross_references": extract_cross_references(content),
                            "section_references": extract_section_references(content),
                        },
                    )
                )

    return _dedupe(all_chunks)
