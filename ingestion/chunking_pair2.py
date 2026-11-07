"""Pairing 2 chunking: section-based (structure-aware) chunks.

Boundaries come from Docling's structured ``section_header`` outline (see
:mod:`ingestion.structure`), so numeric clauses, annexes, and non-numeric headings are all
handled. Content is packed at block boundaries and tables are kept atomic (never split
mid-table), which prevents the giant-section / fragmented-table behaviour of the old
Markdown-regex approach.

Metadata per chunk: ``section_number``, ``section_title``, ``heading_path``, ``document_id``,
``source_file``, ``page_number`` (per chunk, from block provenance), ``content_type="text"``.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import PAIR2_CHUNK_OVERLAP, PAIR2_CHUNK_SIZE, TABLE_MAX_CHARS
from ingestion.chunk_utils import pack_units, split_table_markdown
from ingestion.structure import Section, StructuredDocument

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _table_content(block) -> str:
    """Table markdown with its caption, avoiding duplication when Docling already includes it."""
    if block.caption and block.caption not in block.text:
        return f"**{block.caption}**\n\n{block.text}"
    return block.text


def _section_units(section: Section) -> list[tuple[str, int | None, bool]]:
    """Turn a section's blocks into packable units (tables pre-split to stay bounded)."""
    units: list[tuple[str, int | None, bool]] = []
    for block in section.blocks:
        if block.kind == "table":
            for part in split_table_markdown(_table_content(block), TABLE_MAX_CHARS):
                units.append((part, block.page, True))
        else:
            units.append((block.text, block.page, False))
    return units


def create_section_based_chunks(
    structures: list[StructuredDocument],
    chunk_size: int = PAIR2_CHUNK_SIZE,
    chunk_overlap: int = PAIR2_CHUNK_OVERLAP,
) -> list[Document]:
    """Split each structured document at clause/annex boundaries and attach section metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
    )

    all_chunks: list[Document] = []

    for structured in structures:
        doc_meta = structured.document.metadata
        document_id = doc_meta.get("document_id", "doc")
        source_file = doc_meta.get("source_file")

        for section in structured.sections:
            if not section.blocks:
                continue

            heading = section.heading_line
            units = _section_units(section)
            packed = pack_units(units, chunk_size, splitter.split_text)

            for text, page in packed:
                content = f"{heading}\n\n{text}" if heading else text
                all_chunks.append(
                    Document(
                        page_content=content,
                        metadata={
                            "document_id": document_id,
                            "source_file": source_file,
                            "section_number": section.number,
                            "section_title": section.title,
                            "heading_path": section.heading_path,
                            "page_number": page if page is not None else section.page,
                            "content_type": "text",
                        },
                    )
                )

    return all_chunks
