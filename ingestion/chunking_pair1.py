"""Pairing 1 chunking: recursive parent/child chunks (baseline).

This pairing is intentionally structure-agnostic — it is the naive recursive baseline that
Pairings 2 and 3 are compared against. It splits each source document's Markdown into parent
chunks and then children, linking them by ``parent_id``.

Fixes over the original implementation:
- Degenerate (whitespace / near-empty) chunks are dropped.
- Child page numbers are resolved from the child's absolute character offset, not the parent's.
- Chunk metadata is built explicitly so heavy source metadata (``page_breaks``) is not copied
  into the vector store.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    PAIR1_CHILD_CHUNK_SIZE,
    PAIR1_CHILD_OVERLAP,
    PAIR1_PARENT_CHUNK_SIZE,
    PAIR1_PARENT_OVERLAP,
)
from ingestion.loader import page_for_offset
from ingestion.structure import StructuredDocument

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_MIN_CHUNK_CHARS = 5


def create_parent_child_chunks(
    structures: list[StructuredDocument],
    child_chunk_size: int = PAIR1_CHILD_CHUNK_SIZE,
    child_overlap: int = PAIR1_CHILD_OVERLAP,
    parent_chunk_size: int = PAIR1_PARENT_CHUNK_SIZE,
    parent_overlap: int = PAIR1_PARENT_OVERLAP,
) -> tuple[list[Document], list[Document]]:
    """Split documents into parent and child chunks with parent-child links.

    Returns:
        ``(parent_chunks, child_chunks)``. Every child's ``parent_id`` exists in the
        parent set.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=_SEPARATORS,
        add_start_index=True,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=_SEPARATORS,
        add_start_index=True,
    )

    parent_chunks: list[Document] = []
    child_chunks: list[Document] = []

    for structured in structures:
        doc = structured.document
        doc_meta = doc.metadata
        document_id = doc_meta.get("document_id", "doc")
        source_file = doc_meta.get("source_file")

        for i, parent in enumerate(parent_splitter.split_documents([doc])):
            if len(parent.page_content.strip()) < _MIN_CHUNK_CHARS:
                continue

            parent_id = f"{document_id}_parent_{i}"
            parent_start = parent.metadata.get("start_index", 0)
            parent_page = page_for_offset(doc_meta, parent_start)

            parent.metadata = {
                "parent_id": parent_id,
                "chunk_level": "parent",
                "document_id": document_id,
                "source_file": source_file,
                "page_number": parent_page,
            }
            parent_chunks.append(parent)

            for j, child in enumerate(child_splitter.split_documents([parent])):
                if len(child.page_content.strip()) < _MIN_CHUNK_CHARS:
                    continue
                child_start = child.metadata.get("start_index", 0)
                child_page = page_for_offset(doc_meta, parent_start + child_start)
                child.metadata = {
                    "parent_id": parent_id,
                    "chunk_level": "child",
                    "child_index": j,
                    "document_id": document_id,
                    "source_file": source_file,
                    "page_number": child_page,
                }
                child_chunks.append(child)

    return parent_chunks, child_chunks
