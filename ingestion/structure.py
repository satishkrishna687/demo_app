"""Structured outline derived from a Docling ``DoclingDocument``.

The Markdown export loses the document's real structure: clause numbers and titles are
sometimes split across lines, annexes are not numeric, and page furniture is rendered as
``##`` headings. This module reads Docling's *structured* items (``section_header`` labels,
``TableItem`` objects, and provenance page numbers) and turns them into an ordered list of
:class:`Section` objects, each holding its content :class:`Block` objects.

Chunkers consume these sections so that boundaries, tables, and page numbers come from the
document structure rather than a regex over Markdown. The Markdown text is still used for the
chunk *content*.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Labels that are page furniture / non-content and must never become sections or chunks.
_SKIP_LABELS = {
    "page_header",
    "page_footer",
    "document_index",  # table of contents
    "picture",
    "unspecified",
}

# Section headers that are front-matter / furniture rather than real clauses.
_FURNITURE_HEADING_RE = re.compile(
    r"^(?:"
    r"©\s*ISO|"
    r"iTeh\b|"
    r"\(standards\.iteh\.ai\)|"
    r"https?://|"
    r"standards\.iteh\.ai|"
    r"COPYRIGHT\s+PROTECTED\s+DOCUMENT|"
    r"Contents|"
    r"Foreword|"
    r"Introduction|"
    r"ISO\s+15118-\d:\d{4}|"
    r"Road\s+vehicles|"
    r"Part\s+\d+\s*:|"
    r"Véhicules|"
    r"liaison\s+de\s+données|"
    r"Second\s+edition|"
    r"Published\s+in\s+Switzerland|"
    r"All\s+rights\s+reserved|"
    r"Page\s+\d+"
    r")",
    re.IGNORECASE,
)

# Numeric clause: "4", "4.1", "4.1.2" (title optional).
_NUMERIC_RE = re.compile(r"^(\d+(?:\.\d+)*)\s*(.*)$")
# Annex heading: "Annex A" (title optional).
_ANNEX_RE = re.compile(r"^Annex\s+([A-Z])\b\s*(.*)$", re.IGNORECASE)
# Annex sub-clause: "A.1", "B.2.1" (title optional).
_ANNEX_SUB_RE = re.compile(r"^([A-Z])\.(\d+(?:\.\d+)*)\s*(.*)$")
# Annex letter alone: "A." (title optional).
_ANNEX_LETTER_RE = re.compile(r"^([A-Z])\.\s+(.*)$")


@dataclass
class Block:
    """A single content block within a section."""

    kind: str  # "text" | "list_item" | "code" | "caption" | "footnote" | "heading" | "table"
    text: str
    page: int | None = None
    caption: str | None = None


@dataclass
class Section:
    """A clause/annex section with its ordered content blocks."""

    number: str  # normalised: "4.1", "A", "A.1", or "unknown"
    title: str
    heading_path: str = ""
    page: int | None = None
    level: int = 1
    blocks: list[Block] = field(default_factory=list)

    @property
    def heading_line(self) -> str:
        """Human-readable heading, e.g. ``"4.1 Scope"`` or ``"Annex A"``."""
        return f"{self.number} {self.title}".strip()


@dataclass
class StructuredDocument:
    """A source document plus its structured section outline."""

    document: Any  # langchain_core.documents.Document (Markdown + page_breaks)
    sections: list[Section]


def _label(item: Any) -> str:
    label = getattr(item, "label", None)
    return str(getattr(label, "value", label) or "")


def _page(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    return getattr(prov[0], "page_no", None)


def is_furniture_heading(text: str) -> bool:
    """True for front-matter / page-furniture headings that are not real clauses."""
    return bool(_FURNITURE_HEADING_RE.match(text.strip()))


def parse_heading(text: str) -> tuple[str, str] | None:
    """Parse a heading into ``(normalised_number, title)`` or ``None`` if non-numeric.

    Annex headings are normalised to their letter form (``"Annex A"`` -> ``"A"``,
    ``"A.1"`` -> ``"A.1"``) so cross-reference matching works uniformly.
    """
    text = text.strip()
    if not text:
        return None

    m = _ANNEX_RE.match(text)
    if m:
        return m.group(1).upper(), m.group(2).strip()

    m = _ANNEX_SUB_RE.match(text)
    if m:
        return f"{m.group(1).upper()}.{m.group(2)}", m.group(3).strip()

    m = _NUMERIC_RE.match(text)
    if m:
        return m.group(1), m.group(2).strip()

    m = _ANNEX_LETTER_RE.match(text)
    if m:
        return m.group(1).upper(), m.group(2).strip()

    return None


def _table_markdown(item: Any, dl_doc: Any) -> str:
    try:
        return item.export_to_markdown(dl_doc).strip()
    except Exception:
        return ""


def _table_caption(item: Any, dl_doc: Any) -> str:
    try:
        return (item.caption_text(dl_doc) or "").strip()
    except Exception:
        return ""


def _depth(number: str) -> int:
    if number == "unknown":
        return 0
    return number.count(".") + 1


def _assign_heading_paths(sections: list[Section]) -> None:
    """Fill ``heading_path`` from the section hierarchy (numeric and annex alike)."""
    stack: list[Section] = []
    for section in sections:
        depth = _depth(section.number)
        while stack and _depth(stack[-1].number) >= depth:
            stack.pop()
        parts = [s.heading_line for s in stack if s.heading_line and s.number != "unknown"]
        if section.number != "unknown":
            parts.append(section.heading_line)
        section.heading_path = " -> ".join(p for p in parts if p)
        stack.append(section)


def build_sections(dl_doc: Any) -> list[Section]:
    """Build the ordered section outline from a Docling document.

    Handles Docling's habit of emitting a clause number and its title as two consecutive
    ``section_header`` items (e.g. ``"3.1.1"`` then ``"actor"``) by merging them.
    """
    sections: list[Section] = []
    current: Section | None = None
    pending: tuple[str, int, int | None] | None = None  # (number, level, page)
    pending_title: list[str] = []

    def flush_pending() -> None:
        nonlocal current, pending, pending_title
        if pending is None:
            return
        number, level, page = pending
        title = " ".join(pending_title).strip()
        current = Section(number=number, title=title, page=page, level=level, blocks=[])
        sections.append(current)
        pending = None
        pending_title = []

    def ensure_current(page: int | None, level: int) -> Section:
        nonlocal current
        if current is None:
            current = Section(number="unknown", title="", page=page, level=level, blocks=[])
            sections.append(current)
        return current

    for item, level in dl_doc.iterate_items():
        label = _label(item)
        if label in _SKIP_LABELS:
            continue
        page = _page(item)

        if label == "section_header":
            text = (getattr(item, "text", "") or "").strip()
            if not text or is_furniture_heading(text):
                continue
            parsed = parse_heading(text)
            if parsed is not None:
                number, title = parsed
                flush_pending()
                if title:
                    current = Section(number=number, title=title, page=page, level=level, blocks=[])
                    sections.append(current)
                else:
                    pending = (number, level, page)
                    pending_title = []
            elif pending is not None:
                # Non-numeric heading immediately after a bare clause number -> its title.
                pending_title.append(text)
            else:
                # A real sub-heading without a number (e.g. "Key", "Figure 1 - ...").
                ensure_current(page, level).blocks.append(
                    Block(kind="heading", text=text, page=page)
                )
            continue

        # Content item: close any pending bare-number section first.
        if pending is not None:
            flush_pending()

        if label == "table":
            md = _table_markdown(item, dl_doc)
            if not md:
                continue
            ensure_current(page, level).blocks.append(
                Block(kind="table", text=md, page=page, caption=_table_caption(item, dl_doc))
            )
            continue

        text = (getattr(item, "text", "") or "").strip()
        if text:
            ensure_current(page, level).blocks.append(Block(kind=label, text=text, page=page))

    flush_pending()
    _assign_heading_paths(sections)
    return sections
