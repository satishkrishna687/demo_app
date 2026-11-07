"""Deterministic tests for the structured outline and chunking helpers (no PDF parsing)."""

from __future__ import annotations

from ingestion.chunk_utils import pack_units, split_table_markdown
from ingestion.structure import (
    Block,
    Section,
    StructuredDocument,
    build_sections,
    is_furniture_heading,
    parse_heading,
)


# --------------------------------------------------------------------------- fakes
class _Prov:
    def __init__(self, page: int) -> None:
        self.page_no = page


class _Item:
    def __init__(self, label: str, text: str = "", page: int = 1, table_md: str = "", caption: str = "") -> None:
        self.label = label
        self.text = text
        self.prov = [_Prov(page)]
        self._table_md = table_md
        self._caption = caption

    def export_to_markdown(self, doc) -> str:  # noqa: ANN001
        return self._table_md

    def caption_text(self, doc) -> str:  # noqa: ANN001
        return self._caption


class _FakeDoc:
    def __init__(self, items: list[_Item]) -> None:
        self._items = items

    def iterate_items(self):
        for item in self._items:
            yield item, 1


# --------------------------------------------------------------------------- parse_heading
def test_parse_heading_numeric_and_annex():
    assert parse_heading("4.1 Scope") == ("4.1", "Scope")
    assert parse_heading("3.1.1") == ("3.1.1", "")
    assert parse_heading("Annex A") == ("A", "")
    assert parse_heading("A.1 Relation of modes") == ("A.1", "Relation of modes")
    assert parse_heading("B.2.1") == ("B.2.1", "")
    assert parse_heading("Key") is None
    assert parse_heading("Figure 1 - Overview") is None


def test_is_furniture_heading():
    assert is_furniture_heading("© ISO 2019")
    assert is_furniture_heading("iTeh STANDARD PREVIEW")
    assert is_furniture_heading("COPYRIGHT PROTECTED DOCUMENT")
    assert is_furniture_heading("ISO 15118-2:2014(E)")
    assert is_furniture_heading("https://standards.iteh.ai/catalog")
    assert not is_furniture_heading("4.1 Scope")
    assert not is_furniture_heading("Annex A")


# --------------------------------------------------------------------------- build_sections
def test_build_sections_merges_bare_number_and_title():
    doc = _FakeDoc(
        [
            _Item("section_header", "3 Terms and definitions", page=5),
            _Item("section_header", "3.1", page=5),
            _Item("section_header", "actor", page=5),
            _Item("text", "An actor is ...", page=5),
            _Item("section_header", "Annex A", page=40),
            _Item("section_header", "(informative)", page=40),
            _Item("section_header", "Mapping of use cases", page=40),
            _Item("section_header", "A.1 Relation", page=41),
            _Item("text", "See 3.1 for details.", page=41),
        ]
    )
    sections = build_sections(doc)
    numbers = [s.number for s in sections]
    assert numbers == ["3", "3.1", "A", "A.1"]

    by_num = {s.number: s for s in sections}
    assert by_num["3.1"].title == "actor"
    assert by_num["A"].title == "(informative) Mapping of use cases"
    assert by_num["A.1"].heading_path == "A (informative) Mapping of use cases -> A.1 Relation"
    assert by_num["A.1"].page == 41


def test_build_sections_skips_furniture_and_toc():
    doc = _FakeDoc(
        [
            _Item("section_header", "© ISO 2019", page=2),
            _Item("section_header", "Contents", page=3),
            _Item("document_index", "", page=3),
            _Item("section_header", "1 Scope", page=7),
            _Item("text", "This document ...", page=7),
        ]
    )
    sections = build_sections(doc)
    assert [s.number for s in sections] == ["1"]
    assert sections[0].title == "Scope"


def test_build_sections_table_block_and_page():
    table_md = "| a | b |\n|---|---|\n| 1 | 2 |"
    doc = _FakeDoc(
        [
            _Item("section_header", "5 Tables", page=10),
            _Item("table", page=11, table_md=table_md, caption="Table 1 - Demo"),
        ]
    )
    sections = build_sections(doc)
    assert len(sections) == 1
    block = sections[0].blocks[0]
    assert block.kind == "table"
    assert block.page == 11
    assert block.caption == "Table 1 - Demo"


# --------------------------------------------------------------------------- split_table_markdown
def test_split_table_markdown_repeats_header():
    rows = "\n".join(f"| {i} | value {i} |" for i in range(200))
    md = f"**Table 1 - Demo**\n\n| id | name |\n|---|---|\n{rows}"
    parts = split_table_markdown(md, 500)
    assert len(parts) > 1
    for part in parts:
        assert "| id | name |" in part  # header repeated
        assert "|---|---|" in part
        assert len(part) <= 500 + 200


def test_split_table_markdown_small_or_non_table_unchanged():
    assert split_table_markdown("| a |\n|---|\n| 1 |", 1000) == ["| a |\n|---|\n| 1 |"]
    assert split_table_markdown("just text", 5) == ["just text"]


# --------------------------------------------------------------------------- pack_units
def test_pack_units_keeps_tables_atomic():
    units = [
        ("hello world", 1, False),
        ("| a | b |\n|---|---|\n| 1 | 2 |", 2, True),
        ("more text", 3, False),
    ]
    chunks = pack_units(units, 1000, lambda t: [t])
    assert len(chunks) == 3
    assert chunks[1][0].startswith("| a | b |")
    assert chunks[1][1] == 2


def test_pack_units_splits_oversized_text():
    units = [("x" * 100, 1, False)]
    chunks = pack_units(units, 30, lambda t: [t[i : i + 30] for i in range(0, len(t), 30)])
    assert len(chunks) == 4
    assert all(len(c[0]) <= 30 for c in chunks)


# --------------------------------------------------------------------------- chunker invariants
def _structured(sections: list[Section]) -> StructuredDocument:
    from langchain_core.documents import Document

    return StructuredDocument(
        document=Document(
            page_content="source",
            metadata={"document_id": "ISO-TEST", "source_file": "test.pdf"},
        ),
        sections=sections,
    )


def test_pair2_chunks_carry_section_metadata_and_heading():
    from ingestion.chunking_pair2 import create_section_based_chunks

    sd = _structured(
        [
            Section(
                number="4.1",
                title="Scope",
                heading_path="4 Scope -> 4.1 Scope",
                page=3,
                blocks=[Block(kind="text", text="Body text.", page=3)],
            )
        ]
    )
    chunks = create_section_based_chunks([sd])
    assert len(chunks) == 1
    meta = chunks[0].metadata
    assert meta["section_number"] == "4.1"
    assert meta["heading_path"] == "4 Scope -> 4.1 Scope"
    assert meta["page_number"] == 3
    assert chunks[0].page_content.startswith("4.1 Scope")


def test_pair3_tables_are_standalone_and_bounded():
    from config import TABLE_MAX_CHARS
    from ingestion.chunking_pair3 import create_advanced_iso_chunks

    rows = "\n".join(f"| {i} | value {i} |" for i in range(300))
    table_md = f"| id | name |\n|---|---|\n{rows}"
    sd = _structured(
        [
            Section(
                number="5",
                title="Tables",
                heading_path="5 Tables",
                page=10,
                blocks=[
                    Block(kind="text", text="Intro text.", page=10),
                    Block(kind="table", text=table_md, page=11, caption="Table 1 - Demo"),
                ],
            )
        ]
    )
    chunks = create_advanced_iso_chunks([sd])
    table_chunks = [c for c in chunks if c.metadata["content_type"] == "table"]
    assert table_chunks
    assert all("| id | name |" in c.page_content for c in table_chunks)
    assert all(len(c.page_content) <= TABLE_MAX_CHARS + 200 for c in table_chunks)
    assert all(c.metadata["section_number"] == "5" for c in chunks)
