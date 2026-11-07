"""Tests for the Chunk Explorer data shaping and charts (no document parsing)."""

from __future__ import annotations


def _rec(text: str, **meta) -> dict:
    return {"text": text, "metadata": meta}


def test_chunks_to_df_columns_and_values():
    from ui.chunk_explorer import chunks_to_df

    records = [
        _rec("hello world", document_id="ISO-15118-1", section_number="1", content_type="text", page_number=3),
        _rec("| a | b |", document_id="ISO-15118-2", section_number="8.1", content_type="table",
             page_number=10, cross_references=["A"]),
    ]
    df = chunks_to_df(records)
    assert list(df["idx"]) == [0, 1]
    assert df.loc[0, "chars"] == len("hello world")
    assert df.loc[1, "content_type"] == "table"
    assert df.loc[1, "cross_refs"] == 1
    assert df.loc[0, "words"] == 2


def test_coverage_df_percentages():
    from ui.chunk_explorer import chunks_to_df, coverage_df

    records = [
        _rec("a", document_id="d", section_number="1"),
        _rec("b", document_id="d"),  # no section_number
    ]
    coverage = coverage_df(chunks_to_df(records)).set_index("Field")["Percent"]
    assert coverage["document_id"] == 100.0
    assert coverage["section_number"] == 50.0


def test_content_type_and_per_document_frames():
    from ui.chunk_explorer import chunks_to_df, content_type_df, per_document_df

    records = [
        _rec("a", document_id="d1", content_type="text"),
        _rec("b", document_id="d1", content_type="table"),
        _rec("c", document_id="d2", content_type="text"),
    ]
    df = chunks_to_df(records)

    by_type = content_type_df(df).set_index("content_type")["chunks"]
    assert by_type["text"] == 2
    assert by_type["table"] == 1

    by_doc = per_document_df(df)
    assert set(by_doc["document_id"]) == {"d1", "d2"}


def test_chunk_explorer_charts_build():
    from ui import charts
    from ui.chunk_explorer import chunks_to_df, content_type_df, coverage_df

    records = [
        _rec("x" * 10, document_id="d", content_type="text"),
        _rec("y" * 50, document_id="d", content_type="table"),
    ]
    df = chunks_to_df(records)
    charts.histogram(df, "chars", "size")
    charts.stacked_bar(content_type_df(df), "content_type", "chunks", "content_type", "by type")
    charts.percent_bar(coverage_df(df), "Field", "Percent", "coverage")