"""Shared helpers for structure-aware chunking.

These keep tables atomic (never split mid-table) and bound oversized tables by splitting
them on row boundaries while repeating the header row, so every table chunk is self-contained.
"""

from __future__ import annotations

from typing import Callable, Sequence

# A unit is (text, page, is_table).
Unit = tuple[str, int | None, bool]


def _is_separator_row(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= set("|-: ")


def split_table_markdown(md: str, max_chars: int) -> list[str]:
    """Split a Markdown table into parts no larger than ``max_chars``.

    The table header (and separator row, when present) is repeated in every part so each
    chunk is a valid, self-contained table. Any caption/prefix lines before the table are
    also repeated. Returns ``[md]`` unchanged when it already fits or contains no table.
    """
    if len(md) <= max_chars:
        return [md]

    lines = md.splitlines()
    # Locate the first table row (a caption may precede it).
    header_idx = next((i for i, line in enumerate(lines) if line.lstrip().startswith("|")), None)
    if header_idx is None:
        return [md]

    has_separator = header_idx + 1 < len(lines) and _is_separator_row(lines[header_idx + 1])
    body_start = header_idx + 2 if has_separator else header_idx + 1
    prefix = "\n".join(lines[:body_start])
    body = lines[body_start:]
    if not body:
        return [md]

    parts: list[str] = []
    current: list[str] = []
    current_len = len(prefix) + 1

    for row in body:
        row_len = len(row) + 1
        if current and current_len + row_len > max_chars:
            parts.append(prefix + "\n" + "\n".join(current))
            current = []
            current_len = len(prefix) + 1
        current.append(row)
        current_len += row_len

    if current:
        parts.append(prefix + "\n" + "\n".join(current))
    return parts or [md]


def pack_units(
    units: Sequence[Unit],
    max_chars: int,
    splitter: Callable[[str], list[str]],
) -> list[tuple[str, int | None]]:
    """Greedily pack ``(text, page, is_table)`` units into chunks of <= ``max_chars``.

    - Table units are atomic: they are never merged with other units and never split here
      (callers split oversized tables with :func:`split_table_markdown` beforehand).
    - Oversized non-table units are split with ``splitter`` (a ``str -> list[str]`` callable).

    Returns a list of ``(chunk_text, page)`` where ``page`` is the page of the first unit.
    """
    chunks: list[tuple[str, int | None]] = []
    buffer: list[str] = []
    buffer_len = 0
    buffer_page: int | None = None

    def flush() -> None:
        nonlocal buffer, buffer_len, buffer_page
        if buffer:
            chunks.append(("\n\n".join(buffer), buffer_page))
        buffer = []
        buffer_len = 0
        buffer_page = None

    for text, page, is_table in units:
        text = text.strip()
        if not text:
            continue

        if is_table:
            flush()
            chunks.append((text, page))
            continue

        if len(text) > max_chars:
            flush()
            for piece in splitter(text):
                piece = piece.strip()
                if piece:
                    chunks.append((piece, page))
            continue

        if buffer and buffer_len + len(text) + 2 > max_chars:
            flush()
        if not buffer:
            buffer_page = page
        buffer.append(text)
        buffer_len += len(text) + 2

    flush()
    return chunks
