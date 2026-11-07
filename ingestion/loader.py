"""Multi-PDF ingestion for the ISO 15118 corpus.

Converts every PDF in the corpus directory to Markdown with Docling (layout-aware,
tables preserved as Markdown), strips ISO headers/footers, and attaches corpus
metadata so downstream chunkers and retrievers can scope by document.

Design notes / deviations from the plan (see docs/MILESTONE_3.md):
- Docling's ``ExportType.MARKDOWN`` returns **one document per PDF with no page
  metadata**, so we drive ``DocumentConverter`` directly and derive page boundaries
  from a page-break placeholder. This keeps each PDF as a single Document, which is
  required for section-based chunking (sections must not be split across pages).
- Converted ``DoclingDocument`` objects are cached as JSON under ``DOCLING_CACHE_DIR``
  so re-runs skip the expensive conversion.
"""

from __future__ import annotations

import os

# Set before importing docling/torch (torch caches the native-JIT state at import time).
# See config.py and docs/MILESTONE_2.md.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import logging
import re
from bisect import bisect_right
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from config import DOCLING_BATCH_SIZE, DOCLING_CACHE_DIR
from ingestion.structure import StructuredDocument, build_sections

logger = logging.getLogger(__name__)

_PAGE_OF_RE = re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE)
_STANDALONE_PAGE_RE = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
_ISO_COPYRIGHT_RE = re.compile(
    r"©\s*ISO\s*\d{4}.*?All rights reserved", re.IGNORECASE | re.DOTALL
)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_PAGE_BREAK = "\f"


def derive_document_id(pdf_path: str | Path) -> str:
    """Map a filename to a short document id, e.g. ``ISO-15118-2.pdf`` -> ``ISO-15118-2``."""
    return Path(pdf_path).stem


def strip_iso_headers_footers(text: str) -> str:
    """Remove ISO page headers, footers, page numbers, and copyright lines."""
    text = _PAGE_OF_RE.sub("", text)
    text = _ISO_COPYRIGHT_RE.sub("", text)
    text = _STANDALONE_PAGE_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def _build_converter(use_ocr: bool) -> Any:
    """Create a Docling ``DocumentConverter`` configured for the detected device."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    from models.device import get_device

    opts = PdfPipelineOptions()
    opts.do_ocr = use_ocr
    opts.do_table_structure = True
    # Small batches keep peak memory low; defaults OOM on ~8 GB machines for large PDFs.
    opts.ocr_batch_size = DOCLING_BATCH_SIZE
    opts.layout_batch_size = DOCLING_BATCH_SIZE
    opts.table_batch_size = DOCLING_BATCH_SIZE

    device = {
        "cuda": AcceleratorDevice.CUDA,
        "mps": AcceleratorDevice.MPS,
    }.get(get_device(), AcceleratorDevice.CPU)
    opts.accelerator_options = AcceleratorOptions(device=device)

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def _cache_path(document_id: str) -> Path:
    return Path(DOCLING_CACHE_DIR) / f"{document_id}.json"


def _num_pages(dl_doc: Any) -> int:
    """Best-effort page count for a DoclingDocument across docling versions."""
    for attr in ("num_pages",):
        fn = getattr(dl_doc, attr, None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                pass
    pages = getattr(dl_doc, "pages", None)
    if pages is not None:
        try:
            return len(pages)
        except Exception:
            pass
    return 0


def _convert(pdf_path: Path, use_ocr: bool, converter: Any) -> Any:
    """Convert a PDF to a ``DoclingDocument``, using the JSON cache when available."""
    from docling_core.types.doc.document import DoclingDocument

    cache = _cache_path(derive_document_id(pdf_path))
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            logger.info("Loaded cached Docling document: %s", cache)
            return DoclingDocument.model_validate(data)
        except Exception as exc:  # corrupt/incompatible cache -> re-convert
            logger.warning("Cache read failed for %s (%s); re-converting", cache, exc)

    logger.info("Converting %s with Docling (ocr=%s)...", pdf_path.name, use_ocr)
    result = converter.convert(str(pdf_path))
    dl_doc = result.document

    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(dl_doc.export_to_dict()), encoding="utf-8")
        logger.info("Cached Docling document: %s", cache)
    except Exception as exc:
        logger.warning("Could not write cache %s: %s", cache, exc)

    return dl_doc


def _page_texts(dl_doc: Any) -> list[str]:
    """Return per-page Markdown, using the page-break placeholder with a fallback."""
    full_md = dl_doc.export_to_markdown(page_break_placeholder=_PAGE_BREAK)
    parts = full_md.split(_PAGE_BREAK)

    if len(parts) <= 1:
        n = _num_pages(dl_doc)
        if n > 1:
            parts = [dl_doc.export_to_markdown(page_no=i) for i in range(1, n + 1)]

    return parts


def _doc_to_document(dl_doc: Any, document_id: str, source_file: str) -> Document | None:
    """Build a single LangChain Document for a whole PDF, with page-boundary metadata."""
    parts = [strip_iso_headers_footers(p) for p in _page_texts(dl_doc)]
    if not any(parts):
        return None

    content = "\n\n".join(parts)

    # Character offsets where each page starts (page index == position in `parts`).
    breaks: list[int] = []
    pos = 0
    for part in parts[:-1]:
        pos += len(part) + 2  # +2 for the "\n\n" join separator
        breaks.append(pos)

    return Document(
        page_content=content,
        metadata={
            "document_id": document_id,
            "source_file": source_file,
            "page_number": None,  # per-chunk page resolved via page_for_offset()
            "num_pages": len(parts),
            "page_breaks": breaks,
        },
    )


def page_for_offset(metadata: dict, offset: int) -> int | None:
    """Resolve a character offset within a document to a 1-based page number."""
    breaks = metadata.get("page_breaks") or []
    if not breaks:
        return 1 if metadata.get("num_pages") else None
    return bisect_right(breaks, offset) + 1


def _build_structured(dl_doc: Any, document_id: str, source_file: str) -> StructuredDocument | None:
    """Build a :class:`StructuredDocument` (Markdown Document + section outline)."""
    doc = _doc_to_document(dl_doc, document_id, source_file)
    if doc is None:
        return None
    return StructuredDocument(document=doc, sections=build_sections(dl_doc))


def load_pdf_structured(
    pdf_path: str | Path, use_ocr: bool = False, converter: Any = None
) -> StructuredDocument | None:
    """Load a single PDF into a :class:`StructuredDocument` (or ``None`` on failure)."""
    path = Path(pdf_path)
    converter = converter or _build_converter(use_ocr)
    dl_doc = _convert(path, use_ocr, converter)
    return _build_structured(dl_doc, derive_document_id(path), path.name)


def load_pdf(pdf_path: str | Path, use_ocr: bool = False, converter: Any = None) -> list[Document]:
    """Load a single PDF into a list containing one Document (or empty on failure)."""
    structured = load_pdf_structured(pdf_path, use_ocr=use_ocr, converter=converter)
    return [structured.document] if structured else []


def load_iso_structures(pdf_dir: str = ".", use_ocr: bool = False) -> list[StructuredDocument]:
    """Load every ``*.pdf`` in ``pdf_dir`` into :class:`StructuredDocument` objects.

    Each result carries the Markdown ``document`` (with ``page_breaks`` metadata) and the
    structured ``sections`` outline used by the chunkers.

    Raises:
        FileNotFoundError: If no PDFs are found in ``pdf_dir``.
    """
    paths = sorted(Path(pdf_dir).glob("*.pdf"))
    if not paths:
        raise FileNotFoundError(f"No PDF files found in {pdf_dir!r}")

    converter = _build_converter(use_ocr)
    structures: list[StructuredDocument] = []
    for path in paths:
        structured = load_pdf_structured(path, use_ocr=use_ocr, converter=converter)
        if structured is not None:
            logger.info(
                "Ingested %s -> %d section(s)", path.name, len(structured.sections)
            )
            structures.append(structured)

    return structures


def load_iso_documents(pdf_dir: str = ".", use_ocr: bool = False) -> list[Document]:
    """Load every ``*.pdf`` in ``pdf_dir`` into LangChain Documents.

    Args:
        pdf_dir: Directory containing the ISO PDFs (default ``ISO_PDF_DIR``).
        use_ocr: Enable Docling OCR. Default False — the ISO PDFs have a text layer and
            OCR is memory-heavy (caused OOM on 8 GB machines). Enable for scanned PDFs.

    Returns:
        One Document per PDF, each with ``document_id``, ``source_file``,
        ``num_pages``, and ``page_breaks`` metadata.

    Raises:
        FileNotFoundError: If no PDFs are found in ``pdf_dir``.
    """
    return [s.document for s in load_iso_structures(pdf_dir, use_ocr=use_ocr)]
