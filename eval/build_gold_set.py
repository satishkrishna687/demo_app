"""Build the gold evaluation set from hand-authored seeds.

For each seed query we locate its ``answer_terms`` in the parsed corpus (using the section
chunks as a search index) to derive:
- ``gold_relevant_sections`` — the ``document_id:section_number`` keys containing the terms
  (a secondary, section-based relevance signal), and
- ``gold_answer`` — a short reference snippet around the first match (for correctness judging).

This is weak supervision anchored in corpus text (not in the retriever), so it is reproducible
and not circular. Seeds whose terms are not found are flagged, not dropped.

Run:  python -m eval.build_gold_set
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SEEDS_PATH = "eval/gold_seeds.json"
OUTPUT_PATH = "eval/gold_set.json"
SNIPPET_WINDOW = 200


def _find_snippet(text: str, terms: list[str]) -> tuple[str, str] | None:
    """Return (matched_term, snippet) for the first term found in text."""
    lower = text.lower()
    for term in terms:
        idx = lower.find(term.lower())
        if idx != -1:
            start = max(0, idx - SNIPPET_WINDOW // 2)
            end = min(len(text), idx + len(term) + SNIPPET_WINDOW // 2)
            return term, text[start:end].strip()
    return None


def build_gold_set() -> list[dict]:
    from config import ISO_PDF_DIR
    from ingestion.chunking_pair2 import create_section_based_chunks
    from ingestion.loader import load_iso_structures

    seeds = json.loads(Path(SEEDS_PATH).read_text(encoding="utf-8"))
    documents = load_iso_structures(ISO_PDF_DIR, use_ocr=False)
    sections = create_section_based_chunks(documents)

    gold: list[dict] = []
    matched = 0
    for seed in seeds:
        terms = [t.lower() for t in seed.get("answer_terms", [])]
        hits: list[str] = []
        snippet: str | None = None
        matched_term: str | None = None

        for chunk in sections:
            found = _find_snippet(chunk.page_content or "", terms)
            if not found:
                continue
            term, snip = found
            meta = chunk.metadata
            key = f"{meta.get('document_id')}:{meta.get('section_number')}"
            if key not in hits:
                hits.append(key)
            if snippet is None:
                snippet, matched_term = snip, term

        if hits:
            matched += 1
        entry = dict(seed)
        entry["gold_relevant_sections"] = hits
        entry["gold_answer"] = snippet or ""
        entry["matched_term"] = matched_term
        entry["matched"] = bool(hits)
        gold.append(entry)

    Path(OUTPUT_PATH).write_text(json.dumps(gold, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d/%d seeds matched to sections)", OUTPUT_PATH, matched, len(seeds))
    return gold


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    gold = build_gold_set()
    matched = sum(1 for g in gold if g.get("matched"))
    print(f"gold entries: {len(gold)} | matched to >=1 section: {matched}")
    for g in gold:
        if not g.get("matched"):
            print(f"  UNMATCHED {g['id']}: {g['query']} | terms={g.get('answer_terms')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
