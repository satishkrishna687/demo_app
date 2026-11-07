"""LLM-as-judge scorers for the qualitative metrics in RAG_EVAL_METRICS.md.

Rules enforced (per the spec):
- One judge model, one prompt set, ``temperature=0`` for every pairing.
- Robust parsing: the judge is asked for a single number; the first numeric token is used,
  clamped to [0, 1]. One retry on parse failure, then the score is ``None`` (flagged, not
  silently dropped).

Citation accuracy is computed deterministically (no judge) by comparing section numbers cited
in the answer against the sections actually retrieved.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
_CITATION_RE = re.compile(r"(?:Section|Clause)\s+(\d+(?:\.\d+)?(?:\.\d+)*)", re.IGNORECASE)


class LLMJudge:
    """Deterministic LLM judge with strict prompting and robust parsing."""

    def __init__(self, llm: Any = None, max_retries: int = 1) -> None:
        self._llm = llm
        self._max_retries = max_retries
        self._cache: dict[tuple[str, str], float | None] = {}

    @property
    def llm(self) -> Any:
        if self._llm is None:
            from models.llm import get_llm

            self._llm = get_llm(temperature=0.0)
        return self._llm

    def _ask(self, instruction: str) -> float | None:
        if instruction in self._cache:
            return self._cache[instruction]

        prompt = (
            "You are a strict evaluation judge. Respond with ONLY a single number between "
            "0 and 1 (for example 0.75). Do not include any words, explanation, or punctuation "
            "other than the number.\n\n"
            f"{instruction}"
        )
        score: float | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self.llm.invoke(prompt)
                text = getattr(response, "content", str(response))
                match = _NUMBER_RE.search(text or "")
                if match:
                    score = max(0.0, min(1.0, float(match.group(1))))
                    break
            except Exception as exc:  # network/model failure -> flag, don't crash the run
                logger.warning("Judge call failed (attempt %d): %s", attempt + 1, exc)

        self._cache[instruction] = score
        return score

    # ---- chunking metrics -----------------------------------------------------------------
    def relevance_density(self, query: str, chunk_text: str) -> float | None:
        """How much of the chunk is relevant to the query (0-1)."""
        return self._ask(
            "Task: rate how much of the PASSAGE is relevant to the QUESTION, from 0 (none) "
            "to 1 (entirely relevant).\n\n"
            f"QUESTION: {query}\n\nPASSAGE:\n{chunk_text[:1500]}"
        )

    def boundary_coherence(self, chunk_text: str) -> float | None:
        """1 if the chunk is self-contained, 0 if it is fragmented mid-thought/table/clause."""
        return self._ask(
            "Task: rate whether the PASSAGE is self-contained (1) or fragmented / cut off "
            "mid-sentence, mid-clause, or mid-table (0).\n\n"
            f"PASSAGE:\n{chunk_text[:1500]}"
        )

    # ---- generation metrics ---------------------------------------------------------------
    def faithfulness(self, query: str, answer: str, contexts: Sequence[str]) -> float | None:
        """Whether every claim in the answer is supported by the retrieved context."""
        joined = "\n\n---\n\n".join(c[:1200] for c in contexts[:6])
        return self._ask(
            "Task: rate how fully the ANSWER is supported by the CONTEXT, from 0 (unsupported) "
            "to 1 (fully grounded).\n\n"
            f"QUESTION: {query}\n\nCONTEXT:\n{joined}\n\nANSWER:\n{answer[:1500]}"
        )

    def answer_relevance(self, query: str, answer: str) -> float | None:
        """Whether the answer addresses the question asked."""
        return self._ask(
            "Task: rate how well the ANSWER addresses the QUESTION, from 0 (off-topic) to "
            "1 (directly answers).\n\n"
            f"QUESTION: {query}\n\nANSWER:\n{answer[:1500]}"
        )

    def answer_correctness(self, answer: str, gold_answer: str) -> float | None:
        """Semantic agreement with the gold answer."""
        if not gold_answer:
            return None
        return self._ask(
            "Task: rate how well the ANSWER agrees with the REFERENCE answer, from 0 "
            "(contradicts) to 1 (equivalent).\n\n"
            f"REFERENCE:\n{gold_answer[:1200]}\n\nANSWER:\n{answer[:1500]}"
        )


def citation_accuracy(answer: str, retrieved_sections: Sequence[str]) -> float | None:
    """Deterministic: fraction of cited section numbers that were actually retrieved.

    Returns ``None`` when the answer cites no sections (metric not applicable).
    """
    cited = _CITATION_RE.findall(answer or "")
    if not cited:
        return None
    retrieved = set(retrieved_sections)
    hits = sum(
        1
        for c in cited
        if any(c == s or s.startswith(c + ".") or c.startswith(s + ".") for s in retrieved if s)
    )
    return hits / len(cited)
