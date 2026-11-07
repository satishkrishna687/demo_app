"""Lightweight smoke tests (fast; no model downloads or pipeline builds).

Heavy end-to-end chain checks live in Milestone 14; this file asserts that the core modules
import, configuration loads, device detection works, and the metric library behaves.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval import metrics


def test_config_loads():
    import config

    assert config.LLM_PROVIDER in ("ollama", "openai")
    assert config.EMBEDDING_MODEL
    assert config.TOP_K > 0


def test_device_is_valid():
    from models.device import get_device

    assert get_device() in ("cuda", "mps", "cpu")


def test_gold_seeds_present_and_shaped():
    seeds = json.loads(Path("eval/gold_seeds.json").read_text(encoding="utf-8"))
    assert len(seeds) >= 30
    assert all({"id", "query", "category", "answer_terms"} <= set(s) for s in seeds)


def test_metrics_roundtrip():
    from langchain_core.documents import Document

    docs = [Document(page_content="SLAC setup", metadata={"section_number": "1"})]
    entry = {"answer_terms": ["SLAC"]}
    flags = metrics.relevance_flags(docs, entry)
    assert metrics.recall_at_k(flags, 1) == 1.0
    assert metrics.reciprocal_rank(flags) == 1.0
