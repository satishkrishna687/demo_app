# Project Documentation Rules (Non-Obvious Only)

- `docs/MILESTONE_N.md` files are the authoritative history of design decisions (e.g. why pipelines build one-at-a-time: docs/MILESTONE_7.md); root-level `*_REPORT.md` / `RAG_SPECIFICATION.md` are the original specs and may be outdated.
- "Pairing" means a chunking+retrieval combination, not a pairing of documents: Pairing 1 = Recursive + Parent-Child, Pairing 2 = Section + Hybrid, Pairing 3 = Advanced ISO + Graph.
- Eval results live in `eval/result_pairN.json` and `eval/report_pairN.md`; `eval/gold_set.json` is generated from `eval/gold_seeds.json` by `python -m eval.build_gold_set`.
- `report_parts/` holds the per-chapter Markdown of the academic report; `Grounded_RAG_QA_ISO15118.ipynb` is built by `scripts/build_notebook.py`, not edited directly.
- There is no `main.py`; entry points are `streamlit run ui/app.py` and the `eval` CLIs.
