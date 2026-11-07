# Project Architecture Rules (Non-Obvious Only)

- Each PDF must remain ONE `Document` with `\f` page boundaries (loader drives Docling's `DocumentConverter` directly); section chunking assumes sections never split across pages.
- Graph node ids embed `document_id` + content hash because identical clause numbers exist across the three ISO parts; cross-reference edges are scoped to one `document_id` and fan-out is capped at 20 targets per ref (short refs like "1" would otherwise match every "1.x" clause).
- The graph also adds `hierarchy` edges (child section -> parent) so the graph stays connected even for clauses with no explicit references.
- Metadata sanitisation for Chroma happens only on the copies sent to the store; original `Document` metadata stays intact for in-memory use (the graph builder depends on this).
- Embedding settings (`EMBEDDING_MODEL`, `EMBEDDING_USE_PROMPTS`, `EMBEDDING_STRIP_TABLES`) define the vector space — any change invalidates all existing Chroma indexes.
- Memory budget: one pipeline at a time; `ui/pipeline_registry.py` releases pipelines in compare mode deliberately.
