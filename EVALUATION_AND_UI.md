# Extension to Implementation Plan: Evaluation Report & UI Interface

This document extends the main implementation plan with two additional workstreams:

1. **Comparison Report** — a methodology and template for benchmarking all three chunking + retrieval strategies.
2. **UI Interface** — a plan for a Streamlit-based Q&A interface to interact with the pipelines.

---

## Part A: Comparison Report for Chunking + Retrieval Strategies

### A.1 Purpose

Produce a reproducible report comparing the three pipelines on the same set of queries, using both automated metrics and qualitative analysis. The report demonstrates which pairing is "best for ISO documents" with evidence.

### A.2 Evaluation Framework

#### A.2.1 Test Query Set

Create `eval/test_queries.json` with 20–30 queries spanning these categories:

```json
[
  {
    "id": "q01",
    "query": "What is the maximum AC charging voltage specified?",
    "category": "parameter_lookup",
    "expected_section": "8.3.2",
    "expected_content_type": "table",
    "gold_answer": "..."
  },
  {
    "id": "q02",
    "query": "Describe the SLAC communication setup procedure.",
    "category": "procedure",
    "expected_section": "4.5",
    "expected_content_type": "text",
    "gold_answer": "..."
  },
  {
    "id": "q03",
    "query": "What clauses does Section 4.2 reference?",
    "category": "cross_reference",
    "expected_sections": ["4.2.1", "4.3", "Annex A"],
    "expected_content_type": "text",
    "gold_answer": "..."
  },
  {
    "id": "q04",
    "query": "What does Annex A say about supported message types?",
    "category": "annex_lookup",
    "expected_section": "Annex A",
    "gold_answer": "..."
  }
]
```

**Query categories** (aim for balanced coverage):

| Category | Purpose | Tests |
|---|---|---|
| `parameter_lookup` | Find specific numeric values | Table chunking, exact-term retrieval |
| `procedure` | Describe multi-step processes | Semantic chunking, contextual retrieval |
| `cross_reference` | Multi-hop reasoning | Graph expansion (Pairing 3 only) |
| `annex_lookup` | Non-numeric section navigation | Metadata filtering |
| `definition` | Term/concept lookup | Hybrid keyword + vector |
| `comparison` | Compare two things across clauses | Multi-section retrieval |

#### A.2.2 Evaluation Metrics

Use **RAGAS** (`pip install ragas`) for automated evaluation. This is the industry-standard open-source framework for RAG evaluation.

Add to `requirements.txt`:

```txt
ragas>=0.2.0
datasets>=2.20.0
pandas>=2.0
```

**Metrics to compute per pipeline:**

| Metric | What It Measures | Range | Target |
|---|---|---|---|
| **Faithfulness** | Answer grounded in retrieved context | 0–1 | > 0.85 |
| **Answer Relevance** | Answer addresses the question | 0–1 | > 0.80 |
| **Context Precision** | Retrieved chunks are relevant | 0–1 | > 0.75 |
| **Context Recall** | All needed info was retrieved | 0–1 | > 0.70 |
| **Answer Correctness** | Answer matches gold answer (semantic) | 0–1 | > 0.75 |

**Additional custom metrics:**

| Metric | How to Compute |
|---|---|
| **Retrieval Hit Rate** | % of queries where `expected_section` appears in top-k retrieved chunks |
| **Citation Accuracy** | % of answers that cite the correct section number |
| **Latency (retrieval)** | Time for `retriever.invoke(query)` in ms |
| **Latency (end-to-end)** | Time for `chain.invoke(query)` in ms |
| **Chunk Count** | Total chunks indexed (proxy for storage cost) |
| **Avg Chunk Size** | Tokens per chunk (proxy for embedding cost) |

### A.3 Implementation: Evaluation Module

**Add module `eval/evaluator.py`:**

```python
import time
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Callable, List, Dict


def run_pipeline_evaluation(
    pipeline_name: str,
    retriever,
    rag_chain,
    test_queries_path: str = "eval/test_queries.json",
) -> Dict:
    """
    Run a full evaluation for one pipeline.
    Returns a dict with per-query results and aggregate metrics.
    """
    with open(test_queries_path) as f:
        queries = json.load(f)

    rows = []
    for item in queries:
        q = item["query"]

        # Measure retrieval latency
        t0 = time.perf_counter()
        retrieved_docs = retriever.invoke(q)
        retrieval_latency_ms = (time.perf_counter() - t0) * 1000

        # Measure end-to-end latency
        t0 = time.perf_counter()
        answer = rag_chain.invoke(q)
        e2e_latency_ms = (time.perf_counter() - t0) * 1000

        # Check retrieval hit
        retrieved_sections = [
            d.metadata.get("section_number", "") for d in retrieved_docs
        ]
        expected = item.get("expected_section") or item.get("expected_sections", [])
        if isinstance(expected, str):
            expected = [expected]
        hit = any(
            any(exp in sec for sec in retrieved_sections)
            for exp in expected
        )

        rows.append({
            "query_id": item["id"],
            "query": q,
            "category": item["category"],
            "answer": answer,
            "contexts": [d.page_content for d in retrieved_docs],
            "ground_truth": item.get("gold_answer", ""),
            "retrieval_hit": hit,
            "retrieved_sections": retrieved_sections,
            "retrieval_latency_ms": retrieval_latency_ms,
            "e2e_latency_ms": e2e_latency_ms,
            "num_chunks_retrieved": len(retrieved_docs),
        })

    # RAGAS evaluation
    ragas_dataset = Dataset.from_list([
        {
            "question": r["query"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ])

    ragas_result = evaluate(
        ragas_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_correctness,
        ],
    )

    # Aggregate custom metrics
    aggregate = {
        "pipeline": pipeline_name,
        "hit_rate": sum(r["retrieval_hit"] for r in rows) / len(rows),
        "avg_retrieval_latency_ms": sum(r["retrieval_latency_ms"] for r in rows) / len(rows),
        "avg_e2e_latency_ms": sum(r["e2e_latency_ms"] for r in rows) / len(rows),
        "avg_chunks_retrieved": sum(r["num_chunks_retrieved"] for r in rows) / len(rows),
        **{k: round(v, 4) for k, v in ragas_result.items()},
    }

    return {"rows": rows, "aggregate": aggregate}
```

**Add module `eval/report_generator.py`:**

```python
import pandas as pd
from typing import List, Dict


def generate_comparison_report(results: List[Dict], output_path: str = "eval/report.md"):
    """
    results: list of dicts, one per pipeline (output of run_pipeline_evaluation).
    Writes a Markdown comparison report.
    """
    aggregates = pd.DataFrame([r["aggregate"] for r in results])

    # Per-category hit rate
    category_stats = {}
    for r in results:
        df = pd.DataFrame(r["rows"])
        stats = df.groupby("category")["retrieval_hit"].mean().to_dict()
        category_stats[r["aggregate"]["pipeline"]] = stats
    category_df = pd.DataFrame(category_stats).fillna(0).round(2)

    # Build markdown
    md = []
    md.append("# Chunking + Retrieval Strategy Comparison Report\n")
    md.append("## Executive Summary\n")
    md.append(_write_summary(results))
    md.append("\n## Aggregate Metrics\n")
    md.append(aggregates.to_markdown(index=False))
    md.append("\n## Hit Rate by Query Category\n")
    md.append(category_df.to_markdown())
    md.append("\n## Per-Pipeline Details\n")
    for r in results:
        md.append(_write_pipeline_details(r))
    md.append("\n## Recommendations\n")
    md.append(_write_recommendations(results))

    with open(output_path, "w") as f:
        f.write("\n".join(md))
    return output_path


def _write_summary(results):
    best = max(results, key=lambda r: r["aggregate"]["faithfulness"])
    return (
        f"Evaluated **{len(results)} pipelines** across "
        f"**{len(results[0]['rows'])} queries**. "
        f"Highest faithfulness: **{best['aggregate']['pipeline']}** "
        f"({best['aggregate']['faithfulness']:.3f})."
    )


def _write_pipeline_details(r):
    agg = r["aggregate"]
    return f"""
### {agg['pipeline']}

| Metric | Value |
|---|---|
| Hit Rate | {agg['hit_rate']:.3f} |
| Faithfulness | {agg['faithfulness']:.3f} |
| Answer Relevance | {agg['answer_relevancy']:.3f} |
| Context Precision | {agg['context_precision']:.3f} |
| Context Recall | {agg['context_recall']:.3f} |
| Answer Correctness | {agg['answer_correctness']:.3f} |
| Avg Retrieval Latency | {agg['avg_retrieval_latency_ms']:.0f} ms |
| Avg End-to-End Latency | {agg['avg_e2e_latency_ms']:.0f} ms |
| Avg Chunks Retrieved | {agg['avg_chunks_retrieved']:.1f} |
"""


def _write_recommendations(results):
    df = pd.DataFrame([r["aggregate"] for r in results])
    best_quality = df.loc[df["faithfulness"].idxmax(), "pipeline"]
    best_speed = df.loc[df["avg_e2e_latency_ms"].idxmin(), "pipeline"]
    best_hit = df.loc[df["hit_rate"].idxmax(), "pipeline"]
    return f"""
- **Best answer quality**: {best_quality}
- **Fastest pipeline**: {best_speed}
- **Best retrieval precision**: {best_hit}
- **Recommendation for ISO 15118**: Use Pairing 3 for accuracy-critical queries; Pairing 1 for latency-sensitive interactive use.
"""
```

### A.4 Report Structure (Final Output)

The generated `eval/report.md` should include:

1. **Executive Summary** — 3-sentence verdict.
2. **Methodology** — dataset size, query categories, metrics used, model versions.
3. **Aggregate Comparison Table** — all pipelines side-by-side.
4. **Hit Rate by Category** — shows which pairing wins for tables vs. cross-references vs. procedures.
5. **Latency vs. Quality Trade-off** — a scatter table showing quality scores against latency.
6. **Per-Pipeline Detail Sections** — one section per pipeline.
7. **Qualitative Examples** — 3–5 example queries with answers from each pipeline, side-by-side, so reviewers see the difference.
8. **Recommendations** — which pipeline for which use case.
9. **Reproducibility Appendix** — exact model versions, embedding dimension, chunk sizes, query set hash.

### A.5 Expected Outcomes (Predictions)

Based on your chunking design, the expected ranking should look like:

| Query Category | Likely Winner | Reason |
|---|---|---|
| `parameter_lookup` (tables) | Pairing 3 | Table-first chunking + BM25 exact-match |
| `procedure` | Pairing 2 or 3 | Clause-boundary chunks give coherent procedures |
| `cross_reference` | Pairing 3 | Only pipeline with graph expansion |
| `annex_lookup` | Pairing 2 | Metadata filter on `section_number` |
| `definition` | Pairing 2 | Hybrid search handles exact ISO terms |
| `comparison` | Pairing 3 | Multi-clause retrieval via graph |

**Overall**: Pairing 3 wins on quality; Pairing 1 wins on latency and simplicity; Pairing 2 wins on cost-effectiveness.

### A.6 Running the Evaluation

Add to `main.py`:

```python
from eval.evaluator import run_pipeline_evaluation
from eval.report_generator import generate_comparison_report

# After building all three pipelines and chains:
results = [
    run_pipeline_evaluation("Pairing 1: Recursive + Parent-Child", retriever1, chain1),
    run_pipeline_evaluation("Pairing 2: Section + Hybrid", retriever2, chain2),
    run_pipeline_evaluation("Pairing 3: Advanced ISO + Graph", retriever3, chain3),
]
generate_comparison_report(results, output_path="eval/report.md")
print("Report written to eval/report.md")
```

---

## Part B: UI Interface for Q&A

### B.1 Framework Choice: Streamlit

**Why Streamlit:**
- Zero frontend code — pure Python.
- Native integration with LangChain.
- Built-in chat components (`st.chat_message`, `st.chat_input`).
- Sidebar for config, main panel for chat.
- Deployable as a single file or via Streamlit Cloud.
- Fastest path to a working UI in your time budget.

**Alternative (mention in report)**: Gradio is equally simple if you prefer a different aesthetic. FastAPI + React is overkill for this project.

Add to `requirements.txt`:

```txt
streamlit>=1.38.0
```

### B.2 UI Feature Set

| Feature | Description |
|---|---|
| **Pipeline selector** | Sidebar dropdown to switch between Pairing 1 / 2 / 3 |
| **Chat interface** | Multi-turn Q&A with conversation history |
| **Source citations** | Expandable panel under each answer showing retrieved chunks |
| **Metadata display** | Each source shows section number, page, content type |
| **Latency display** | Show retrieval + generation time per answer |
| **Query filter** | Optional: filter by `section_number` before retrieval |
| **Export chat** | Button to download conversation as Markdown |
| **Comparison mode** | Send the same query to all three pipelines simultaneously, show results side-by-side |

### B.3 File Structure Addition

```
iso-rag-project/
├── ui/
│   ├── __init__.py
│   ├── app.py                    # Streamlit main app
│   ├── pipeline_registry.py      # Loads/caches all three pipelines
│   └── components.py             # Reusable UI components (chat, sources, etc.)
```

### B.4 Module Specifications

#### `ui/pipeline_registry.py`

**Purpose**: Build all three pipelines once and cache them so the UI doesn't re-embed on every session.

```python
import streamlit as st
from ingestion.loader import load_iso_document
from ingestion.chunking_pair1 import create_parent_child_chunks
from ingestion.chunking_pair2 import create_section_based_chunks
from ingestion.chunking_pair3 import create_advanced_iso_chunks
from retrieval.pair1_retriever import build_parent_child_retriever
from retrieval.pair2_retriever import build_metadata_filtered_hybrid_retriever
from retrieval.pair3_retriever import (
    build_graph_enhanced_retriever,
    graph_expanded_retrieve,
)
from graph.builder import build_cross_reference_graph
from chain.rag_chain import build_rag_chain


@st.cache_resource(show_spinner="Building pipelines...")
def load_all_pipelines(pdf_path: str):
    """
    Build all three pipelines once. Cached across Streamlit sessions.
    Returns dict: {name: {"retriever": ..., "chain": ..., "graph": ..., "vectorstore": ...}}
    """
    docs = load_iso_document(pdf_path, use_ocr=True)

    pipelines = {}

    # Pairing 1
    parent_chunks, child_chunks = create_parent_child_chunks(docs)
    r1 = build_parent_child_retriever(child_chunks, parent_chunks)
    pipelines["Pairing 1: Recursive + Parent-Child"] = {
        "retriever": r1,
        "chain": build_rag_chain(r1),
        "graph": None,
        "vectorstore": None,
    }

    # Pairing 2
    chunks2 = create_section_based_chunks(docs)
    r2, vs2 = build_metadata_filtered_hybrid_retriever(chunks2)
    pipelines["Pairing 2: Section + Hybrid"] = {
        "retriever": r2,
        "chain": build_rag_chain(r2),
        "graph": None,
        "vectorstore": vs2,
    }

    # Pairing 3
    chunks3 = create_advanced_iso_chunks(docs)
    graph = build_cross_reference_graph(chunks3)
    r3, vs3, graph = build_graph_enhanced_retriever(chunks3, graph)

    def graph_retriever(query):
        return graph_expanded_retrieve(query, r3, graph)

    pipelines["Pairing 3: Advanced ISO + Graph"] = {
        "retriever": graph_retriever,
        "chain": build_rag_chain(graph_retriever),
        "graph": graph,
        "vectorstore": vs3,
    }

    return pipelines
```

#### `ui/components.py`

**Purpose**: Reusable UI building blocks.

```python
import streamlit as st
import time
from typing import List


def render_source_panel(docs: List, latency_ms: float = None):
    """
    Render an expandable panel with retrieved chunks and their metadata.
    """
    with st.expander(f"📚 Sources ({len(docs)} chunks)", expanded=False):
        if latency_ms is not None:
            st.caption(f"Retrieval latency: {latency_ms:.0f} ms")

        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            header = (
                f"**{i}. Section {meta.get('section_number', 'N/A')}** "
                f"— {meta.get('section_title', '')} "
                f"*(type: {meta.get('content_type', 'text')}, "
                f"page: {meta.get('page_number', '?')})*"
            )
            st.markdown(header)

            # Truncate long chunks for display
            content = doc.page_content
            if len(content) > 800:
                content = content[:800] + "\n\n... *(truncated)*"
            st.markdown(f"```\n{content}\n```")

            if meta.get("cross_references"):
                st.caption(f"🔗 Cross-references: {', '.join(meta['cross_references'])}")

            st.divider()


def render_answer(answer: str, latency_ms: float = None):
    """Render the LLM answer with optional latency badge."""
    st.markdown(answer)
    if latency_ms is not None:
        st.caption(f"⏱️ Generated in {latency_ms:.0f} ms")


def render_chat_history(messages: List[dict]):
    """Render the full conversation."""
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                render_source_panel(msg["sources"], msg.get("latency_ms"))


def export_chat_button(messages: List[dict]):
    """Provide a download button for the conversation as Markdown."""
    md_lines = ["# ISO RAG Chat Export\n"]
    for msg in messages:
        md_lines.append(f"## {msg['role'].capitalize()}\n")
        md_lines.append(msg["content"] + "\n")
        for src in msg.get("sources", []):
            md_lines.append(
                f"- Section {src.metadata.get('section_number', 'N/A')}: "
                f"{src.metadata.get('section_title', '')}\n"
            )
    st.download_button(
        "📥 Export chat",
        data="\n".join(md_lines),
        file_name="chat_export.md",
        mime="text/markdown",
    )
```

#### `ui/app.py`

**Purpose**: Main Streamlit app.

```python
import streamlit as st
import time
from ui.pipeline_registry import load_all_pipelines
from ui.components import (
    render_chat_history,
    render_source_panel,
    export_chat_button,
)
from config import ISO_PDF_PATH


st.set_page_config(
    page_title="ISO RAG Q&A",
    page_icon="📘",
    layout="wide",
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Configuration")

    pipelines = load_all_pipelines(ISO_PDF_PATH)
    pipeline_names = list(pipelines.keys())

    selected_pipeline = st.selectbox(
        "Pipeline",
        options=pipeline_names,
        index=2,  # default to Pairing 3
        help="Choose which chunking + retrieval strategy to use.",
    )

    mode = st.radio(
        "Mode",
        options=["Single pipeline", "Compare all three"],
        index=0,
    )

    st.divider()

    if selected_pipeline == "Pairing 2: Section + Hybrid":
        section_filter = st.text_input(
            "Filter by section (optional)",
            placeholder="e.g., 4.1",
        )
    else:
        section_filter = None

    st.divider()

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    if "messages" in st.session_state:
        export_chat_button(st.session_state.messages)


# ---------- Main ----------
st.title("📘 ISO 15118 RAG Q&A")
st.caption(f"Active pipeline: **{selected_pipeline}** | Mode: **{mode}**")

if "messages" not in st.session_state:
    st.session_state.messages = []

render_chat_history(st.session_state.messages)

user_input = st.chat_input("Ask a question about the ISO specification...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if mode == "Single pipeline":
        pipeline = pipelines[selected_pipeline]

        # Retrieval
        t0 = time.perf_counter()
        if section_filter and pipeline.get("vectorstore"):
            from retrieval.pair2_retriever import filtered_retrieve
            sources = filtered_retrieve(
                pipeline["vectorstore"], user_input, section_filter
            )
        else:
            sources = pipeline["retriever"].invoke(user_input)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        # Generation
        t0 = time.perf_counter()
        answer = pipeline["chain"].invoke(user_input)
        e2e_ms = (time.perf_counter() - t0) * 1000

        with st.chat_message("assistant"):
            st.markdown(answer)
            st.caption(f"⏱️ Retrieval: {retrieval_ms:.0f} ms | Total: {e2e_ms:.0f} ms")
            render_source_panel(sources, retrieval_ms)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "latency_ms": e2e_ms,
        })

    else:  # Compare all three
        with st.chat_message("assistant"):
            st.markdown(f"### Comparing all three pipelines for: *{user_input}*")
            for name, pipeline in pipelines.items():
                st.markdown(f"---\n#### 🔹 {name}")
                t0 = time.perf_counter()
                sources = pipeline["retriever"].invoke(user_input)
                retrieval_ms = (time.perf_counter() - t0) * 1000

                t0 = time.perf_counter()
                answer = pipeline["chain"].invoke(user_input)
                e2e_ms = (time.perf_counter() - t0) * 1000

                st.markdown(answer)
                st.caption(f"⏱️ Retrieval: {retrieval_ms:.0f} ms | Total: {e2e_ms:.0f} ms")
                render_source_panel(sources, retrieval_ms)

        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Compared all 3 pipelines for: {user_input}",
            "sources": [],
        })
```

### B.5 Running the UI

Add to project root:

```bash
streamlit run ui/app.py
```

Open browser at `http://localhost:8501`.

### B.6 UI Flow Diagram

```
┌───────────────────────────────────────────────────────────┐
│  Sidebar                     │  Main Panel                 │
│  ─────────                   │  ──────────                 │
│  • Pipeline selector         │  📘 Title + active config   │
│  • Mode (single/compare)     │                             │
│  • Section filter (Pairing 2)│  Chat history               │
│  • Clear chat                │    ├─ user message          │
│  • Export chat               │    └─ assistant message     │
│                              │        + sources expander   │
│                              │        + latency badge      │
│                              │                             │
│                              │  [ Chat input box ]         │
└───────────────────────────────────────────────────────────┘
```

### B.7 Optional Enhancements (Post-MVP)

| Enhancement | Effort | Value |
|---|---|---|
| **Streaming answers** | Low | Replace `chain.invoke` with `chain.stream` and `st.write_stream` |
| **Session persistence** | Low | Save chat to `st.session_state` + local JSON |
| **Feedback buttons** | Low | 👍/👎 per answer, log to CSV for evaluation |
| **PDF viewer** | Medium | Embed `st.pdf` next to sources, jump to page |
| **Admin dashboard** | Medium | Show graph visualization (`st_networkx`) and chunk stats |
| **Multi-document support** | Medium | Upload new PDFs via sidebar, rebuild index |
| **Authentication** | Medium | `streamlit-authenticator` if deploying publicly |

### B.8 Deployment Options

| Option | Command | Notes |
|---|---|---|
| **Local** | `streamlit run ui/app.py` | For development/demo |
| **Docker** | Build image with `python:3.11-slim` + deps | For reproducibility |
| **Streamlit Community Cloud** | Push to GitHub, connect repo | Free, public |
| **Hugging Face Spaces** | Use Streamlit SDK | Free, GPU options for local models |

**Docker file (optional):**

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## Part C: Updated Milestones

Extend Section 6 of the main plan with two new milestones:

### ✅ Milestone 7: Evaluation Harness
- Create `eval/test_queries.json` with 20–30 queries.
- Implement `eval/evaluator.py` and `eval/report_generator.py`.
- Add `ragas`, `datasets`, `pandas` to `requirements.txt`.
- Run evaluation, confirm `eval/report.md` is generated.

### ✅ Milestone 8: Streamlit UI
- Add `streamlit` to `requirements.txt`.
- Implement `ui/pipeline_registry.py`, `ui/components.py`, `ui/app.py`.
- Launch UI, verify:
  - Pipeline selector works.
  - Chat Q&A returns answers with sources.
  - "Compare all three" mode shows side-by-side results.
  - Export chat produces a valid Markdown file.

### ✅ Milestone 9: Final Deliverables
- `eval/report.md` (comparison report).
- Screenshots of UI (single mode + compare mode).
- `README.md` with setup + run instructions.
- `.env.example` template.

---

## Part D: Updated Acceptance Criteria

Add to Section 8:

- [ ] `eval/report.md` generated with all three pipelines compared on ≥ 20 queries.
- [ ] All five RAGAS metrics computed for each pipeline.
- [ ] Streamlit UI runs at `localhost:8501` without errors.
- [ ] UI supports single-pipeline and compare-all-three modes.
- [ ] Every UI answer shows source chunks with section metadata.
- [ ] Chat export produces valid Markdown.

---

## Part E: Repository Layout (Final)

```
iso-rag-project/
├── .env.example
├── requirements.txt
├── config.py
├── main.py                           # CLI demo
├── README.md
├── ingestion/
│   ├── loader.py
│   ├── chunking_pair1.py
│   ├── chunking_pair2.py
│   └── chunking_pair3.py
├── graph/
│   └── builder.py
├── retrieval/
│   ├── pair1_retriever.py
│   ├── pair2_retriever.py
│   └── pair3_retriever.py
├── chain/
│   └── rag_chain.py
├── eval/
│   ├── test_queries.json
│   ├── evaluator.py
│   ├── report_generator.py
│   └── report.md                     # generated
├── ui/
│   ├── app.py
│   ├── pipeline_registry.py
│   └── components.py
├── data/
│   └── iso_15118.pdf
└── tests/
    └── smoke_test.py
```

---

**END OF EXTENSION**

> **Instructions to coding LLM**: After completing Milestones 1–6 from the main plan, proceed with Milestones 7–9 defined here. Implement `eval/` first (it does not depend on the UI), then `ui/`. Ensure all CLI and UI code paths invoke the *same* pipeline builders so results are consistent. Use `@st.cache_resource` for expensive pipeline initialization. Do not silently swallow exceptions — surface them in the UI via `st.error`.