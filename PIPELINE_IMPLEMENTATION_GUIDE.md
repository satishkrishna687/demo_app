# Detailed Implementation Guide: RAG Pipeline for ISO Documents with LangChain

This guide walks you through building three progressively sophisticated RAG pipelines, each pairing a chunking strategy with a retrieval strategy. All components use LangChain as the orchestration framework.

---

## 📁 Project Structure

```
iso-rag-project/
├── .env                          # API keys, config
├── requirements.txt
├── config.py                     # Shared configuration
├── ingestion/
│   ├── __init__.py
│   ├── loader.py                 # PDF loading + parsing
│   ├── chunking_pair1.py         # Recursive + Parent-Child
│   ├── chunking_pair2.py         # Section-Based + Metadata
│   ├── chunking_pair3.py         # Advanced ISO Chunking
│   └── metadata_extractor.py     # ISO-specific metadata extraction
├── retrieval/
│   ├── __init__.py
│   ├── pair1_retriever.py        # Parent-Child Retriever
│   ├── pair2_retriever.py        # Metadata-Filtered Hybrid Search
│   └── pair3_retriever.py        # Graph-Enhanced Hybrid Retrieval
├── graph/
│   ├── __init__.py
│   ├── builder.py                # Knowledge graph construction
│   └── traverser.py              # Cross-reference traversal
├── vectorstore/
│   └── store.py                  # Chroma vector store setup
├── chain/
│   ├── __init__.py
│   └── rag_chain.py              # LangChain RAG chain assembly
└── main.py                       # Entry point
```

---

## 🔧 Step 0: Environment Setup

**requirements.txt:**

```txt
# Core LangChain
langchain>=0.3.0
langchain-core>=0.3.0
langchain-community>=0.3.0
langchain-chroma>=0.2.0
langchain-openai>=0.2.0

# PDF Parsing (choose one or more)
docling>=2.0.0
langchain-docling>=0.1.0
langchain-pymupdf4llm>=0.1.0
pymupdf4llm>=0.1.0

# Retrieval
rank_bm25>=0.2.2
chromadb>=0.5.0
networkx>=3.0              # For knowledge graph prototype
neo4j>=5.0                 # For production graph (optional)

# Embeddings & LLM
sentence-transformers>=3.0  # For local embedding models
openai>=1.0                 # If using OpenAI embeddings

# Utilities
python-dotenv>=1.0
pydantic>=2.0
```

**config.py:**

```python
import os
from dotenv import load_dotenv

load_dotenv()

# LLM & Embedding Config
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chunking Config
PAIR1_CHILD_CHUNK_SIZE = 256
PAIR1_CHILD_OVERLAP = 25
PAIR1_PARENT_CHUNK_SIZE = 1024
PAIR1_PARENT_OVERLAP = 100

PAIR2_CHUNK_SIZE = 512
PAIR2_CHUNK_OVERLAP = 50

# Retrieval Config
TOP_K = 5
HYBRID_WEIGHTS = [0.6, 0.4]  # [vector_weight, bm25_weight]

# Vector Store
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "iso_15118"
```

---

## 📥 Step 1: Document Ingestion (Shared)

All three pairings share a common ingestion pipeline that parses the ISO PDF into structured documents.

**ingestion/loader.py:**

```python
from langchain_docling import DoclingLoader
from docling.datamodel.pipeline_options import PipelineOptions
from langchain_core.documents import Document
from typing import List

def load_iso_document(pdf_path: str, use_ocr: bool = True) -> List[Document]:
    """
    Load an ISO PDF using Docling for high-fidelity parsing.
    Docling preserves layout, reading order, and table structures.
    """
    pipeline_options = PipelineOptions()
    pipeline_options.do_ocr = use_ocr
    pipeline_options.do_table_structure = True

    loader = DoclingLoader(
        file_path=pdf_path,
        pipeline_options=pipeline_options,
    )
    documents = loader.load()

    # Docling outputs Markdown with tables preserved as Markdown tables
    # Clean headers/footers that pollute embeddings
    cleaned_docs = []
    for doc in documents:
        content = doc.page_content
        # Remove common ISO header/footer patterns
        content = _strip_iso_headers_footers(content)
        cleaned_docs.append(Document(
            page_content=content,
            metadata={
                **doc.metadata,
                "source_file": pdf_path,
                "page_number": doc.metadata.get("page", None),
            }
        ))

    return cleaned_docs


def _strip_iso_headers_footers(text: str) -> str:
    """Remove ISO page headers, footers, and page numbers."""
    import re
    # Remove "Page X of Y" patterns
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", "", text)
    # Remove standalone page numbers on their own line
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Remove ISO copyright headers
    text = re.sub(r"©\s*ISO\s*\d{4}.*?All rights reserved", "", text)
    return text.strip()
```

**Why Docling?** It runs locally, handles OCR for scanned ISO PDFs, and recovers table structures as Markdown—critical for ISO 15118's numerous parameter tables.

---

## 🧩 Pairing 1: Recursive Character Splitting + Parent-Child Retrieval

### Chunking Implementation

**ingestion/chunking_pair1.py:**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Tuple

def create_parent_child_chunks(
    documents: List[Document],
    child_chunk_size: int = 256,
    child_overlap: int = 25,
    parent_chunk_size: int = 1024,
    parent_overlap: int = 100,
) -> Tuple[List[Document], List[Document]]:
    """
    Create two levels of chunks:
    - Parent chunks: larger, context-rich
    - Child chunks: smaller, precise for embedding

    Returns (parent_chunks, child_chunks) with parent-child links in metadata.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_chunks = []
    child_chunks = []

    for doc in documents:
        # First split into parent chunks
        parents = parent_splitter.split_documents([doc])
        for i, parent in enumerate(parents):
            parent_id = f"{doc.metadata.get('source_file', 'doc')}_parent_{i}"
            parent.metadata["parent_id"] = parent_id
            parent.metadata["chunk_level"] = "parent"
            parent_chunks.append(parent)

            # Split each parent into child chunks
            children = child_splitter.split_documents([parent])
            for j, child in enumerate(children):
                child.metadata["parent_id"] = parent_id
                child.metadata["chunk_level"] = "child"
                child.metadata["child_index"] = j
                child_chunks.append(child)

    return parent_chunks, child_chunks
```

### Retriever Implementation

**retrieval/pair1_retriever.py:**

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List

def build_parent_child_retriever(
    child_chunks: List[Document],
    parent_chunks: List[Document],
    persist_directory: str = "./chroma_db",
    collection_name: str = "pair1_parent_child",
):
    """
    Build a ParentDocumentRetriever.

    At query time:
    1. Query matches against child chunks (small, precise embeddings)
    2. System looks up parent_id and returns parent chunks (large, contextual)

    LangChain's ParentDocumentRetriever handles the parent-child mapping
    automatically.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Docstore holds parent documents in memory
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=None,  # We already split
        parent_splitter=None, # We already split
    )

    # Add documents with explicit parent-child mapping
    retriever.add_documents(
        documents=parent_chunks,
        ids=[p.metadata["parent_id"] for p in parent_chunks],
    )

    # Index child chunks in vectorstore
    vectorstore.add_documents(child_chunks)

    return retriever
```

**Usage:**

```python
# In main.py or chain setup
retriever = build_parent_child_retriever(child_chunks, parent_chunks)

# Query — returns parent chunks, not child chunks
results = retriever.invoke("What are the AC charging voltage requirements?")
for doc in results:
    print(f"Parent chunk (length: {len(doc.page_content)}):")
    print(doc.page_content[:500])
```

**Why this works:** Child chunks (256 tokens) produce precise embeddings. When a child matches, the retriever returns its parent (1024 tokens), giving the LLM the full clause context.

---

## 🧩 Pairing 2: Section-Based Chunking + Metadata-Filtered Hybrid Search

### Chunking Implementation

**ingestion/chunking_pair2.py:**

```python
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def create_section_based_chunks(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Split at clause boundaries (e.g., 4.1, 4.1.2, Annex A).
    Prepend heading path to each chunk for context.
    Attach ISO-specific metadata.
    """
    section_pattern = re.compile(
        r'^(#{1,4}\s*)?(\d+(?:\.\d+)*)\s+(.+)$',  # Matches "4.1 Title" or "## 4.1 Title"
        re.MULTILINE
    )

    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    for doc in documents:
        text = doc.page_content
        # Find all section boundaries
        matches = list(section_pattern.finditer(text))

        if not matches:
            # No section structure detected — fall back to recursive
            chunks = text_splitter.split_documents([doc])
            for chunk in chunks:
                chunk.metadata.update({
                    "section_number": "unknown",
                    "section_title": "unknown",
                    "heading_path": "",
                    "source_file": doc.metadata.get("source_file"),
                    "page_number": doc.metadata.get("page_number"),
                })
            all_chunks.extend(chunks)
            continue

        # Extract sections between boundaries
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            section_number = match.group(2)  # e.g., "4.1.2"
            section_title = match.group(3).strip()

            # Build heading path from parent sections
            heading_path = _build_heading_path(section_number, section_title, matches)

            # Split long sections further
            if len(section_text) > chunk_size * 1.5:
                sub_chunks = text_splitter.create_documents(
                    texts=[section_text],
                    metadatas=[{
                        "section_number": section_number,
                        "section_title": section_title,
                        "heading_path": heading_path,
                        "source_file": doc.metadata.get("source_file"),
                        "page_number": doc.metadata.get("page_number"),
                        "content_type": "text",
                    }]
                )
                all_chunks.extend(sub_chunks)
            else:
                all_chunks.append(Document(
                    page_content=section_text,
                    metadata={
                        "section_number": section_number,
                        "section_title": section_title,
                        "heading_path": heading_path,
                        "source_file": doc.metadata.get("source_file"),
                        "page_number": doc.metadata.get("page_number"),
                        "content_type": "text",
                    }
                ))

    return all_chunks


def _build_heading_path(section_number: str, title: str, all_matches) -> str:
    """Build hierarchical path like '4.2 Message Flow -> 4.2.1 Supported Messages'."""
    parts = section_number.split(".")
    path_parts = []
    for m in all_matches:
        m_num = m.group(2)
        m_title = m.group(3).strip()
        m_parts = m_num.split(".")
        # Check if this is a parent section
        if len(m_parts) < len(parts) and section_number.startswith(m_num + "."):
            path_parts.append(f"{m_num} {m_title}")
    path_parts.append(f"{section_number} {title}")
    return " -> ".join(path_parts)
```

### Retriever Implementation

**retrieval/pair2_retriever.py:**

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional

def build_metadata_filtered_hybrid_retriever(
    chunks: List[Document],
    persist_directory: str = "./chroma_db",
    collection_name: str = "pair2_section_hybrid",
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
):
    """
    Build a hybrid retriever (vector + BM25) with metadata filtering.

    Chroma supports metadata filtering via the `filter` parameter.
    EnsembleRetriever fuses results using Reciprocal Rank Fusion.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Vector store with metadata
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.add_documents(chunks)

    # Vector retriever (with optional metadata filter at query time)
    vector_retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5}
    )

    # BM25 retriever for exact keyword matching
    bm25_retriever = BM25Retriever.from_documents(
        chunks,
        metadatas=[doc.metadata for doc in chunks],
    )
    bm25_retriever.k = 5

    # Ensemble (hybrid search)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )

    return ensemble_retriever, vectorstore


def filtered_retrieve(
    vectorstore: Chroma,
    query: str,
    section_filter: Optional[str] = None,
    k: int = 5,
):
    """
    Perform metadata-filtered retrieval.

    Example: section_filter="4.1" returns only chunks from Section 4.1.
    """
    search_kwargs = {"k": k}
    if section_filter:
        search_kwargs["filter"] = {"section_number": {"$eq": section_filter}}

    return vectorstore.similarity_search(query, **search_kwargs)
```

**Usage:**

```python
retriever, vectorstore = build_metadata_filtered_hybrid_retriever(chunks)

# Hybrid search (all sections)
results = retriever.invoke("SECC communication setup")

# Metadata-filtered search (only Section 4.1)
filtered = filtered_retrieve(
    vectorstore,
    query="communication setup",
    section_filter="4.1",
)
```

**Why this works:** Section-based chunks already carry rich metadata (`section_number`, `title`, `heading_path`). Metadata filtering provides deterministic precision, while hybrid search handles both semantic queries and exact ISO acronyms like "SECC" and "SLAC".

---

## 🧩 Pairing 3: Advanced ISO Chunking + Graph-Enhanced Hybrid Retrieval

This is the most sophisticated pipeline. It combines table-first chunking, semantic clause chunking, rich metadata, and cross-reference graphs.

### Chunking Implementation

**ingestion/chunking_pair3.py:**

```python
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Set

def create_advanced_iso_chunks(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Document]:
    """
    Advanced ISO chunking that:
    (a) Extracts tables as standalone Markdown chunks
    (b) Splits text at clause boundaries
    (c) Attaches rich metadata (section, title, source, page)
    (d) Extracts cross-references into metadata
    """
    all_chunks = []

    for doc in documents:
        text = doc.page_content
        source_file = doc.metadata.get("source_file", "unknown")
        page_num = doc.metadata.get("page_number", None)

        # --- (a) Extract tables as standalone chunks ---
        table_chunks, text_without_tables = _extract_tables_as_chunks(
            text, source_file, page_num
        )
        all_chunks.extend(table_chunks)

        # --- (b) Split remaining text at clause boundaries ---
        text_chunks = _split_at_clauses(
            text_without_tables, source_file, page_num,
            chunk_size, chunk_overlap
        )
        all_chunks.extend(text_chunks)

    # --- (d) Extract cross-references for each chunk ---
    for chunk in all_chunks:
        refs = _extract_cross_references(chunk.page_content)
        chunk.metadata["cross_references"] = refs  # List of clause numbers

    return all_chunks


def _extract_tables_as_chunks(
    text: str, source_file: str, page_num: int
) -> tuple:
    """
    Extract Markdown tables as standalone chunks.
    Docling outputs tables as Markdown tables, so we detect them.
    """
    # Match Markdown tables (lines starting with |)
    table_pattern = re.compile(
        r'((?:^\|.*\|$\n?)+)',  # One or more consecutive table rows
        re.MULTILINE
    )

    table_chunks = []
    text_without_tables = text

    for i, match in enumerate(table_pattern.finditer(text)):
        table_md = match.group(1).strip()
        # Extract caption if present (line before table)
        caption_match = re.search(
            r'(Table\s+\d+[^\n]*)\n' + re.escape(table_md),
            text
        )
        caption = caption_match.group(1) if caption_match else f"Table {i+1}"

        table_chunks.append(Document(
            page_content=f"**{caption}**\n\n{table_md}",
            metadata={
                "section_number": _find_section_for_table(text, match.start()),
                "section_title": "",
                "heading_path": "",
                "source_file": source_file,
                "page_number": page_num,
                "content_type": "table",
                "table_caption": caption,
                "cross_references": [],
            }
        ))
        # Remove table from text
        text_without_tables = text_without_tables.replace(table_md, "")

    return table_chunks, text_without_tables


def _split_at_clauses(
    text: str, source_file: str, page_num: int,
    chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Split text at clause boundaries with heading context."""
    section_pattern = re.compile(
        r'^(#{1,4}\s*)?(\d+(?:\.\d+)*)\s+(.+)$',
        re.MULTILINE
    )
    matches = list(section_pattern.finditer(text))
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    if not matches:
        return text_splitter.create_documents(
            [text],
            metadatas=[{
                "section_number": "unknown",
                "source_file": source_file,
                "page_number": page_num,
                "content_type": "text",
            }]
        )

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        section_number = match.group(2)
        section_title = match.group(3).strip()

        heading_path = f"{section_number} {section_title}"

        if len(section_text) > chunk_size * 1.5:
            sub_chunks = text_splitter.create_documents(
                [section_text],
                metadatas=[{
                    "section_number": section_number,
                    "section_title": section_title,
                    "heading_path": heading_path,
                    "source_file": source_file,
                    "page_number": page_num,
                    "content_type": "text",
                }]
            )
            chunks.extend(sub_chunks)
        else:
            chunks.append(Document(
                page_content=section_text,
                metadata={
                    "section_number": section_number,
                    "section_title": section_title,
                    "heading_path": heading_path,
                    "source_file": source_file,
                    "page_number": page_num,
                    "content_type": "text",
                }
            ))

    return chunks


def _extract_cross_references(text: str) -> List[str]:
    """
    Extract cross-references like 'see 4.2.3', 'refer to Annex A', 'in 3.1'.
    Returns a list of clause numbers/identifiers.
    """
    ref_patterns = [
        r'see\s+(\d+(?:\.\d+)*)',
        r'refer\s+to\s+(\d+(?:\.\d+)*)',
        r'in\s+(?:clause|section)\s+(\d+(?:\.\d+)*)',
        r'as\s+(?:defined|specified)\s+in\s+(\d+(?:\.\d+)*)',
        r'Annex\s+([A-Z])',
        r'Table\s+(\d+)',
        r'Figure\s+(\d+)',
    ]
    refs = set()
    for pattern in ref_patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            refs.add(match.group(1))
    return sorted(list(refs))


def _find_section_for_table(text: str, table_pos: int) -> str:
    """Find the nearest section number before a table position."""
    section_pattern = re.compile(
        r'^(#{1,4}\s*)?(\d+(?:\.\d+)*)\s+(.+)$',
        re.MULTILINE
    )
    last_section = "unknown"
    for match in section_pattern.finditer(text):
        if match.start() < table_pos:
            last_section = match.group(2)
        else:
            break
    return last_section
```

### Knowledge Graph Construction

**graph/builder.py:**

```python
import networkx as nx
from langchain_core.documents import Document
from typing import List, Dict

def build_cross_reference_graph(chunks: List[Document]) -> nx.DiGraph:
    """
    Build a directed graph where:
    - Nodes = chunks (identified by section_number + content hash)
    - Edges = cross-references between chunks

    This enables graph traversal during retrieval: when a chunk is retrieved,
    we can follow its cross-references to pull in related clauses.
    """
    G = nx.DiGraph()

    # First pass: add all chunks as nodes
    for chunk in chunks:
        node_id = _chunk_id(chunk)
        G.add_node(node_id, chunk=chunk, section=chunk.metadata.get("section_number"))

    # Second pass: add cross-reference edges
    for chunk in chunks:
        source_id = _chunk_id(chunk)
        refs = chunk.metadata.get("cross_references", [])
        for ref in refs:
            # Find the chunk(s) that contain this reference
            for target_chunk in chunks:
                target_section = target_chunk.metadata.get("section_number", "")
                if target_section == ref or target_section.startswith(ref + "."):
                    target_id = _chunk_id(target_chunk)
                    G.add_edge(source_id, target_id, type="cross_reference", ref=ref)

    return G


def _chunk_id(chunk: Document) -> str:
    """Create a unique ID for a chunk."""
    import hashlib
    section = chunk.metadata.get("section_number", "unknown")
    content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
    return f"{section}_{content_hash}"
```

### Graph-Enhanced Retrieval

**retrieval/pair3_retriever.py:**

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_openai import ChatOpenAI
from typing import List, Optional
import networkx as nx


def build_graph_enhanced_retriever(
    chunks: List[Document],
    graph: nx.DiGraph,
    persist_directory: str = "./chroma_db",
    collection_name: str = "pair3_graph_hybrid",
):
    """
    Build a graph-enhanced hybrid retriever.

    Retrieval stages:
    1. Metadata-filtered candidate generation
    2. Hybrid search (vector + BM25) with RRF
    3. Cross-reference graph expansion
    4. (Optional) Cross-encoder reranking
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    vectorstore.add_documents(chunks)

    # Stage 1 & 2: Self-query retriever for automatic metadata filtering
    # SelfQueryRetriever converts natural language to metadata filters
    metadata_field_info = [
        AttributeInfo(
            name="section_number",
            description="The ISO clause number (e.g., '4.1.2')",
            type="string",
        ),
        AttributeInfo(
            name="content_type",
            description="Type of content: 'text' or 'table'",
            type="string",
        ),
        AttributeInfo(
            name="source_file",
            description="The source PDF file name",
            type="string",
        ),
    ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    self_query_retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents="ISO 15118 engineering specification",
        metadata_field_info=metadata_field_info,
        search_kwargs={"k": 5},
    )

    # BM25 retriever for exact term matching
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    # Ensemble: self-query (vector + filter) + BM25
    hybrid_retriever = EnsembleRetriever(
        retrievers=[self_query_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

    return hybrid_retriever, vectorstore, graph


def graph_expanded_retrieve(
    query: str,
    hybrid_retriever,
    graph: nx.DiGraph,
    max_depth: int = 1,
    top_k: int = 5,
) -> List[Document]:
    """
    Stage 3: Cross-reference graph expansion.

    1. Run hybrid search to get initial results
    2. For each result, follow cross-reference edges in the graph
    3. Add linked chunks to the context pool
    4. Return deduplicated, ranked results
    """
    # Stage 1 & 2: Hybrid search
    initial_results = hybrid_retriever.invoke(query)

    # Stage 3: Graph expansion
    expanded_chunks = list(initial_results)
    seen_ids = {_chunk_id(c) for c in initial_results}

    for chunk in initial_results:
        chunk_id = _chunk_id(chunk)
        if chunk_id not in graph:
            continue

        # BFS traversal up to max_depth
        for depth in range(max_depth):
            neighbors = list(graph.successors(chunk_id)) + list(graph.predecessors(chunk_id))
            for neighbor_id in neighbors:
                if neighbor_id not in seen_ids and neighbor_id in graph:
                    neighbor_chunk = graph.nodes[neighbor_id].get("chunk")
                    if neighbor_chunk:
                        expanded_chunks.append(neighbor_chunk)
                        seen_ids.add(neighbor_id)

    # Stage 4: (Optional) Rerank — deduplicate and return top_k
    # For simplicity, we return the expanded set; add a cross-encoder for production
    return expanded_chunks[:top_k * 2]


def _chunk_id(chunk: Document) -> str:
    import hashlib
    section = chunk.metadata.get("section_number", "unknown")
    content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()[:8]
    return f"{section}_{content_hash}"
```

**Usage:**

```python
# Build graph and retriever
graph = build_cross_reference_graph(chunks)
retriever, vectorstore, graph = build_graph_enhanced_retriever(chunks, graph)

# Graph-expanded retrieval
results = graph_expanded_retrieve(
    query="What are the voltage requirements referenced in Section 4.2?",
    hybrid_retriever=retriever,
    graph=graph,
    max_depth=1,
)
```

**Why this works:** This pipeline directly consumes every chunking investment. Markdown tables work for both BM25 (exact header matching) and vector search (semantic content). The cross-reference graph enables multi-hop reasoning—critical for ISO documents where clauses frequently reference each other.

---

## 🔗 Step 2: Assembling the RAG Chain

**chain/rag_chain.py:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_rag_chain(retriever, llm_model: str = "gpt-4o-mini"):
    """
    Build a standard RAG chain using LangChain Expression Language (LCEL).
    """
    llm = ChatOpenAI(model=llm_model, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert assistant for ISO 15118 engineering specifications.
Answer the user's question based ONLY on the provided context.
If the context doesn't contain the answer, say "I don't have enough information."
When referencing information, cite the section number (e.g., "According to Section 4.2...").
If the context includes tables, reproduce the relevant table values accurately."""),
        ("human", """Context:
{context}

Question: {question}"""),
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Section {d.metadata.get('section_number', 'N/A')}] {d.page_content}"
            for d in docs
        )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
```

---

## 🚀 Step 3: Main Entry Point

**main.py:**

```python
from ingestion.loader import load_iso_document
from ingestion.chunking_pair1 import create_parent_child_chunks
from ingestion.chunking_pair2 import create_section_based_chunks
from ingestion.chunking_pair3 import create_advanced_iso_chunks
from retrieval.pair1_retriever import build_parent_child_retriever
from retrieval.pair2_retriever import build_metadata_filtered_hybrid_retriever
from retrieval.pair3_retriever import build_graph_enhanced_retriever, graph_expanded_retrieve
from graph.builder import build_cross_reference_graph
from chain.rag_chain import build_rag_chain

# --- Load documents ---
docs = load_iso_document("iso_15118.pdf", use_ocr=True)

# --- Pairing 1: Recursive + Parent-Child ---
parent_chunks, child_chunks = create_parent_child_chunks(docs)
retriever1 = build_parent_child_retriever(child_chunks, parent_chunks)
chain1 = build_rag_chain(retriever1)
answer1 = chain1.invoke("What are the AC charging voltage requirements?")

# --- Pairing 2: Section-Based + Metadata-Filtered Hybrid ---
chunks2 = create_section_based_chunks(docs)
retriever2, vectorstore2 = build_metadata_filtered_hybrid_retriever(chunks2)
chain2 = build_rag_chain(retriever2)
answer2 = chain2.invoke("What does Section 4.1 specify about communication setup?")

# --- Pairing 3: Advanced ISO + Graph-Enhanced ---
chunks3 = create_advanced_iso_chunks(docs)
graph = build_cross_reference_graph(chunks3)
retriever3, vectorstore3, graph = build_graph_enhanced_retriever(chunks3, graph)
results3 = graph_expanded_retrieve(
    "What voltage requirements are referenced in Section 4.2?",
    retriever3, graph
)
# Build a chain that uses the graph-expanded retriever
chain3 = build_rag_chain(lambda q: graph_expanded_retrieve(q, retriever3, graph))
answer3 = chain3.invoke("What voltage requirements are referenced in Section 4.2?")
```

---

## 📊 Summary Comparison

| Feature | Pairing 1 | Pairing 2 | Pairing 3 |
|---|---|---|---|
| **Chunking** | Recursive (child 256 / parent 1024 tokens) | Section-based (512 tokens, clause boundaries) | Table-first + semantic + metadata + cross-refs |
| **Retrieval** | ParentDocumentRetriever | EnsembleRetriever (vector + BM25) + metadata filter | SelfQuery + Ensemble + Graph expansion |
| **LangChain Components** | `ParentDocumentRetriever`, `InMemoryStore` | `EnsembleRetriever`, `BM25Retriever`, `Chroma` filter | `SelfQueryRetriever`, `EnsembleRetriever`, `NetworkX` |
| **Key Library** | `langchain.retrievers` | `langchain_community.retrievers` | `langchain.retrievers` + `networkx` |
| **Setup Complexity** | Low | Medium | High |
| **Best For** | Quick baseline | Structured ISO docs | Production ISO 15118 |

---

## 💡 Implementation Tips

1. **Start with Pairing 2**: It offers the best balance of simplicity and performance for ISO documents. You get clause-level precision and hybrid search without the complexity of graph construction.

2. **Use Local Embeddings if Needed**: If you're running an embedded LLM, replace `OpenAIEmbeddings` with `HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")` from `langchain_huggingface`.

3. **Persist the Vector Store**: Chroma's `persist_directory` ensures you don't re-embed on every run. Use `vectorstore.persist()` after ingestion.

4. **Batch Ingestion**: For large ISO documents, process chunks in batches to avoid memory issues:

   ```python
   from langchain_core.documents import Document
   batch_size = 100
   for i in range(0, len(chunks), batch_size):
       vectorstore.add_documents(chunks[i:i+batch_size])
   ```

5. **Test Retrieval Independently**: Before building the full chain, verify retrieval quality with a set of known queries and expected sections. This isolates chunking/retrieval issues from LLM generation issues.