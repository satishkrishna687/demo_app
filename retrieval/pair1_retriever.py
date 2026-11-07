"""Pairing 1 retrieval: Parent-Child retriever.

At query time the retriever matches against **child** chunks (small, precise embeddings)
and returns the corresponding **parent** chunks (large, contextual) from the docstore.

Implementation note: we pre-split chunks in :mod:`ingestion.chunking_pair1`, so the
``ParentDocumentRetriever`` is configured with ``id_key="parent_id"`` and no splitters.
Parents are placed in the docstore keyed by ``parent_id``; children (which carry
``parent_id`` metadata) are indexed in Chroma. The retriever's built-in logic then does
the child-search → parent-fetch mapping.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_core.documents import Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_PERSIST_DIR,
    PAIR1_CHILD_CHUNK_SIZE,
    PAIR1_CHILD_OVERLAP,
    PAIR1_PARENT_CHUNK_SIZE,
    PAIR1_PARENT_OVERLAP,
    TOP_K,
)
from models.embeddings import get_embeddings
from vectorstore.store import ensure_indexed


def build_parent_child_retriever(
    child_chunks: list[Document],
    parent_chunks: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = "pair1_parent_child",
    k: int = TOP_K,
) -> ParentDocumentRetriever:
    """Build a ``ParentDocumentRetriever`` over pre-split parent/child chunks."""
    embeddings = get_embeddings()

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    # Parents live in the docstore, keyed by parent_id.
    store = InMemoryStore()
    store.mset([(p.metadata["parent_id"], p) for p in parent_chunks])

    # Children are embedded; deterministic ids keep re-runs idempotent.
    child_ids = [
        f"{c.metadata['parent_id']}_child_{c.metadata.get('child_index', i)}"
        for i, c in enumerate(child_chunks)
    ]
    ensure_indexed(vectorstore, child_chunks, ids=child_ids)

    # The model requires TextSplitter instances, but they are never used: we pre-split
    # in ingestion.chunking_pair1 and populate the stores manually, so add_documents()
    # is not called.
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PAIR1_CHILD_CHUNK_SIZE, chunk_overlap=PAIR1_CHILD_OVERLAP
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PAIR1_PARENT_CHUNK_SIZE, chunk_overlap=PAIR1_PARENT_OVERLAP
    )

    return ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        id_key="parent_id",
        search_kwargs={"k": k},
    )
