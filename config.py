"""Shared configuration for the ISO 15118 RAG pipeline.

All values are sourced from environment variables (see `.env.example`) with
sensible defaults. This module must not import heavy dependencies (torch,
langchain) so it stays importable everywhere.
"""

import os

# These MUST be set before torch is imported anywhere in the process, because torch
# caches the native-JIT state at import time.
# - TORCH_DISABLE_NATIVE_JIT: avoid Triton JIT needing a C compiler (see docs/MILESTONE_2.md).
# - PYTORCH_ENABLE_MPS_FALLBACK: allow CPU fallback for unsupported Apple MPS ops.
os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from dotenv import load_dotenv

load_dotenv()

# --- LLM (hybrid provider switch) ---
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
# Default aligned to the model verified present in the dev environment (see docs/MILESTONE_2.md).
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:3b-instruct-q4_K_M")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
# OpenAI-compatible endpoint override. Blank = real OpenAI; set to
# "https://openrouter.ai/api/v1" to route the chat LLM through OpenRouter.
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "").strip()
# API key env var name for the OpenAI-compatible provider (resolved lazily so the
# key is never stored in module state or logs).
LLM_API_KEY_ENV: str = os.getenv("LLM_API_KEY_ENV", "OPENAI_API_KEY")

# --- Embeddings / reranker ---
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# Embedding prompts: Qwen3-Embedding is instruction-aware and ships built-in prompts
# (query = retrieval instruction, document = empty). "auto" enables them only for models
# that define them; "1"/"0" force on/off. Set EMBEDDING_USE_PROMPTS=0 for non-Qwen models.
# Embedding provider: "local" (sentence-transformers, default) or "openai"
# (any OpenAI-compatible /embeddings endpoint, e.g. OpenRouter).
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
# Base URL for the OpenAI-compatible embeddings endpoint (OpenRouter by default).
EMBEDDING_BASE_URL: str = os.getenv(
    "EMBEDDING_BASE_URL", "https://openrouter.ai/api/v1"
).strip()
# API key env var name for the embeddings endpoint (resolved lazily).
EMBEDDING_API_KEY_ENV: str = os.getenv("EMBEDDING_API_KEY_ENV", "OPENROUTER_API_KEY")
EMBEDDING_USE_PROMPTS: str = os.getenv("EMBEDDING_USE_PROMPTS", "auto").strip().lower()
EMBEDDING_QUERY_PROMPT_NAME: str = os.getenv("EMBEDDING_QUERY_PROMPT_NAME", "query")
EMBEDDING_DOC_PROMPT_NAME: str = os.getenv("EMBEDDING_DOC_PROMPT_NAME", "document")
# Flatten Markdown tables (drop pipes/separators) before embedding; stored text is unchanged.
EMBEDDING_STRIP_TABLES: bool = os.getenv("EMBEDDING_STRIP_TABLES", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# Cross-encoder max sequence length. bge-reranker-v2-m3 supports up to 8192; 512 truncated
# our (now larger) table chunks, so the default is raised.
RERANKER_MAX_LENGTH: int = int(os.getenv("RERANKER_MAX_LENGTH", 1024))

# Device override: blank means auto-detect (see models/device.py).
DEVICE_OVERRIDE: str | None = os.getenv("DEVICE", "").strip() or None

# --- Chunking: Pairing 1 (recursive parent/child) ---
PAIR1_CHILD_CHUNK_SIZE: int = 256
PAIR1_CHILD_OVERLAP: int = 25
PAIR1_PARENT_CHUNK_SIZE: int = 1024
PAIR1_PARENT_OVERLAP: int = 100

# --- Chunking: Pairings 2 and 3 ---
PAIR2_CHUNK_SIZE: int = 512
PAIR2_CHUNK_OVERLAP: int = 50
PAIR3_CHUNK_SIZE: int = 512
PAIR3_CHUNK_OVERLAP: int = 50
# Oversized tables are split on row boundaries (header repeated) above this size.
TABLE_MAX_CHARS: int = int(os.getenv("TABLE_MAX_CHARS", 4000))

# --- Retrieval ---
TOP_K: int = int(os.getenv("TOP_K", 5))
HYBRID_VECTOR_WEIGHT: float = float(os.getenv("HYBRID_VECTOR_WEIGHT", 0.6))
HYBRID_BM25_WEIGHT: float = float(os.getenv("HYBRID_BM25_WEIGHT", 0.4))
RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", 20))

# --- Vector store ---
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# --- Input corpus ---
ISO_PDF_DIR: str = os.getenv("ISO_PDF_DIR", "./Iso")

# --- Parsed-document cache (Docling JSON) to avoid re-converting PDFs ---
DOCLING_CACHE_DIR: str = os.getenv("DOCLING_CACHE_DIR", "./data/parsed")

# Docling batch size. 1 keeps peak memory low (8 GB machines OOM with defaults on large PDFs).
DOCLING_BATCH_SIZE: int = int(os.getenv("DOCLING_BATCH_SIZE", 1))

# Vector store ingestion batch size (Chroma caps a single upsert at ~5461).
VECTORSTORE_BATCH_SIZE: int = int(os.getenv("VECTORSTORE_BATCH_SIZE", 1000))

# Cloud mode: load the pre-built Chroma index from disk and skip Docling ingestion
# entirely (the hosted container has no torch/docling). Set CLOUD_MODE=1 when deploying.
CLOUD_MODE: bool = os.getenv("CLOUD_MODE", "0").strip().lower() in ("1", "true", "yes", "on")
