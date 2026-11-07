To use open-source models for your RAG pipeline, you'll need to select three components: embedding models, cross-encoder rerankers, and a generator LLM. The models below are all fully open-weight and integrate well with LangChain.

### 🧠 Open-Source Component Options

Here are strong open-source candidates for each part of your pipeline, with a focus on technical document retrieval.

| Component | Recommended Model(s) | Key Strengths & Notes |
| :--- | :--- | :--- |
| **Embedding Models** | **Qwen3-Embedding** (0.6B, 4B, 8B) | Multilingual, strong on technical/code retrieval, supports MRL for flexible dimensions. A top open-source choice. |
| | **IBM Granite Embedding R2** (English/Multilingual) | Apache 2.0 license. Excellent for enterprise use, with strong performance on code and long-document retrieval. |
| | **BGE-Large** (v1.5, ISO-finetuned variants) | High-quality English embeddings. Fine-tuned versions (e.g., for ISO 27001) exist, showing the architecture adapts well to standards. |
| | **EmbeddingGemma** (308M) | Best-in-class under 500M params. Designed for on-device use, multilingual, and integrates with Ollama and LangChain. |
| **Reranker (Cross-Encoder)** | **BGE-Reranker-v2-m3** | Multilingual, strong open-source performance, and widely used for RAG. |
| | **mxbai-rerank-large-v2** | Currently the best open-source accuracy for English-focused RAG pipelines. |
| | **IBM Granite Embedding English Reranker r2** | A 149M param cross-encoder from IBM, Apache 2.0 licensed, and optimized for enterprise retrieval. |
| **Generator LLM** | **Llama 3 / 3.2** (8B, 11B) | A proven, widely used open-weight family that runs locally via Ollama. |
| | **Command A+** | Cohere's open-source model, explicitly optimized for RAG and agentic workflows. |
| | **Gemma 3** (e.g., 27B) | Another strong local option, with smaller variants that can run on consumer hardware. |

### 🔌 LangChain Integration Snippets

Below are code examples for swapping your current OpenAI-dependent components with these open-source models.

#### 1. Embedding Models

Use `HuggingFaceEmbeddings` from `langchain-huggingface` for local inference. This replaces your current `OpenAIEmbeddings`.

```python
from langchain_huggingface import HuggingFaceEmbeddings

# Example with Qwen3-Embedding-0.6B
embeddings = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_kwargs={"device": "cuda"}, # or "cpu"
    encode_kwargs={"normalize_embeddings": True}
)

# Example with IBM Granite Embedding
# embeddings = HuggingFaceEmbeddings(
#     model_name="ibm-granite/granite-embedding-english-r2",
#     model_kwargs={"device": "cpu"}
# )
```

In your code, simply replace:
```python
# from langchain_openai import OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```
with the snippet above and pass the `embeddings` object to your `Chroma` vector store.

#### 2. Cross-Encoder Reranker

Use `sentence-transformers`'s `CrossEncoder` for the optional reranking stage in Pairing 3.

```python
from sentence_transformers import CrossEncoder

# Example with BGE-Reranker-v2-m3
reranker = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    max_length=512,
    device="cuda" # or "cpu"
)

# Score query-chunk pairs
pairs = [[query, doc.page_content] for doc in retrieved_docs]
scores = reranker.predict(pairs)

# Sort docs by score and take the top-k
ranked_results = [doc for _, doc in sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)]
final_context = ranked_results[:TOP_K]
```

#### 3. Generator LLM (Local via Ollama)

Instead of `ChatOpenAI`, use `ChatOllama` from `langchain-ollama` to run models like Llama 3 locally.

```python
# First, ensure Ollama is running and the model is pulled:
# ollama pull llama3

from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3",
    temperature=0,
    base_url="http://localhost:11434" # Default Ollama URL
)

# The `build_rag_chain` function can now use this `llm` instance
# instead of the one from ChatOpenAI.
```

### 💎 Recommendation for Your ISO Project

Based on the search results and your specific use case of ISO specification documents:

1.  **Embedding Model**: Start with **Qwen3-Embedding-0.6B** or **IBM Granite Embedding R2**. Both are modern, open-source, and show strong performance on technical and enterprise retrieval tasks. The Qwen3 family also has a matching reranker for a consistent stack.
2.  **Reranker**: Pair it with **BGE-Reranker-v2-m3**. It's multilingual and highly regarded in the open-source community for RAG applications. For an English-only ISO document, **mxbai-rerank-large-v2** is a strong alternative.
3.  **LLM**: Use **Llama 3 (8B)** via Ollama. It is a well-established choice for local RAG, balancing performance and hardware requirements.

This combination gives you a fully open-source, locally runnable stack that should perform well on the structured, technical content of ISO 15118.