"""LCEL RAG chain assembly.

Builds a retrieval-augmented generation chain from any retriever (a LangChain
``BaseRetriever`` or a plain callable) and the configured chat LLM. The retriever is
provider-agnostic, so the same chain works for all three pairings.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from models.llm import get_llm

_SYSTEM_PROMPT = """You are an expert assistant for the ISO 15118 engineering specifications \
(Parts 1, 2, and 3).
Answer the user's question based ONLY on the provided context.
If the context does not contain the answer, say "I don't have enough information."
When referencing information, cite the document and section (e.g., "According to ISO 15118-2, \
Section 4.2...").
If the context includes tables, reproduce the relevant table values accurately."""

_HUMAN_PROMPT = """Context:
{context}

Question: {question}"""


def format_docs(docs: list[Document]) -> str:
    """Render retrieved documents with document/section/page headers."""
    parts = []
    for d in docs:
        meta = d.metadata
        header = (
            f"[{meta.get('document_id', 'N/A')} | "
            f"Section {meta.get('section_number', 'N/A')} | "
            f"page {meta.get('page_number', '?')}]"
        )
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_rag_chain(
    retriever: Runnable | Callable[[str], list[Document]],
    llm_model: str | None = None,
    temperature: float = 0.0,
) -> Any:
    """Assemble the LCEL RAG chain.

    Args:
        retriever: A LangChain retriever or a callable ``query -> list[Document]``.
        llm_model: Optional model override; defaults to the configured ``LLM_MODEL``.
        temperature: Sampling temperature (0 for deterministic answers).

    Returns:
        A runnable chain: ``invoke(question) -> str``.
    """
    llm = get_llm(model=llm_model, temperature=temperature) if llm_model else get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("human", _HUMAN_PROMPT)]
    )

    retriever_runnable = (
        retriever if isinstance(retriever, Runnable) else RunnableLambda(retriever)
    )

    return (
        {"context": retriever_runnable | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


_CONDENSE_SYSTEM = """Given the conversation so far and a follow-up question, rewrite the \
follow-up into a standalone question that can be understood on its own. Return ONLY the \
rewritten question, with no preamble or explanation. If the follow-up is already \
standalone, return it unchanged."""


def build_conversational_chain(
    llm_model: str | None = None,
    temperature: float = 0.0,
    with_history: bool = True,
) -> Any:
    """Build an answer chain that accepts ``{context, question, history}``.

    Unlike :func:`build_rag_chain`, retrieval is done by the caller, so the same retrieved
    context can be reused for the source panel. When ``with_history`` is true the prompt
    includes a ``history`` placeholder (a list of chat messages), giving the LLM memory of
    previous turns.

    Returns:
        A runnable: ``invoke({"context": str, "question": str, "history": list}) -> str``.
    """
    llm = get_llm(model=llm_model, temperature=temperature) if llm_model else get_llm()

    messages: list[Any] = [("system", _SYSTEM_PROMPT)]
    if with_history:
        messages.append(MessagesPlaceholder("history"))
    messages.append(("human", _HUMAN_PROMPT))

    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt | llm | StrOutputParser()


def build_condense_chain(llm_model: str | None = None, temperature: float = 0.0) -> Any:
    """Build a callable that rewrites a follow-up into a standalone question.

    Returns:
        ``condense(history, question) -> str``. Falls back to the raw question on any error
        or when there is no history.
    """
    llm = get_llm(model=llm_model, temperature=temperature) if llm_model else get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _CONDENSE_SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()

    def condense(history: list, question: str) -> str:
        if not history:
            return question
        try:
            rewritten = chain.invoke({"history": history, "question": question}).strip()
            return rewritten or question
        except Exception:
            return question

    return condense
