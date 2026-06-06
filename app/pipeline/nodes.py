import json
import re
from typing import AsyncGenerator, Optional

from openai import OpenAI

from app.config import get_settings
from app.models import Chunk, Citation
from app.pipeline.state import RAGState
from app.services.retriever import Retriever
from app.services.reranker import Reranker

settings = get_settings()
_llm_client: Optional[OpenAI] = None
_retriever: Optional[Retriever] = None
_reranker: Optional[Reranker] = None


def get_llm():
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            base_url=settings.llm_base_url,
            api_key="ollama",
        )
    return _llm_client


def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker


def query_rewriter(state: RAGState) -> dict:
    query = state["query"]
    prompt = (
        "You are a query rewriter for a document Q&A system. "
        "Rewrite the following query to be more precise and optimized for retrieval. "
        "Resolve pronouns, abbreviations, and ambiguous terms. "
        "Output ONLY the rewritten query with no explanation.\n\n"
        f"Query: {query}\nRewritten query:"
    )
    client = get_llm()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    rewritten = response.choices[0].message.content.strip().strip('"')
    return {"rewritten_query": rewritten or query}


def retriever_node(state: RAGState) -> dict:
    rewritten_query = state.get("rewritten_query") or state["query"]
    doc_id = state.get("doc_id", "")
    retriever = get_retriever()
    chunks = retriever.search(rewritten_query, doc_id=doc_id)
    return {"retrieved_chunks": chunks}


def reranker_node(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["query"]
    chunks = state.get("retrieved_chunks", [])
    reranker = get_reranker()
    reranked = reranker.rerank(query, chunks)
    return {"reranked_chunks": reranked}


def should_generate(state: RAGState) -> str:
    reranked = state.get("reranked_chunks", [])
    if not reranked or max((c.relevance_score for c in reranked), default=0) < 0.3:
        return "no_context_fallback"
    return "generate"


def no_context_fallback(state: RAGState) -> dict:
    return {
        "answer": "I don't have information on this.",
        "citations": [],
    }


async def answer_generator(state: RAGState) -> AsyncGenerator[str, None]:
    chunks = state.get("reranked_chunks", [])
    query = state.get("rewritten_query") or state["query"]

    context_parts = []
    for c in chunks:
        header = f"[{c.section_header}] " if c.section_header else ""
        context_parts.append(
            f"Source: doc_{c.doc_id[:8]}, page {c.page_number}\n"
            f"{header}{c.text}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a precise document analyst. Answer ONLY using the provided context.\n"
        "For each claim, add an inline citation like [doc_name, p.N].\n"
        "If the answer is not in context, say: 'I don't have information on this.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    client = get_llm()
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield f"data: {json.dumps({'token': token})}\n\n"
    yield "data: [DONE]\n\n"


def answer_generation(state: RAGState) -> dict:
    chunks = state.get("reranked_chunks", [])
    query = state.get("rewritten_query") or state["query"]

    context_parts = []
    for c in chunks:
        header = f"[{c.section_header}] " if c.section_header else ""
        context_parts.append(
            f"Source: doc_{c.doc_id[:8]}, page {c.page_number}\n"
            f"{header}{c.text}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a precise document analyst. Answer ONLY using the provided context.\n"
        "For each claim, add an inline citation like [doc_name, p.N].\n"
        "If the answer is not in context, say: 'I don't have information on this.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    client = get_llm()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return {"answer": response.choices[0].message.content}


async def answer_generator_blocking(state: RAGState) -> dict:
    chunks = state.get("reranked_chunks", [])
    query = state.get("rewritten_query") or state["query"]

    context_parts = []
    for c in chunks:
        header = f"[{c.section_header}] " if c.section_header else ""
        context_parts.append(
            f"Source: doc_{c.doc_id[:8]}, page {c.page_number}\n"
            f"{header}{c.text}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a precise document analyst. Answer ONLY using the provided context.\n"
        "For each claim, add an inline citation like [doc_name, p.N].\n"
        "If the answer is not in context, say: 'I don't have information on this.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\nAnswer:"
    )

    client = get_llm()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return {"answer": response.choices[0].message.content}


def citation_formatter(state: RAGState) -> dict:
    answer = state.get("answer", "")
    citations = []
    pattern = r'\[(.*?),\s*p\.(\d+)\]'
    matches = re.findall(pattern, answer)
    for doc_name, page in matches:
        citations.append(Citation(
            doc_name=doc_name,
            page=int(page),
            text="",
        ))
    return {"answer": answer, "citations": citations}
