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


def _llm_call(system: str, prompt: str, temperature: float = 0.1) -> str:
    client = get_llm()
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


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


# ──────── Agent 1: Guardrails Agent ────────

def guardrails_agent(state: RAGState) -> dict:
    query = state["query"]
    system = (
        "You are a guardrails agent for a document Q&A system. "
        "Your job is to decide if a user query is related to document content "
        "(e.g., questions about a PDF, its topics, summaries, specific sections) "
        "or if it is off-topic (greetings, unrelated questions, code generation, etc.)."
    )
    prompt = (
        f"Query: {query}\n\n"
        "Respond with a JSON object with two keys:\n"
        '  "passed": true/false (whether the query is document-related)\n'
        '  "reason": "short explanation of your decision"\n'
        "JSON:"
    )
    raw = _llm_call(system, prompt, temperature=0.0)
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"passed": True, "reason": "guardrails parse failed, allowing"}
    return {
        "guardrails_passed": result.get("passed", True),
        "guardrails_reason": result.get("reason", ""),
    }


def guardrails_router(state: RAGState) -> str:
    if state.get("guardrails_passed", True):
        return "rewrite"
    return "reject_query"


def reject_query(state: RAGState) -> dict:
    reason = state.get("guardrails_reason", "off-topic query")
    return {
        "answer": f"I can only answer questions based on the uploaded documents. {reason}",
        "citations": [],
        "quality_passed": True,
    }


# ──────── Agent 2: Query Rewriter ────────

def query_rewriter(state: RAGState) -> dict:
    query = state["query"]
    system = (
        "You are a query rewriter for a document Q&A system. "
        "Rewrite queries to be precise and optimized for retrieval. "
        "Resolve pronouns, abbreviations, and ambiguous terms."
    )
    prompt = (
        f"Query: {query}\n\nRewritten query (output ONLY the rewritten text):"
    )
    rewritten = _llm_call(system, prompt, temperature=0.1)
    rewritten = rewritten.strip('"').strip("'")
    return {"rewritten_query": rewritten or query}


# ──────── Agent 3: Retrieval Strategist ────────

def retrieval_strategist(state: RAGState) -> dict:
    rewritten = state.get("rewritten_query") or state["query"]
    system = (
        "You are a retrieval strategist agent. Given a query, pick the best "
        "retrieval strategy and any document ID filter if a specific doc is referenced. "
        "Available strategies:\n"
        '- "semantic": general semantic search (default)\n'
        '- "document_specific": restrict to one document\n'
    )
    prompt = (
        f"Query: {rewritten}\n\n"
        "Respond ONLY with a JSON object:\n"
        '  {"strategy": "semantic" | "document_specific", "doc_id": "" | "doc_<id>"}\n'
        "JSON:"
    )
    raw = _llm_call(system, prompt, temperature=0.0)
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"strategy": "semantic", "doc_id": ""}
    return {
        "retrieval_strategy": result.get("strategy", "semantic"),
        "doc_id": result.get("doc_id", state.get("doc_id", "")),
    }


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

    system = (
        "You are a precise document analyst. Answer ONLY using the provided context. "
        "For each claim, add an inline citation like [doc_name, p.N]. "
        "If the answer is not in context, say: 'I don't have information on this.'"
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    return {"answer": _llm_call(system, prompt, temperature=0.1)}


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


# ──────── Agent 4: Quality Reviewer Agent ────────

def quality_reviewer_agent(state: RAGState) -> dict:
    answer = state.get("answer", "")
    query = state.get("query", "")
    retries = state.get("max_retries", 0)

    system = (
        "You are a quality review agent for a document Q&A system. "
        "Evaluate the answer on: relevance to the query, citation usage, "
        "and whether it hallucinates (makes claims not supported by citations)."
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Answer: {answer}\n\n"
        "Respond ONLY with a JSON object:\n"
        '  {"score": 1-10, "passed": true/false, "reason": "short feedback"}\n'
        'Pass if score >= 6 or if we have already attempted 2 regenerations.\n'
        f"Current regeneration count: {retries}\n"
        "JSON:"
    )
    raw = _llm_call(system, prompt, temperature=0.0)
    raw = raw.removeprefix("```json").removesuffix("```").strip()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"score": 6, "passed": True, "reason": "parse failed, allowing"}

    return {
        "quality_score": result.get("score", 5),
        "quality_passed": result.get("passed", True),
        "max_retries": retries + 1,
    }


def quality_router(state: RAGState) -> str:
    if state.get("quality_passed", True):
        return "format"
    retries = state.get("max_retries", 0)
    if retries >= 3:
        return "format"
    return "regenerate"


# ──────── Streaming ────────

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

    system = (
        "You are a precise document analyst. Answer ONLY using the provided context. "
        "For each claim, add an inline citation like [doc_name, p.N]. "
        "If the answer is not in context, say: 'I don't have information on this.'"
    )
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    client = get_llm()
    stream = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        stream=True,
    )
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            yield f"data: {json.dumps({'token': token})}\n\n"
    yield "data: [DONE]\n\n"
