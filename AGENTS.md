# RAGFlow — Multi-Agent RAG Pipeline

A production-grade multi-agent RAG pipeline built as a FastAPI microservice. Ingests PDF documents and answers natural-language queries with cited, grounded responses using a LangGraph StateGraph with **4 autonomous agents**.

## Architecture (Multi-Agent)

```
User Query
    │
    ▼
┌─────────────────────────┐
│  Agent 1: Guardrails    │  LLM decides: document-related? → pass/reject
└────────┬────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Rewrite    Reject Query
    │      (off-topic → END)
    ▼
┌─────────────────────────┐
│  Agent 2: Query Rewriter│  LLM rewrites for precision
└────────┬────────────────┘
         ▼
┌──────────────────────────────┐
│  Agent 3: Retrieval Strategist│  LLM picks strategy: semantic | doc_specific
└────────┬─────────────────────┘
         ▼
┌─────────────────┐
│    Retriever     │  ChromaDB semantic search (top-20)
└────────┬────────┘
         ▼
┌─────────────────┐
│    Reranker      │  Cohere Rerank (top-20 → top-5)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Generate  No Context
    │      Fallback
    ▼         │
┌─────────────────────────┐
│  Agent 4: Quality Review│  LLM scores 1-10, checks citations, hallucination
└────────┬────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Format    Regenerate
 (cite)   (loop back to Generate, max 3)
    │
    ▼
 Answer + Citations
```

## Agents

| Agent | Decision | Tools |
|-------|----------|-------|
| **Guardrails Agent** | Is query document-related? | LLM classification |
| **Query Rewriter** | Resolve pronouns/ambiguity | LLM rewrite |
| **Retrieval Strategist** | Which strategy/doc filter? | LLM strategy pick |
| **Quality Reviewer Agent** | Pass quality gate? Regenerate? | LLM scoring (1-10) |

## Conditional Loops
- **Guardrails**: Off-topic → early reject ("I can only answer...")
- **Reranker → Generate/Fallback**: relevance < 0.3 → no_context_fallback
- **Quality Review → Regenerate**: score < 6 & retries < 3 → loop back to answer_generation

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.11+ |
| Agent Framework | **LangGraph** (StateGraph with conditional edges & loops) |
| LLM | Ollama (`llama3.2:3b` via OpenAI-compatible API) |
| Embeddings | Ollama (`nomic-embed-text`, 768d) |
| Vector DB | ChromaDB (`:memory:` or persistent, shared singleton) |
| Reranking | Cohere Rerank (free tier) |
| API | FastAPI + Uvicorn |
| PDF Parsing | PyMuPDF (fitz) |
| Streaming | SSE (Server-Sent Events) |
| Testing | pytest + httpx async client |
| Config | Pydantic Settings v2 |

## Key Design Decisions

1. **Multi-Agent (not multi-node)**: Each agent makes LLM-based decisions — Guardrails classifies, Strategist picks strategy, Quality Reviewer scores and gates. This is genuinely multi-agent: agents have reasoning + decision-making loops.

2. **Quality Review Loop**: The Quality Reviewer scores answers 1-10. Below 6 triggers regeneration (up to 3 retries). This prevents hallucination and low-quality output.

3. **Guardrails Gate**: Off-topic queries (greetings, unrelated) are rejected before they reach the pipeline, saving LLM calls and preventing irrelevant responses.

4. **Ollama local**: `llama3.2:3b` for LLM, `nomic-embed-text` for embeddings. Swapped from Gemini due to free-tier rate limits.

5. **ChromaDB**: Runs in-process (no Docker needed). Shared singleton means Ingestor and Retriever use same instance. Use `:memory:` for ephemeral or a directory path for persistence.

6. **Conditional edges**: Relevance < 0.3 → no_context_fallback instead of hallucinating. Quality < 6 → regenerate loop.

## Multi-Agent vs Multi-Node

Each agent above is a genuine agent because:
- It makes an **LLM-based decision** (not a deterministic computation)
- It has **autonomy** (e.g., Quality Reviewer can trigger regeneration)
- The graph has **conditional loops** based on agent decisions
- Each agent has its own **prompt + reasoning** tailored to its role

## Quick Start

```bash
cd ragflow-api
uvicorn app.main:app --port 8000 --reload
```

## Project Structure

```
ragflow-api/
├── app/
│   ├── main.py              # FastAPI app, endpoints
│   ├── config.py            # Pydantic Settings (env-based)
│   ├── models.py            # Pydantic schemas
│   ├── pipeline/
│   │   ├── graph.py         # LangGraph StateGraph (4 agents, conditional edges)
│   │   ├── nodes.py         # Agent functions + pipeline nodes
│   │   └── state.py         # RAGState TypedDict
│   ├── services/
│   │   ├── ingestor.py      # PDF → chunk → embed → ChromaDB
│   │   ├── retriever.py     # ChromaDB query wrapper
│   │   ├── reranker.py      # Cohere rerank wrapper
│   │   └── chromadb_service.py  # Shared ChromaDB singleton
│   └── middleware/
│       └── auth.py          # (removed for local dev)
├── tests/
│   ├── test_ingest.py
│   └── test_query.py
├── frontend/                # Next.js chat UI
└── requirements.txt
```
