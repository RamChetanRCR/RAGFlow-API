# RAGFlow — LangGraph RAG Pipeline

A production-grade RAG (Retrieval-Augmented Generation) pipeline built as a FastAPI microservice. Ingests PDF documents and answers natural-language queries with cited, grounded responses using a LangGraph StateGraph.

## Architecture

```
User Query → Query Rewriter → Retriever (ChromaDB) → Reranker (Cohere) → Answer Generator → Citation Formatter → Response
                                                                                │
                                                                          No Context?
                                                                                │
                                                                         "I don't know"
```

Built with **LangGraph StateGraph** — 5 connected nodes with conditional edges that prevent hallucination.

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.11+ |
| Agent Framework | LangGraph (StateGraph) |
| LLM | Ollama (local) / Gemini / OpenAI |
| Embeddings | Ollama nomic-embed-text / Gemini |
| Vector DB | ChromaDB (in-memory or persistent) |
| Reranking | Cohere Rerank (optional fallback) |
| API | FastAPI + Uvicorn |
| PDF Parsing | PyMuPDF (fitz) |
| Auth | API Key middleware |
| Streaming | Server-Sent Events |
| Frontend | Next.js + TypeScript |
| Container | Docker + docker-compose |

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) with `llama3.2:3b` and `nomic-embed-text` models
- Node.js 18+ (for frontend)

### Setup

```bash
# Clone and enter
cd ragflow-api

# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Environment
cp .env.example .env
# Edit .env as needed (defaults work with Ollama + in-memory ChromaDB)
```

### Run Backend

```bash
# Make sure Ollama is running
ollama serve

# Start API
uvicorn app.main:app --reload --port 8000
```

### Ingest a PDF and Query

```bash
# Ingest
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: your-api-key" \
  -F "file=@document.pdf"

# Query
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'

# Streaming
curl -N "http://localhost:8000/query/stream?query=What+is+this" \
  -H "X-API-Key: your-api-key"
```

### Run Frontend

```bash
cd frontend
npm install
npm run dev
# Opens at http://localhost:3000
```

### Docker (with persistent storage)

```bash
docker-compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | API version, ChromaDB status, LLM check |
| POST | `/ingest` | Upload PDF → parse → chunk → embed → store |
| POST | `/query` | Full pipeline: rewrite → retrieve → rerank → generate → cite |
| GET | `/query/stream` | SSE streaming version |
| GET | `/docs` | List ingested documents |
| GET | `/docs/{doc_id}` | Get document metadata |
| DELETE | `/docs/{doc_id}` | Remove document + vectors |

## Project Structure

```
ragflow-api/
├── app/
│   ├── main.py              # FastAPI app, lifespan, routers
│   ├── config.py            # Pydantic Settings (env-based)
│   ├── models.py            # Pydantic request/response schemas
│   ├── pipeline/
│   │   ├── graph.py         # LangGraph StateGraph definition
│   │   ├── nodes.py         # 5 node functions + answer generation
│   │   └── state.py         # RAGState TypedDict
│   ├── services/
│   │   ├── ingestor.py      # PDF parsing → chunking → embed → upsert
│   │   ├── retriever.py     # ChromaDB search wrapper
│   │   ├── reranker.py      # Cohere rerank wrapper
│   │   └── chromadb_service.py  # Shared ChromaDB singleton
│   └── middleware/
│       └── auth.py          # API key validation
├── frontend/                # Next.js chat UI
├── tests/                   # pytest suite (7 tests)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── AGENTS.md                # Build notes & decisions
└── README.md
```

## LangGraph Pipeline (Multi-Node, Not Multi-Agent)

The core pipeline uses `langgraph.graph.StateGraph` with 5 deterministic nodes. Each node is a fixed function — **not** an autonomous agent with tools/reasoning loops. True multi-agent systems (like chat-langchain's `create_agent`) give each agent its own tools, middleware, and decision-making capability.

1. **query_rewriter** — LLM rewrites ambiguous queries for better retrieval
2. **retriever** — ChromaDB semantic search (configurable top-k)
3. **reranker** — Cohere rerank (or falls back to raw scores)
4. **answer_generation** — LLM generates answer from context with citations
5. **citation_formatter** — Extracts `[doc_name, p.N]` citations from output

**Conditional edge**: If reranker returns relevance < 0.3, routes to `no_context_fallback` instead of generating — prevents hallucination.

## Configuration

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama endpoint |
| `LLM_MODEL` | `llama3.2:3b` | LLM model |
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | Embedding endpoint |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model |
| `CHROMA_PERSIST_DIRECTORY` | `:memory:` | ChromaDB (path for persistent) |
| `API_KEY` | `changeme` | Auth header key |
| `COHERE_API_KEY` | — | Optional reranker |
| `TOP_K_RETRIEVE` | 20 | Retrieved chunks |
| `TOP_K_RERANK` | 5 | Reranked chunks |
