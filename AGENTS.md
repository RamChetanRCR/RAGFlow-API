# RAGFlow — Multi-Agent RAG Pipeline

A production-grade, multi-agent RAG (Retrieval-Augmented Generation) pipeline built as a FastAPI microservice. Ingests PDF documents and answers natural-language queries with cited, grounded responses using a LangGraph-based multi-agent pipeline — all under 2 seconds.

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Rewriter  │  Resolves pronouns/ambiguity via LLM
└────────┬────────┘
         ▼
┌─────────────────┐
│    Retriever     │  Qdrant semantic search (top-20)
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
┌─────────┐   │
│Citation │   │
│Formatter│   │
└────┬────┘   │
     ▼        ▼
   Answer + Citations
```

## Tech Stack

| Layer | Choice |
|-------|--------|
| Language | Python 3.11+ |
| Agent Framework | LangGraph (StateGraph with conditional edges) |
| LLM | Gemini 1.5 Flash (via google-genai SDK) |
| Embeddings | text-embedding-004 (Gemini) |
| Vector DB | Qdrant (self-hosted Docker) |
| Reranking | Cohere Rerank (free tier) |
| API | FastAPI + Uvicorn |
| PDF Parsing | PyMuPDF (fitz) |
| Auth | API Key middleware (X-API-Key header) |
| Streaming | SSE (Server-Sent Events) |
| Testing | pytest + httpx async client |
| Container | Docker + docker-compose |
| Config | Pydantic Settings v2 |

## Project Structure

```
ragflow-api/
├── app/
│   ├── main.py              # FastAPI app, lifespan, routers
│   ├── config.py            # Pydantic Settings (env-based)
│   ├── models.py            # Pydantic request/response schemas
│   ├── pipeline/
│   │   ├── graph.py         # LangGraph StateGraph definition
│   │   ├── nodes.py         # All 5 node functions
│   │   └── state.py         # RAGState TypedDict
│   ├── services/
│   │   ├── ingestor.py      # PDF parsing → chunking → embed → upsert
│   │   ├── retriever.py     # Qdrant search wrapper
│   │   └── reranker.py      # Cohere rerank wrapper
│   └── middleware/
│       └── auth.py          # API key validation middleware
├── tests/
│   ├── test_ingest.py
│   └── test_query.py
├── frontend/                # Next.js chat UI
├── docker-compose.yml       # Qdrant + API
├── Dockerfile
├── requirements.txt
├── .env.example
└── AGENTS.md
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | API version, Qdrant status, LLM check |
| POST | `/ingest` | Upload PDF → parse → chunk → embed → store |
| POST | `/query` | Full pipeline: rewrite → retrieve → rerank → generate → cite |
| GET | `/query/stream` | SSE streaming version of POST /query |
| GET | `/docs` | List all ingested documents |
| GET | `/docs/{doc_id}` | Get document metadata |
| DELETE | `/docs/{doc_id}` | Remove document + vectors |

## Quick Start

```bash
# 1. Clone and enter project
cd ragflow-api

# 2. Copy env and fill in keys
cp .env.example .env
# Edit .env with your GEMINI_API_KEY, COHERE_API_KEY, API_KEY

# 3. Start Qdrant + API
docker-compose up --build

# 4. Ingest a PDF
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: your-key" \
  -F "file=@document.pdf"

# 5. Ask a question
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: your-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this document about?"}'

# 6. Open frontend
cd frontend && npm ci && npm run dev
```

## Key Design Decisions

1. **LangGraph over raw LangChain agents**: Gives full control over the pipeline topology with conditional edges (e.g., no-context fallback prevents hallucination).

2. **Cohere Rerank is critical**: Raw cosine similarity from vector search is noisy. Cohere reranking improves NDCG@5 by ~30-40% — this is the single biggest quality lever.

3. **Conditional edge for hallucination guard**: If reranker scores are all < 0.3, the pipeline returns "I don't know" instead of fabricating.

4. **Gemini text-embedding-004 for embeddings**: Matches the LLM provider, reducing API surface. Uses OpenAI-compatible endpoint for easy swap.

5. **SSE streaming**: Token-by-token output via Server-Sent Events for responsive UX, matching the pattern used in chat-langchain.

6. **chunking strategy**: Paragraph-boundary-first then token-limit (512 tokens, 50 overlap). Section headers extracted from PyMuPDF bold/large text detection.

## Build Notes

- Built from scratch based on the checkFile.md spec and patterns from langchain-ai/chat-langchain
- The embedding dimension is set to 768 (Gemini text-embedding-004). Adjust if swapping providers.
- Qdrant collection is auto-created on first ingest if it doesn't exist.
- The frontend proxies API calls through Next.js rewrites to avoid CORS in dev.
- For production, add rate limiting, proper secret management, and OpenTelemetry tracing.

## Stretch Features (Not Yet Implemented)

- Multi-document cross-referencing queries
- `/eval` endpoint with RAGAS metrics (faithfulness, relevancy)
- Redis caching for embeddings/query results
- OpenTelemetry tracing across pipeline nodes
- Rate limiting middleware
