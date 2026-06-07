import json
import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.models import QueryRequest, QueryResponse, IngestResponse, DocInfo, HealthResponse, AboutResponse
from app.pipeline.graph import rag_graph
from app.pipeline.nodes import answer_generator
from app.services.ingestor import Ingestor
from app.services.retriever import Retriever
from app.services.reranker import Reranker

settings = get_settings()
ingestor = Ingestor()
retriever = Retriever()
reranker = Reranker()


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("uploads", exist_ok=True)
    yield


app = FastAPI(
    title="RAGFlow API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/about", response_model=AboutResponse)
async def about():
    try:
        docs = ingestor.list_docs()
        docs_count = len(docs)
    except ValueError:
        docs_count = 0
    return AboutResponse(
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
        docs_count=docs_count,
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    chroma_ok = ingestor.check_health()
    return HealthResponse(
        chroma_status="ok" if chroma_ok else "unreachable",
        llm_status="configured" if settings.llm_base_url else "missing config",
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    suffix = ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = ingestor.ingest(tmp_path, file.filename)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    state = {
        "query": body.query,
        "rewritten_query": "",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "citations": [],
        "doc_id": body.doc_id or "",
        "should_retrieve": True,
        "guardrails_passed": True,
        "guardrails_reason": "",
        "retrieval_strategy": "semantic",
        "quality_score": 0,
        "quality_passed": True,
        "max_retries": 0,
    }

    result = await rag_graph.ainvoke(state)
    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        rewritten_query=result.get("rewritten_query", ""),
    )


@app.get("/query/stream")
async def query_stream(query: str, doc_id: str = ""):
    from app.pipeline.nodes import get_llm

    llm = get_llm()
    rewritten_resp = llm.chat.completions.create(
        model=settings.llm_model,
        messages=[{
            "role": "user",
            "content": (
                "You are a query rewriter. Rewrite this query to be more precise "
                "for retrieval. Output ONLY the rewritten query.\n\n"
                f"Query: {query}\nRewritten query:"
            ),
        }],
        temperature=0.1,
    )
    rewritten_query = rewritten_resp.choices[0].message.content.strip().strip('"') or query

    retrieved = retriever.search(rewritten_query, doc_id=doc_id)
    reranked = reranker.rerank(rewritten_query, retrieved)

    if not reranked or max((c.relevance_score for c in reranked), default=0) < 0.3:
        async def fallback_stream():
            yield f"data: {json.dumps({'token': "I don't have information on this."})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(fallback_stream(), media_type="text/event-stream")

    state = {
        "query": query,
        "rewritten_query": rewritten_query,
        "retrieved_chunks": retrieved,
        "reranked_chunks": reranked,
        "answer": "",
        "citations": [],
        "doc_id": doc_id,
        "should_retrieve": True,
        "guardrails_passed": True,
        "guardrails_reason": "",
        "retrieval_strategy": "semantic",
        "quality_score": 0,
        "quality_passed": True,
        "max_retries": 0,
    }

    return StreamingResponse(
        answer_generator(state),
        media_type="text/event-stream",
    )


@app.get("/docs", response_model=list[DocInfo])
async def list_docs():
    docs = ingestor.list_docs()
    return [DocInfo(**d) for d in docs]


@app.get("/docs/{doc_id}", response_model=DocInfo)
async def get_doc(doc_id: str):
    info = ingestor.get_doc_info(doc_id)
    if not info:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocInfo(**info)


@app.delete("/docs/{doc_id}")
async def delete_doc(doc_id: str):
    success = ingestor.delete_doc(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": f"Document {doc_id} deleted"}


@app.post("/eval")
async def evaluate():
    raise HTTPException(status_code=501, detail="Evaluation endpoint not yet implemented")
