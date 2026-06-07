from pydantic import BaseModel
from typing import Optional


class Chunk(BaseModel):
    doc_id: str
    page_number: int
    chunk_index: int
    char_offset: int
    text: str
    section_header: str = ""
    score: float = 0.0
    relevance_score: float = 0.0


class Citation(BaseModel):
    doc_name: str
    page: int
    text: str


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    chunk_count: int
    message: str = ""


class QueryRequest(BaseModel):
    query: str
    doc_id: Optional[str] = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    rewritten_query: str = ""


class DocInfo(BaseModel):
    doc_id: str
    filename: str
    page_count: int
    ingestion_timestamp: str
    chunk_count: int


class HealthResponse(BaseModel):
    version: str = "1.0.0"
    chroma_status: str = ""
    llm_status: str = ""


class AboutResponse(BaseModel):
    project: str = "RAGFlow API"
    description: str = "Multi-agent RAG pipeline for PDF Q&A with LangGraph"
    version: str = "1.0.0"
    llm_model: str = ""
    embedding_model: str = ""
    vector_db: str = "ChromaDB (:memory:)"
    reranker: str = "Cohere (optional, falls back to cosine)"
    agent_count: int = 4
    agents: list[str] = [
        "Guardrails Agent",
        "Query Rewriter",
        "Retrieval Strategist",
        "Quality Reviewer",
    ]
    docs_count: int = 0
