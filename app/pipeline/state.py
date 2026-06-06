from typing import TypedDict
from app.models import Chunk, Citation


class RAGState(TypedDict):
    query: str
    rewritten_query: str
    retrieved_chunks: list[Chunk]
    reranked_chunks: list[Chunk]
    answer: str
    citations: list[Citation]
    should_retrieve: bool
    doc_id: str
