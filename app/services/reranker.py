from typing import Optional
import cohere

from app.config import get_settings
from app.models import Chunk


class Reranker:
    def __init__(self):
        self.settings = get_settings()
        self.client: Optional[cohere.ClientV2] = None
        if self.settings.cohere_api_key:
            self.client = cohere.ClientV2(self.settings.cohere_api_key)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int = 0) -> list[Chunk]:
        if top_k == 0:
            top_k = self.settings.top_k_rerank

        if not self.client or not chunks:
            for c in chunks:
                c.relevance_score = c.score
            return chunks[:top_k]

        documents = [c.text for c in chunks]
        response = self.client.rerank(
            query=query,
            documents=documents,
            top_n=top_k,
            model="rerank-english-v3.0",
        )

        reranked = []
        for result in response.results:
            chunk = chunks[result.index]
            chunk.relevance_score = result.relevance_score
            reranked.append(chunk)

        reranked.sort(key=lambda c: c.relevance_score, reverse=True)
        return reranked
