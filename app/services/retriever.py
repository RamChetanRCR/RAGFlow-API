from qdrant_client import models
from openai import OpenAI

from app.config import get_settings
from app.models import Chunk
from app.services.qdrant import get_qdrant


class Retriever:
    def __init__(self):
        self.settings = get_settings()
        self._embedding_client = None

    @property
    def qdrant(self):
        return get_qdrant()

    @property
    def embedding_client(self):
        if self._embedding_client is None:
            self._embedding_client = OpenAI(
                base_url=self.settings.embedding_base_url,
                api_key=self.settings.gemini_api_key or "no-key-set",
            )
        return self._embedding_client

    def embed_query(self, query: str) -> list[float]:
        response = self.embedding_client.embeddings.create(
            model=self.settings.embedding_model,
            input=[query],
        )
        return response.data[0].embedding

    def search(self, query: str, doc_id: str = "", top_k: int = 0) -> list[Chunk]:
        if top_k == 0:
            top_k = self.settings.top_k_retrieve
        query_vector = self.embed_query(query)
        filter_condition = None
        if doc_id:
            filter_condition = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ],
            )

        results = self.qdrant.query_points(
            collection_name=self.settings.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=filter_condition,
            with_payload=True,
        )

        chunks = []
        for r in results.points:
            chunk = Chunk(
                doc_id=r.payload.get("doc_id", ""),
                page_number=r.payload.get("page_number", 0),
                chunk_index=r.payload.get("chunk_index", 0),
                char_offset=r.payload.get("char_offset", 0),
                text=r.payload.get("text", ""),
                section_header=r.payload.get("section_header", ""),
                score=r.score,
            )
            chunks.append(chunk)

        return chunks
