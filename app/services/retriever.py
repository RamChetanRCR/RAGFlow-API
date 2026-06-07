from openai import OpenAI

from app.config import get_settings
from app.models import Chunk
from app.services.chromadb_service import get_or_create_collection


class Retriever:
    def __init__(self):
        self.settings = get_settings()
        self._embedding_client = None

    @property
    def collection(self):
        return get_or_create_collection()

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
        where_filter = {"doc_id": doc_id} if doc_id else None

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["metadatas", "distances"],
        )

        chunks = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            score = 1.0 - distance
            chunk = Chunk(
                doc_id=meta.get("doc_id", ""),
                page_number=meta.get("page_number", 0),
                chunk_index=meta.get("chunk_index", 0),
                char_offset=meta.get("char_offset", 0),
                text=meta.get("text", ""),
                section_header=meta.get("section_header", ""),
                score=score,
            )
            chunks.append(chunk)

        return chunks
