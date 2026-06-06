import uuid
from datetime import datetime, timezone
from typing import Optional

import fitz
from qdrant_client.http import models as qmodels
from openai import OpenAI

from app.config import get_settings
from app.models import Chunk
from app.services.qdrant import get_qdrant


class Ingestor:
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

    def ensure_collection(self):
        collections = self.qdrant.get_collections().collections
        exists = any(c.name == self.settings.collection_name for c in collections)
        if not exists:
            self.qdrant.create_collection(
                collection_name=self.settings.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.settings.embedding_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def parse_pdf(self, filepath: str) -> tuple[list[dict], int]:
        doc = fitz.open(filepath)
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            blocks = page.get_text("dict")["blocks"]
            section_headers = []
            for block in blocks:
                if block.get("type") == 0:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            if span.get("size", 0) > 14 or "bold" in span.get("font", "").lower():
                                section_headers.append(span["text"])
            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "section_header": section_headers[0] if section_headers else "",
            })
        return pages, len(doc)

    def chunk_text(self, text: str, section_header: str = "") -> list[str]:
        settings = self.settings
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = len(para.split())
            if current_tokens + para_tokens > settings.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_words = current_chunk.split()[-settings.chunk_overlap:]
                current_chunk = " ".join(overlap_words) + "\n\n" + para
                current_tokens = len(current_chunk.split())
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_tokens += para_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.embedding_client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def ingest(self, filepath: str, filename: str) -> dict:
        self.ensure_collection()
        doc_id = str(uuid.uuid4())
        pages, page_count = self.parse_pdf(filepath)
        points = []
        chunk_count = 0

        for page in pages:
            raw_chunks = self.chunk_text(
                page["text"],
                section_header=page["section_header"],
            )
            chunk_texts = []
            for ci, chunk_text in enumerate(raw_chunks):
                chunk = Chunk(
                    doc_id=doc_id,
                    page_number=page["page_number"],
                    chunk_index=chunk_count,
                    char_offset=0,
                    text=chunk_text,
                    section_header=page["section_header"],
                )
                chunk_texts.append(chunk)

            if chunk_texts:
                embeddings = self.embed_texts([c.text for c in chunk_texts])
                for chunk, embedding in zip(chunk_texts, embeddings):
                    points.append(qmodels.PointStruct(
                        id=chunk_count,
                        vector=embedding,
                        payload={
                            "doc_id": chunk.doc_id,
                            "page_number": chunk.page_number,
                            "chunk_index": chunk.chunk_index,
                            "text": chunk.text,
                            "section_header": chunk.section_header,
                            "filename": filename,
                        },
                    ))
                    chunk_count += 1

        if points:
            self.qdrant.upsert(
                collection_name=self.settings.collection_name,
                points=points,
            )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_doc_info(self, doc_id: str) -> Optional[dict]:
        response = self.qdrant.scroll(
            collection_name=self.settings.collection_name,
            scroll_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="doc_id",
                        match=qmodels.MatchValue(value=doc_id),
                    )
                ],
            ),
            limit=1,
        )
        if not response[0]:
            return None
        payload = response[0][0].payload
        all_points = self.qdrant.scroll(
            collection_name=self.settings.collection_name,
            scroll_filter=qmodels.Filter(
                must=[qmodels.FieldCondition(
                    key="doc_id",
                    match=qmodels.MatchValue(value=doc_id),
                )],
            ),
            limit=10000,
        )[0]
        page_numbers = set(p.payload.get("page_number") for p in all_points)
        return {
            "doc_id": doc_id,
            "filename": payload.get("filename", ""),
            "page_count": len(page_numbers),
            "ingestion_timestamp": "",
            "chunk_count": self.qdrant.count(
                collection_name=self.settings.collection_name,
                count_filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(
                        key="doc_id",
                        match=qmodels.MatchValue(value=doc_id),
                    )],
                ),
            ).count,
        }

    def list_docs(self) -> list[dict]:
        response = self.qdrant.scroll(
            collection_name=self.settings.collection_name,
            limit=10000,
            with_payload=["doc_id", "filename"],
            with_vectors=False,
        )
        seen = {}
        for point in response[0]:
            pid = point.payload.get("doc_id")
            if pid and pid not in seen:
                seen[pid] = {
                    "doc_id": pid,
                    "filename": point.payload.get("filename", ""),
                }
        return list(seen.values())

    def delete_doc(self, doc_id: str) -> bool:
        try:
            self.qdrant.delete(
                collection_name=self.settings.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[qmodels.FieldCondition(
                            key="doc_id",
                            match=qmodels.MatchValue(value=doc_id),
                        )],
                    ),
                ),
            )
            return True
        except Exception:
            return False

    def check_health(self) -> bool:
        try:
            self.qdrant.get_collections()
            return True
        except Exception:
            return False
