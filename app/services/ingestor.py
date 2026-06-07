import uuid
from datetime import datetime, timezone
from typing import Optional

import fitz
from openai import OpenAI

from app.config import get_settings
from app.models import Chunk
from app.services.chromadb_service import get_or_create_collection


class Ingestor:
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
        doc_id = str(uuid.uuid4())
        pages, page_count = self.parse_pdf(filepath)
        chunk_count = 0
        all_ids = []
        all_embeddings = []
        all_metadatas = []

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
                    all_ids.append(str(chunk_count))
                    all_embeddings.append(embedding)
                    all_metadatas.append({
                        "doc_id": chunk.doc_id,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                        "text": chunk.text,
                        "section_header": chunk.section_header,
                        "filename": filename,
                    })
                    chunk_count += 1

        if all_ids:
            self.collection.add(
                embeddings=all_embeddings,
                metadatas=all_metadatas,
                ids=all_ids,
            )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "page_count": page_count,
            "chunk_count": chunk_count,
            "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_doc_info(self, doc_id: str) -> Optional[dict]:
        try:
            results = self.collection.get(where={"doc_id": doc_id})
        except ValueError:
            return None
        if not results["ids"]:
            return None
        page_numbers = set(m.get("page_number") for m in results["metadatas"])
        filename = results["metadatas"][0].get("filename", "")
        return {
            "doc_id": doc_id,
            "filename": filename,
            "page_count": len(page_numbers),
            "ingestion_timestamp": "",
            "chunk_count": len(results["ids"]),
        }

    def list_docs(self) -> list[dict]:
        try:
            results = self.collection.get(limit=10000)
        except ValueError:
            return []
        seen = {}
        for meta in results["metadatas"]:
            pid = meta.get("doc_id")
            if pid and pid not in seen:
                seen[pid] = {
                    "doc_id": pid,
                    "filename": meta.get("filename", ""),
                }
        return list(seen.values())

    def delete_doc(self, doc_id: str) -> bool:
        try:
            self.collection.delete(where={"doc_id": doc_id})
            return True
        except Exception:
            return False

    def check_health(self) -> bool:
        try:
            self.collection
            return True
        except Exception:
            return False
