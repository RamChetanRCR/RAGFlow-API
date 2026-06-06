from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    gemini_api_key: str = ""
    cohere_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    api_key: str = "changeme_in_prod"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.2:3b"
    llm_base_url: str = "http://localhost:11434/v1"
    embedding_base_url: str = "http://localhost:11434/v1"
    top_k_retrieve: int = 20
    top_k_rerank: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    collection_name: str = "documents"
    embedding_size: int = 768

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()
