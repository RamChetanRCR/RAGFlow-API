import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

_chroma_instance = None


def get_chroma():
    global _chroma_instance
    if _chroma_instance is None:
        from app.config import get_settings
        settings = get_settings()
        persist_dir = settings.chroma_persist_directory
        if persist_dir == ":memory:":
            _chroma_instance = chromadb.Client(Settings(anonymized_telemetry=False))
        else:
            _chroma_instance = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
    return _chroma_instance


def get_or_create_collection():
    client = get_chroma()
    from app.config import get_settings
    settings = get_settings()
    try:
        return client.get_collection(settings.chroma_collection_name)
    except NotFoundError:
        return client.create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
