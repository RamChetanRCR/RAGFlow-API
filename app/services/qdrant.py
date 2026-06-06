from qdrant_client import QdrantClient

_qdrant_instance = None


def get_qdrant() -> QdrantClient:
    global _qdrant_instance
    if _qdrant_instance is None:
        from app.config import get_settings
        settings = get_settings()
        url = settings.qdrant_url
        if url == ":memory:":
            _qdrant_instance = QdrantClient(":memory:")
        else:
            _qdrant_instance = QdrantClient(
                url=url,
                api_key=settings.qdrant_api_key or None,
                check_compatibility=False,
            )
    return _qdrant_instance
