import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.config import get_settings


@pytest.fixture
def settings():
    s = get_settings()
    s.api_key = "test-key"
    return s


@pytest.fixture
def headers(settings):
    return {"X-API-Key": "test-key"}


@pytest.mark.asyncio
async def test_health_no_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_with_auth(headers):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health", headers=headers)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_query_no_auth():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={"query": "test"})
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_query_empty_docs(headers):
    mock_response = MagicMock()
    mock_response.text = "Test answer with citation [doc1, p.1]"

    with (
        patch("app.pipeline.nodes.get_llm") as mock_get_llm,
        patch("app.services.retriever.Retriever.search", return_value=[]),
    ):
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_get_llm.return_value = mock_client

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/query",
                json={"query": "what is this document about?"},
                headers=headers,
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
