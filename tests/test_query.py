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
async def test_query_basic():
    with (
        patch("app.pipeline.nodes.get_llm") as mock_get_llm,
        patch("app.services.retriever.Retriever.search", return_value=[]),
    ):
        responses = iter([
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"passed": true, "reason": "on topic"}'))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="rewritten test query"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"strategy": "semantic", "doc_id": ""}'))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="I don't have information on this."))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"score": 7, "passed": true, "reason": "ok"}'))]),
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: next(responses)
        mock_get_llm.return_value = mock_client

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/query", json={"query": "test"})
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data


@pytest.mark.asyncio
async def test_query_empty_docs():
    with (
        patch("app.pipeline.nodes.get_llm") as mock_get_llm,
        patch("app.services.retriever.Retriever.search", return_value=[]),
    ):
        responses = iter([
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"passed": true, "reason": "on topic"}'))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="rewritten test query"))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"strategy": "semantic", "doc_id": ""}'))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content="I don't have information on this."))]),
            MagicMock(choices=[MagicMock(message=MagicMock(content='{"score": 7, "passed": true, "reason": "ok"}'))]),
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = lambda *a, **kw: next(responses)
        mock_get_llm.return_value = mock_client

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/query",
                json={"query": "what is this document about?"},
            )
            assert response.status_code == 200
            data = response.json()
            assert "answer" in data
