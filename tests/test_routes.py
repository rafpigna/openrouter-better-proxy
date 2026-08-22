"""Tests for HTTP routes."""

import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from routes import app, init_routes, ChatCompletionRequest
from router import Router
from cache import EndpointCache
from backoff import BackoffManager
from session import SessionManager


@pytest.fixture(autouse=True)
def setup_env():
    """Set dummy API key for all tests."""
    os.environ["OPENROUTER_API_KEY"] = "dummy-key-for-testing"
    yield
    os.environ.pop("OPENROUTER_API_KEY", None)


@pytest.fixture
def components():
    """Create router components for testing."""
    backoff = BackoffManager()
    sessions = SessionManager()
    cache = EndpointCache(data_dir="/tmp/test-cache-routes")
    router = Router(backoff, sessions, cache)
    init_routes(router, cache)
    return router, sessions, cache


@pytest.fixture
def client(components):
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestModelsEndpoint:
    """Test GET /v1/models endpoint."""

    def test_list_models(self, client):
        """Test that configured models are returned."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        model_ids = [m["id"] for m in data["data"]]
        assert "deepseek/deepseek-v4-flash-0731" in model_ids


class TestChatCompletionsEndpoint:
    """Test POST /v1/chat/completions endpoint."""

    def test_selects_provider(self, client, components):
        """Test that provider is selected and request is forwarded."""
        router, sessions, cache = components

        # Mock endpoints
        from datetime import datetime
        endpoints = [
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.080", "completion": "0.18", "input_cache_read": "0.016"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        # Mock httpx response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "gen-123",
            "model": "deepseek/deepseek-v4-flash-0731",
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch("routes.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek/deepseek-v4-flash-0731",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )

            assert resp.status_code == 200
            data = resp.json()
            assert data["choices"][0]["message"]["content"] == "Hello"

            # Verify upstream call
            assert mock_client.post.called
            call_args = mock_client.post.call_args
            assert "openrouter.ai" in call_args[0][0]
            body = call_args[1]["json"]
            assert body["provider"] == {"only": ["deepinfra/fp8"]}

    def test_session_stickiness(self, client, components):
        """Test that session_id is forwarded."""
        router, sessions, cache = components

        from datetime import datetime
        endpoints = [
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.080", "completion": "0.18", "input_cache_read": "0.016"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "gen-123",
            "model": "deepseek/deepseek-v4-flash-0731",
            "choices": [{"message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        with patch("routes.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(return_value=mock_response)
            MockClient.return_value = mock_client

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek/deepseek-v4-flash-0731",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "session_id": "test-session-123",
                },
            )

            assert resp.status_code == 200

            # Verify session_id was forwarded
            call_args = mock_client.post.call_args
            body = call_args[1]["json"]
            assert body["session_id"] == "test-session-123"

    def test_no_provider_returns_400(self, client, components):
        """Test that missing provider returns 400."""
        router, sessions, cache = components
        # No endpoints cached

        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "nonexistent/model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

        assert resp.status_code == 400


class TestRefreshEndpoint:
    """Test POST /refresh endpoint."""

    def test_refresh(self, client, components):
        """Test manual refresh trigger."""
        from datetime import datetime
        from cache import EndpointCache

        cache = components[2]
        endpoints = [
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.080", "completion": "0.18", "input_cache_read": "0.016"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        resp = client.post("/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "deepseek/deepseek-v4-flash-0731" in data["refreshed_models"]


class TestStatusEndpoint:
    """Test GET /status endpoint."""

    def test_status(self, client, components):
        """Test status endpoint."""
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "backoff" in data
        assert "cached_models" in data
