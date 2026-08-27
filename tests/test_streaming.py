"""Tests for streaming functionality.

Aligned to the current routes.py API (2026-08-27):
- _forward_stream lives in routes.py (not main.py) and takes
  (body: dict, candidates: list[tuple[str, str]])
- the SSE stream is consumed via client.stream(...) context manager
"""

import pytest
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

import routes as routes_module

MODEL = "deepseek/deepseek-v4-flash-0731"


@pytest.fixture(autouse=True)
def _api_key_env():
    """Provide a dummy API key for all streaming tests."""
    os.environ["OPENROUTER_API_KEY"] = "dummy-key-for-testing"
    yield
    os.environ.pop("OPENROUTER_API_KEY", None)

ENDPOINTS = {
    "endpoints": [
        {
            "tag": "deepinfra/fp8",
            "quantization": "fp8",
            "status": 0,
            "pricing": {
                # Per-token strings (deepinfra/fp8 = $0.08/M input)
                "prompt": "0.00000008",
                "completion": "0.00000018",
                "input_cache_read": "0.000000016",
            },
        }
    ],
    "fetched_at": datetime.now(timezone.utc).isoformat(),
}


def _make_stream_response(chunks):
    """MagicMock of an httpx streaming 200 response with aiter_lines."""
    mock_response = MagicMock()
    mock_response.status_code = 200

    async def aiter_lines():
        for c in chunks:
            yield c

    mock_response.aiter_lines = aiter_lines
    return mock_response


SSE_CHUNKS = [
    f'data: {{"id":"t1","provider":"deepinfra","model":"{MODEL}",'
    '"choices":[{"delta":{"content":"Hi"}}]}',
    'data: {"id":"t1","choices":[],"usage":{"prompt_tokens":3,'
    '"completion_tokens":1,"total_tokens":4}}',
    "data: [DONE]",
]


def _capturing_client_factory(captured):
    """Return a factory building an httpx.AsyncClient-like mock whose
    .stream() (sync call returning an async CM) records the upstream JSON
    body into `captured`."""

    def factory():
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=_make_stream_response(SSE_CHUNKS))
        ctx.__aexit__ = AsyncMock(return_value=False)

        def fake_stream(method, url, json=None, headers=None):
            captured["upstream_body"] = json
            captured["url"] = url
            return ctx

        mock_client.stream = MagicMock(side_effect=fake_stream)
        return mock_client

    return factory


@pytest.mark.asyncio
async def test_streaming_returns_sse_format():
    """Test that streaming returns proper SSE format end-to-end."""
    from fastapi.testclient import TestClient
    from main import app
    from router import Router
    from cache import EndpointCache
    from backoff import BackoffManager
    from session import SessionManager

    cache = EndpointCache(data_dir="data/test-cache-streaming")
    cache.set(MODEL, ENDPOINTS)
    router = Router(BackoffManager(), SessionManager(), cache)
    routes_module.init_routes(router, cache)

    with patch("routes.httpx.AsyncClient") as MockClient:
        stream_ctx = MagicMock()
        stream_ctx.__aenter__ = AsyncMock(
            return_value=_make_stream_response(SSE_CHUNKS)
        )
        stream_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.stream = MagicMock(return_value=stream_ctx)
        MockClient.return_value = mock_client

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        content = response.content.decode()
        assert "data:" in content
        assert "[DONE]" in content


@pytest.mark.asyncio
async def test_non_streaming_returns_json():
    """Test that non-streaming returns JSON."""
    from fastapi.testclient import TestClient
    from main import app
    from router import Router
    from cache import EndpointCache
    from backoff import BackoffManager
    from session import SessionManager

    cache = EndpointCache(data_dir="data/test-cache-nonstreaming")
    cache.set(MODEL, ENDPOINTS)
    router = Router(BackoffManager(), SessionManager(), cache)
    routes_module.init_routes(router, cache)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.aread = AsyncMock(return_value=b"")
    mock_response.json.return_value = {
        "id": "test",
        "choices": [{"message": {"content": "Hello"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }

    with patch("routes.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["content"] == "Hello"


@pytest.mark.asyncio
async def test_streaming_forwards_session_id():
    """Test that session_id reaches upstream inside the pinned body."""
    from router import Router
    from cache import EndpointCache
    from backoff import BackoffManager
    from session import SessionManager

    cache = EndpointCache(data_dir="data/test-cache-fwd")
    cache.set(MODEL, ENDPOINTS)
    router = Router(BackoffManager(), SessionManager(), cache)
    routes_module.init_routes(router, cache)

    captured = {}

    with patch("routes.httpx.AsyncClient") as mock_client_class:
        mock_client_class.side_effect = None
        mock_client_class.return_value = _capturing_client_factory(captured)()

        body = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "session_id": "test-session-123",
        }

        generator = routes_module._forward_stream(body, [("deepinfra/fp8", "fp8")])
        chunks = [chunk async for chunk in generator]

        assert chunks, "no chunks produced"
        upstream_json = captured.get("upstream_body") or {}
        assert upstream_json.get("session_id") == "test-session-123"
        # Verbatim passthrough + routing pin only
        assert upstream_json.get("provider") == {"only": ["deepinfra/fp8"]}
