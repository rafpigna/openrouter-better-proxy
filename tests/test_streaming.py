"""Tests for streaming functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import json


@pytest.mark.asyncio
async def test_streaming_returns_sse_format():
    """Test that streaming returns proper SSE format."""
    from fastapi.testclient import TestClient
    from main import app

    # Mock the router and cache
    with patch('routes._router') as mock_router, \
         patch('routes._endpoint_cache'), \
         patch('routes.config') as mock_config:

        mock_router.select_provider.return_value = ("deepinfra/fp8", "fp8")
        mock_router.record_success = MagicMock()
        mock_router.record_error = MagicMock()
        mock_config.openrouter_api_key = "test-key"

        # Mock httpx response with SSE chunks
        sse_chunks = [
            'data: {"id":"test","choices":[{"delta":{"content":"Hello"}}]}',
            'data: {"id":"test","choices":[{"delta":{"content":" world"}}]}',
            'data: [DONE]',
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.aiter_lines.return_value = AsyncMock(__aiter__=lambda self: iter(sse_chunks))

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_client.stream.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_client_class.return_value = mock_client

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek/deepseek-v4-flash-0731",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

            # Check SSE format
            content = response.content.decode()
            assert "data:" in content
            assert "[DONE]" in content


@pytest.mark.asyncio
async def test_non_streaming_returns_json():
    """Test that non-streaming returns JSON."""
    from fastapi.testclient import TestClient
    from main import app

    with patch('routes._router') as mock_router, \
         patch('routes._endpoint_cache'), \
         patch('routes.config') as mock_config:

        mock_router.select_provider.return_value = ("deepinfra/fp8", "fp8")
        mock_router.record_success = MagicMock()
        mock_router.record_error = MagicMock()
        mock_config.openrouter_api_key = "test-key"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "test",
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }

        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            client = TestClient(app)
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek/deepseek-v4-flash-0731",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": False,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == "Hello"


@pytest.mark.asyncio
async def test_streaming_forwards_session_id():
    """Test that session_id is forwarded to upstream."""
    from main import _forward_stream
    from router import Router
    from cache import EndpointCache
    from config import Config

    router = Router(Config())
    cache = EndpointCache()
    init_routes(router, cache)

    with patch('routes.config') as mock_config, \
         patch('httpx.AsyncClient') as mock_client_class:

        mock_config.openrouter_api_key = "test-key"

        # Capture the request body
        captured_body = {}

        async def mock_post(url, json=None, headers=None):
            captured_body.update(json or {})
            mock_resp = MagicMock()
            mock_resp.status_code = 200

            async def aiter_lines():
                yield 'data: {"id":"test","choices":[{"delta":{"content":"Hi"}}]}'
                yield 'data: [DONE]'

            mock_resp.aiter_lines = aiter_lines
            return mock_resp

        mock_client = AsyncMock()
        mock_client.stream = AsyncMock(return_value=MagicMock(
            __aenter__=AsyncMock(return_value=MagicMock(aiter_lines=mock_client.stream)),
            __aexit__=AsyncMock(return_value=False)
        ))
        mock_client_class.return_value = mock_client

        # Test with session_id
        body = {
            "model": "deepseek/deepseek-v4-flash-0731",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
            "session_id": "test-session-123",
        }

        generator = _forward_stream(body, "deepinfra/fp8")
        chunks = [chunk async for chunk in generator]

        # Verify session_id was in the request
        assert captured_body.get("session_id") == "test-session-123"
