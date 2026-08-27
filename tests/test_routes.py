"""Tests for HTTP routes.

Aligned to the current routes.py API (2026-08-27):
- app is built in main.py; routes.py exposes an APIRouter
- chat_completions accepts `request: dict` (no pydantic model)
- VERBATIM PASSTHROUGH: the upstream body must equal the incoming body
  plus ONLY the per-attempt `provider` pin.
"""

import pytest
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from main import app
from routes import init_routes
import routes as routes_module
from router import Router
from cache import EndpointCache
from backoff import BackoffManager
from session import SessionManager

MODEL = "deepseek/deepseek-v4-flash-0731"


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
    cache = EndpointCache(data_dir="data/test-cache-routes")
    router = Router(backoff, sessions, cache)
    init_routes(router, cache)
    return router, sessions, cache


@pytest.fixture
def client(components):
    """Create test client."""
    return TestClient(app)


def _seed_cache(cache):
    """Seed the endpoint cache with one authorized provider.

    Prices are PER-TOKEN strings as returned by OpenRouter
    (deepinfra/fp8 = $0.08/M input). Router converts to $/M for the
    max_price gate (DESIGN Appendix A.4).
    """
    cache.set(
        MODEL,
        {
            "endpoints": [
                {
                    "tag": "deepinfra/fp8",
                    "quantization": "fp8",
                    "status": 0,
                    "pricing": {
                        "prompt": "0.00000008",
                        "completion": "0.00000018",
                        "input_cache_read": "0.000000016",
                    },
                }
            ],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
    )


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
        assert MODEL in model_ids


def _mock_upstream_json_response():
    """Standard mocked 200 non-streaming OpenRouter response."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.aread = AsyncMock(return_value=b"")
    mock_response.json.return_value = {
        "id": "gen-123",
        "model": MODEL,
        "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    return mock_response


def _mock_httpx_client(response=None):
    """Async httpx client mock for non-streaming calls."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=response or _mock_upstream_json_response())
    return mock_client


class TestChatCompletionsEndpoint:
    """Test POST /v1/chat/completions endpoint."""

    def test_selects_provider(self, client, components):
        """Test that provider is selected and request is forwarded."""
        router, sessions, cache = components
        _seed_cache(cache)

        with patch("routes.httpx.AsyncClient") as MockClient:
            mock_client = _mock_httpx_client()
            MockClient.return_value = mock_client

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL,
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
        _seed_cache(cache)

        with patch("routes.httpx.AsyncClient") as MockClient:
            mock_client = _mock_httpx_client()
            MockClient.return_value = mock_client

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL,
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


class TestVerbatimPassthrough:
    """VERBATIM PASSTHROUGH contract: routing-only proxy.

    The proxy must forward the request CONTENT unchanged. The ONLY allowed
    difference in the upstream body is the `provider` pin added by the
    forward loop (routing concern). Headers follow the same principle:
    everything except hop-by-hop/managed headers goes through verbatim
    (app attribution HTTP-Referer / X-Title included).
    """

    EXTRA_FIELDS_BODY = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ],
        "temperature": 0.42,
        "top_p": 0.9,
        "max_tokens": 555,
        "stop": ["END"],
        "seed": 1234,
        "response_format": {"type": "text"},
        "reasoning": {"effort": "medium"},
        "session_id": "sess-passthrough-test",
        "some_future_field": {"nested": [1, 2, 3]},
    }

    ATTRIBUTION_HEADERS = {
        "HTTP-Referer": "https://hermes.local",
        "X-Title": "Hermes Agent",
        "X-Custom-Trace": "trace-42",
    }

    def test_non_streaming_body_is_verbatim_plus_provider_only(
        self, client, components
    ):
        """Upstream JSON body == incoming body + provider pin. Nothing dropped,
        nothing rewritten."""
        router, sessions, cache = components
        _seed_cache(cache)

        captured = {}

        with patch("routes.httpx.AsyncClient") as MockClient:
            mock_client = _mock_httpx_client()
            original_post = mock_client.post

            async def capture_post(url, json=None, headers=None):
                captured.update(json or {})
                return await original_post(url, json=json, headers=headers)

            mock_client.post = AsyncMock(side_effect=capture_post)
            MockClient.return_value = mock_client

            incoming = dict(self.EXTRA_FIELDS_BODY)
            resp = client.post("/v1/chat/completions", json=incoming)

            assert resp.status_code == 200

            expected = dict(incoming)
            expected["provider"] = {"only": ["deepinfra/fp8"]}
            assert captured == expected, (
                "Upstream body diverges from incoming body: "
                f"missing={ {k: v for k, v in expected.items() if k not in captured} }, "
                f"extra={ {k: v for k, v in captured.items() if k not in expected} }"
            )
            # Explicit: no field of the incoming request was lost
            for key, value in incoming.items():
                assert captured.get(key) == value, (
                    f"Field '{key}' was altered by the proxy"
                )

    def test_unknown_future_field_reaches_openrouter(self, client, components):
        """Unknown fields (OpenAI params added later, custom fields) must not
        be silently filtered out anymore."""
        router, sessions, cache = components
        _seed_cache(cache)

        captured = {}
        with patch("routes.httpx.AsyncClient") as MockClient:
            response_obj = _mock_upstream_json_response()

            async def capture_post(url, json=None, headers=None):
                captured.update(json or {})
                return response_obj

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=capture_post)
            MockClient.return_value = mock_client

            weird = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "brand_new_param_2099": True,
            }
            resp = client.post("/v1/chat/completions", json=weird)

            assert resp.status_code == 200
            assert captured.get("brand_new_param_2099") is True

    def test_app_attribution_headers_reach_openrouter(self, client, components):
        """HTTP-Referer / X-Title / custom headers are forwarded verbatim so
        OpenRouter shows the real app instead of 'Unknown'; hop-by-hop and
        request-managed headers do NOT leak upstream."""
        router, sessions, cache = components
        _seed_cache(cache)

        captured_headers = {}
        with patch("routes.httpx.AsyncClient") as MockClient:
            response_obj = _mock_upstream_json_response()

            async def capture_post(url, json=None, headers=None):
                captured_headers.update(headers or {})
                return response_obj

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=capture_post)
            MockClient.return_value = mock_client

            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "session_id": "sess-hdr",
                },
                headers=self.ATTRIBUTION_HEADERS,
            )

            assert resp.status_code == 200

            lower = {k.lower(): v for k, v in captured_headers.items()}
            # Attribution + custom headers forwarded verbatim
            assert lower.get("http-referer") == "https://hermes.local"
            assert lower.get("x-title") == "Hermes Agent"
            assert lower.get("x-custom-trace") == "trace-42"
            # Authorization replaced with the proxy's key (client one stripped)
            assert lower.get("authorization") == "Bearer dummy-key-for-testing"
            # Hop-by-hop / managed headers never forwarded
            for banned in ("host", "content-length", "connection",
                           "keep-alive", "transfer-encoding", "accept-encoding"):
                assert banned not in lower, f"'{banned}' leaked upstream"

    def test_client_authorization_never_leaks(self, client, components):
        """Whatever key the client sends, upstream always carries the
        proxy's own OPENROUTER_API_KEY."""
        router, sessions, cache = components
        _seed_cache(cache)

        captured_headers = {}
        with patch("routes.httpx.AsyncClient") as MockClient:
            response_obj = _mock_upstream_json_response()

            async def capture_post(url, json=None, headers=None):
                captured_headers.update(headers or {})
                return response_obj

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=capture_post)
            MockClient.return_value = mock_client

            resp = client.post(
                "/v1/chat/completions",
                json={"model": MODEL, "messages": [{"role": "user", "content": "Hi"}]},
                headers={"Authorization": "Bearer client-secret-key"},
            )
            assert resp.status_code == 200
            lower = {k.lower(): v for k, v in captured_headers.items()}
            assert lower.get("authorization") == "Bearer dummy-key-for-testing"
            assert "client-secret-key" not in lower.get("authorization", "")


ATTRIB_HEADERS = {
    "HTTP-Referer": "https://openwebui.example",
    "X-Title": "OpenWebUI",
}


class TestAttributionModes:
    """attribution.mode = passthrough|fallback|force (config `attribution:`).

    fallback/force exist for clients that attach app attribution only when
    talking to openrouter.ai directly (they show as 'App: Unknown' through a
    custom base_url otherwise)."""

    def _post_capture(self, client, components, send_client_headers: bool):
        router, sessions, cache = components
        _seed_cache(cache)
        captured_headers = {}
        with patch("routes.httpx.AsyncClient") as MockClient:
            response_obj = _mock_upstream_json_response()

            async def capture_post(url, json=None, headers=None):
                captured_headers.update(headers or {})
                return response_obj

            mock_client = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.post = AsyncMock(side_effect=capture_post)
            MockClient.return_value = mock_client

            extra_headers = ATTRIB_HEADERS if send_client_headers else {}
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "session_id": "sess-attr",
                },
                headers=extra_headers,
            )
            assert resp.status_code == 200
        return {k.lower(): v for k, v in captured_headers.items()}

    def _set_attribution(self, mode: str | None, headers: dict | None):
        """Inject/restore the `attribution:` section of the live Config.raw."""
        from routes import config as live_cfg
        self._saved_attrib = live_cfg.raw.get("attribution")
        if mode is None:
            live_cfg.raw.pop("attribution", None)
        else:
            live_cfg.raw["attribution"] = {"mode": mode, "headers": headers or {}}

    def _restore_attribution(self):
        from routes import config as live_cfg
        if getattr(self, "_saved_attrib", None) is not None:
            live_cfg.raw["attribution"] = self._saved_attrib
        else:
            live_cfg.raw.pop("attribution", None)

    def test_passthrough_default_adds_nothing(self, client, components):
        """Default repo config has no attribution section -> passthrough:
        nothing is added when the client sends no attribution headers."""
        try:
            self._set_attribution(None, None)
            up = self._post_capture(client, components, send_client_headers=False)
            assert up.get("http-referer") != ATTRIB_HEADERS["HTTP-Referer"]
        finally:
            self._restore_attribution()

    def test_fallback_fills_only_missing(self, client, components):
        """fallback: missing from client -> added; present from client ->
        client value wins (per-header basis)."""
        try:
            self._set_attribution("fallback", {
                "HTTP-Referer": "https://configured.example",
                "X-Title": "Configured",
            })
            # Client sends NOTHING -> proxy fills
            built = routes_module._build_upstream_headers({})
            assert built["http-referer"] == "https://configured.example"
            assert built["x-title"] == "Configured"

            # Client sends its own -> client wins in fallback
            built2 = routes_module._build_upstream_headers({
                "http-referer": "https://client-sent.example",
                "x-title": "ClientApp",
            })
            assert built2["http-referer"] == "https://client-sent.example"
            assert built2["x-title"] == "ClientApp"

            # Client sends only ONE of the two -> that one kept, other filled
            built3 = routes_module._build_upstream_headers({"x-title": "ClientApp"})
            assert built3["x-title"] == "ClientApp"
            assert built3["http-referer"] == "https://configured.example"
        finally:
            self._restore_attribution()

    def test_force_overwrites_client(self, client, components):
        """force: configured headers always win over whatever the client sent."""
        try:
            self._set_attribution("force", {
                "HTTP-Referer": "https://configured.example",
                "X-Title": "Configured",
            })
            built = routes_module._build_upstream_headers({
                "http-referer": "https://client-sent.example",
                "x-title": "ClientApp",
            })
            assert built["http-referer"] == "https://configured.example"
            assert built["x-title"] == "Configured"
        finally:
            self._restore_attribution()


class TestRefreshEndpoint:
    """Test POST /refresh endpoint."""

    def test_refresh(self, client, components):
        """Test manual refresh trigger."""
        scheduler = MagicMock()
        from datetime import datetime as _dt

        scheduler.manual_refresh = AsyncMock(
            return_value={MODEL: "success"}
        )

        cache = components[2]
        _seed_cache(cache)

        with patch("main.get_scheduler", return_value=scheduler):
            resp = client.post("/refresh")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert MODEL in data["refreshed_models"]


class TestStatusEndpoint:
    """Test GET /status endpoint."""

    def test_status(self, client, components):
        """Test status endpoint (no scheduler attached)."""
        with patch("main.get_scheduler", return_value=None):
            resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "backoff" in data
