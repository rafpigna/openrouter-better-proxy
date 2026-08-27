"""Preset routing tests (strategy: presets).

Covers:
- helpers (_is_preset_mode, _preset_slug_for) against config.raw
- non-stream forward: model swap to preset slug, NO provider pin in body,
  per-attempt resolution keyed on provider BASE
- missing mapping -> candidate skipped, upstream never called, >=500 response
- multi-candidate: failover advances to the NEXT preset in order
- providers-mode regression: body carries the provider pin as before
- config schema validation (web_routes.ModelConfig)
"""
import copy
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from main import app
from routes import init_routes, _is_preset_mode, _preset_slug_for
from router import Router
from cache import EndpointCache
from backoff import BackoffManager
from session import SessionManager

MODEL = "deepseek/deepseek-v4-flash-0731"
PRESET_DI = MODEL + "@preset/deepseekv4flash-deepinfra"
PRESET_SL = MODEL + "@preset/deepseekv4flash-streamlake"


@pytest.fixture(autouse=True)
def setup_env():
    os.environ["OPENROUTER_API_KEY"] = "dummy-key-for-testing"
    yield
    os.environ.pop("OPENROUTER_API_KEY", None)


def _current_models():
    from config import config
    return copy.deepcopy(config.raw.get("models") or {})


@pytest.fixture
def components():
    backoff = BackoffManager()
    sessions = SessionManager()
    cache = EndpointCache(data_dir="data/test-cache-preset")
    router = Router(backoff, sessions, cache)
    init_routes(router, cache)
    return router, sessions, cache


@pytest.fixture
def client(components):
    return TestClient(app)


def _seed_cache(cache):
    """Two authorized providers with realistic per-token prices."""
    cache.set(MODEL, {
        "endpoints": [
            {"tag": "deepinfra/fp8", "quantization": "fp8", "status": 0,
             "pricing": {"prompt": "0.00000008", "completion": "0.00000018",
                         "input_cache_read": "0.000000016"}},
            {"tag": "streamlake/fp8", "quantization": "fp8", "status": 0,
             "pricing": {"prompt": "0.00000009", "completion": "0.00000019",
                         "input_cache_read": "0.000000017"}},
        ],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    })


def _with_models(monkeypatch, models_dict):
    """Temporarily replace config.raw['models'] (auto-restored by monkeypatch)."""
    from config import config
    monkeypatch.setitem(config.raw, "models", models_dict)


CFG_PRESET_BOTH = {
    MODEL: {
        "quantizations": ["fp8"],
        "providers": ["deepinfra", "streamlake"],
        "max_price": {"input": 0.10, "completion": 0.25, "cache": 0.05},
        "strategy": "presets",
        "presets": {"deepinfra": PRESET_DI, "streamlake": PRESET_SL},
    },
}


def _mock_client_with_post(handler):
    """MagicMock async client whose .post delegates to `handler(url, json=...)`."""
    mc = MagicMock()
    mc.__aenter__ = AsyncMock(return_value=mc)
    mc.__aexit__ = AsyncMock(return_value=False)

    async def post(url, json=None, headers=None):
        return await handler(url, json, headers)

    mc.post = AsyncMock(side_effect=post)
    return mc


def _ok_response(model_field):
    r = MagicMock()
    r.status_code = 200
    r.aread = AsyncMock(return_value=b"")
    r.json.return_value = {
        "id": "gen-1",
        "model": model_field,
        "provider": "DeepInfra",
        "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return r


def _err_response(code):
    r = MagicMock()
    r.status_code = code
    r.aread = AsyncMock(return_value=b'{"error":{"message":"boom"}}')
    return r


class TestHelpers:
    def test_is_preset_mode_on(self, monkeypatch):
        _with_models(monkeypatch, CFG_PRESET_BOTH)
        assert _is_preset_mode(MODEL) is True

    def test_is_preset_mode_default_off(self, monkeypatch):
        _with_models(monkeypatch, {MODEL: {"strategy": "providers"}})
        assert _is_preset_mode(MODEL) is False

    def test_is_preset_mode_absent_config(self):
        assert _is_preset_mode("unknown/model") is False

    def test_preset_slug_resolution_by_base(self, monkeypatch):
        _with_models(monkeypatch, CFG_PRESET_BOTH)
        assert _preset_slug_for(MODEL, "deepinfra/fp8") == PRESET_DI
        assert _preset_slug_for(MODEL, "streamlake/fp8") == PRESET_SL

    def test_preset_slug_missing_provider_returns_none(self, monkeypatch):
        _with_models(monkeypatch, CFG_PRESET_BOTH)
        assert _preset_slug_for(MODEL, "gmicloud/fp8") is None


class TestNonStreamPresetForward:
    def test_body_swaps_model_and_drops_pin(self, client, components, monkeypatch):
        router, sessions, cache = components
        _seed_cache(cache)
        _with_models(monkeypatch, CFG_PRESET_BOTH)

        captured = {}

        async def handler(url, json_body, headers):
            captured["body"] = json_body
            return _ok_response(PRESET_DI)

        mc = _mock_client_with_post(handler)
        with patch("routes.httpx.AsyncClient") as MockClient:
            MockClient.return_value = mc
            resp = client.post("/v1/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "session_id": "s-presets",
                "temperature": 0.7,
            })

        assert resp.status_code == 200
        b = captured["body"]
        assert b["model"] == PRESET_DI          # swapped ONLY the model field
        assert "provider" not in b              # no pin injected
        assert b["temperature"] == 0.7          # verbatim passthrough preserved
        assert b["session_id"] == "s-presets"

    def test_missing_mapping_skips_candidate(self, client, components, monkeypatch):
        router, sessions, cache = components
        _seed_cache(cache)
        cfg = copy.deepcopy(CFG_PRESET_BOTH)
        cfg[MODEL]["providers"] = ["deepinfra"]
        cfg[MODEL]["presets"] = {"gmicloud": MODEL + "@preset/x"}   # unusable mapping
        _with_models(monkeypatch, cfg)

        mc = _mock_client_with_post(lambda u, j, h: _ok_response(PRESET_DI))
        with patch("routes.httpx.AsyncClient") as MockClient:
            MockClient.return_value = mc
            resp = client.post("/v1/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
            })

        assert resp.status_code >= 500          # honest failure, silent 200 forbidden
        assert not mc.post.called               # nothing was ever attempted

    def test_failover_to_next_preset(self, client, components, monkeypatch):
        router, sessions, cache = components
        _seed_cache(cache)
        _with_models(monkeypatch, CFG_PRESET_BOTH)

        attempts = []

        async def handler(url, json_body, headers):
            attempts.append(json_body["model"])
            if json_body["model"] == PRESET_DI:
                return _err_response(503)       # first preset: always transient error
            return _ok_response(PRESET_SL)

        mc = _mock_client_with_post(handler)
        with patch("routes.httpx.AsyncClient") as MockClient:
            MockClient.return_value = mc
            resp = client.post("/v1/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
            })

        assert resp.status_code == 200
        assert attempts[-1] == PRESET_SL        # second candidate served the request
        assert PRESET_DI in attempts            # first candidate really attempted
        assert attempts.count(PRESET_DI) >= 1

    def test_providers_mode_regression_pin_intact(self, client, components, monkeypatch):
        router, sessions, cache = components
        _seed_cache(cache)
        # Full providers-mode config (empty dict is falsy -> "No config" 400).
        _with_models(monkeypatch, {
            MODEL: {
                "quantizations": ["fp8"],
                "providers": ["deepinfra"],
                "max_price": {"input": 0.10, "completion": 0.25, "cache": 0.05},
            },
        })

        captured = {}

        async def handler(url, json_body, headers):
            captured["body"] = json_body
            return _ok_response(MODEL)

        mc = _mock_client_with_post(handler)
        with patch("routes.httpx.AsyncClient") as MockClient:
            MockClient.return_value = mc
            resp = client.post("/v1/chat/completions", json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
            })

        assert resp.status_code == 200
        body = captured["body"]
        assert body["provider"] == {"only": ["deepinfra/fp8"]}
        assert body["model"] == MODEL           # untouched in providers mode


class TestModelConfigSchema:
    """Validation rules added to web_routes.ModelConfig."""

    def _validate(self, model_cfg):
        from web_routes import ConfigSchema
        data = {
            "server": {"host": "127.0.0.1", "port": 8787},
            "models": {MODEL: model_cfg},
        }
        return ConfigSchema(**data)

    def test_valid_preset_model(self):
        m = self._validate({
            "quantizations": ["fp8"], "providers": ["deepinfra"],
            "max_price": {"input": 0.10, "completion": 0.25, "cache": 0.05},
            "strategy": "presets", "presets": {"deepinfra": PRESET_DI},
        })
        assert m.models[MODEL].presets == {"deepinfra": PRESET_DI}

    def test_default_strategy_is_none(self):
        m = self._validate({
            "quantizations": ["fp8"], "providers": ["deepinfra"],
            "max_price": {"input": 0.10, "completion": 0.25, "cache": 0.05},
        })
        assert m.models[MODEL].strategy is None
        assert m.models[MODEL].presets is None

    def test_invalid_strategy_rejected(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            self._validate({
                "quantizations": ["fp8"], "providers": ["deepinfra"],
                "max_price": {"input": 0.10, "completion": 0.25, "cache": 0.05},
                "strategy": "magic",
            })

    def test_bad_preset_slug_rejected(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            self._validate({
                "quantizations": ["fp8"], "providers": ["deepinfra"],
                "max_price": {"input": 0.10, "completion": 0.25, "cache": 0.05},
                "strategy": "presets",
                "presets": {"deepinfra": "deepseek/deepseek-v4-flash-0731"},
            })

    def test_presets_key_must_be_base_name(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            self._validate({
                "quantizations": ["fp8"], "providers": ["deepinfra"],
                "max_price": {"input": 0.10, "completion": 0.25, "cache": 0.05},
                "strategy": "presets",
                "presets": {"deepinfra/fp8": PRESET_DI},   # slash forbidden
            })

    def test_strategy_presets_requires_mapping(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            self._validate({
                "quantizations": ["fp8"], "providers": ["deepinfra"],
                "max_price": {"input": 0.10, "completion": 0.25, "cache": 0.05},
                "strategy": "presets",
            })
