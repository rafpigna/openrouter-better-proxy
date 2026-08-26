"""Tests for router core logic.

Covers the provider selection rules:
- allowlist gate (never pick a provider outside the configured list)
- tier fallback ONLY among authorized providers
- max_price filter
- session stickiness (sticky-first, rebound to the winning provider)
- select_candidates() -> ordered list for in-process failover
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from config import Config, config
from backoff import BackoffManager
from session import SessionManager
from cache import EndpointCache
from router import Router


@pytest.fixture
def mock_config():
    """Create a mock config with test data."""
    cfg = MagicMock()
    cfg.get_model_config.return_value = {
        "quantizations": ["fp8", "fp4"],
        "providers": ["deepseek", "deepinfra", "streamlake", "gmicloud"],
        "max_price": {
            "input": 0.10,
            "completion": 0.25,
            "cache": 0.05,
        },
    }
    return cfg


@pytest.fixture
def components(mock_config, monkeypatch):
    """Create router components with a deterministic config.

    The router reads the module-level `config` singleton, so we monkeypatch
    its get_model_config to the mock (avoiding dependency on the real
    routing_config.yaml on disk).
    """
    backoff = BackoffManager()
    sessions = SessionManager()
    cache = EndpointCache(data_dir="/tmp/test-cache")
    router = Router(backoff, sessions, cache)
    monkeypatch.setattr(config, "get_model_config", mock_config.get_model_config)
    return router, backoff, sessions, cache


# Per-token prices used below (=$/M ÷ 1e6):
#   deepinfra   0.08 / 0.18 / 0.016
#   streamlake  0.0786 / 0.15719 / 0.01572
#   gmicloud    0.112 / 0.224 / 0.0224
#   open-inference 0.065 / 0.14 / 0.014  (NOT in allowlist)


class TestRouterSelection:
    """Test provider selection logic."""

    def test_select_best_fp8_provider(self, components):
        """Test selection of best fp8 provider."""
        router, _, _, cache = components

        endpoints = [
            {"tag": "streamlake/fp8", "pricing": {"prompt": "0.0000000786", "completion": "0.00000015719", "input_cache_read": "0.00000001572"}},
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.00000008", "completion": "0.00000018", "input_cache_read": "0.000000016"}},
            {"tag": "gmicloud/fp8", "pricing": {"prompt": "0.000000112", "completion": "0.000000224", "input_cache_read": "0.0000000224"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        result = router.select_provider("deepseek/deepseek-v4-flash-0731")
        assert result is not None
        slug, tier = result
        assert tier == "fp8"
        # deepinfra wins (higher priority in config: ["deepseek", "deepinfra", "streamlake", "gmicloud"])
        assert slug == "deepinfra/fp8"

    def test_allowlist_gate_excludes_non_authorized(self, components):
        """A non-authorized, cheaper provider is NEVER a candidate."""
        router, _, _, cache = components

        endpoints = [
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.00000008", "completion": "0.00000018", "input_cache_read": "0.000000016"}},
            # cheaper but NOT in providers list
            {"tag": "open-inference/fp4", "pricing": {"prompt": "0.000000065", "completion": "0.00000014", "input_cache_read": "0.000000014"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        candidates = router.select_candidates("deepseek/deepseek-v4-flash-0731")
        assert [c[0] for c in candidates] == ["deepinfra/fp8"]
        assert router.select_provider("deepseek/deepseek-v4-flash-0731")[0] == "deepinfra/fp8"

    def test_never_fallback_to_non_authorized_provider(self, components):
        """When the only available providers are non-authorized -> None (no fallback)."""
        router, backoff, _, cache = components

        # Only NON-authorized fp4 providers available
        endpoints = [
            {"tag": "open-inference/fp4", "pricing": {"prompt": "0.000000065", "completion": "0.00000014", "input_cache_read": "0.000000014"}},
            {"tag": "sail-research/fp4", "pricing": {"prompt": "0.000000065", "completion": "0.00000018", "input_cache_read": "0.000000020"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        # Authorized fp8 providers down (simulate all unavailable)
        backoff.mark_error("streamlake/fp8")
        backoff.mark_error("deepinfra/fp8")

        assert router.select_candidates("deepseek/deepseek-v4-flash-0731") == []
        assert router.select_provider("deepseek/deepseek-v4-flash-0731") is None

    def test_fallback_to_lower_tier_when_authorized(self, components, monkeypatch):
        """Tier fallback (fp8 -> fp4) happens ONLY among authorized providers."""
        router, backoff, _, cache = components

        # Authorize a provider that has an fp4 variant, plus fp8 deepinfra
        cfg = MagicMock()
        cfg.get_model_config.return_value = {
            "quantizations": ["fp8", "fp4"],
            "providers": ["deepseek", "deepinfra", "open-inference"],
            "max_price": {"input": 0.10, "completion": 0.25, "cache": 0.05},
        }
        monkeypatch.setattr(config, "get_model_config", cfg.get_model_config)

        endpoints = [
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.00000008", "completion": "0.00000018", "input_cache_read": "0.000000016"}},
            {"tag": "open-inference/fp4", "pricing": {"prompt": "0.000000065", "completion": "0.00000014", "input_cache_read": "0.000000014"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        # fp8 authorized provider down -> fall back to authorized fp4
        backoff.mark_error("deepinfra/fp8")

        result = router.select_provider("deepseek/deepseek-v4-flash-0731")
        assert result is not None
        slug, tier = result
        assert tier == "fp4"
        assert slug == "open-inference/fp4"

    def test_select_candidates_ordered_list(self, components):
        """select_candidates returns the ordered list used for failover."""
        router, _, _, cache = components

        endpoints = [
            {"tag": "streamlake/fp8", "pricing": {"prompt": "0.0000000786", "completion": "0.00000015719", "input_cache_read": "0.00000001572"}},
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.00000008", "completion": "0.00000018", "input_cache_read": "0.000000016"}},
            # gmicloud input 0.112 $/M > max_price input 0.10 -> filtered by price
            {"tag": "gmicloud/fp8", "pricing": {"prompt": "0.000000112", "completion": "0.000000224", "input_cache_read": "0.0000000224"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        candidates = router.select_candidates("deepseek/deepseek-v4-flash-0731")
        # order by provider priority: deepinfra (idx1) then streamlake (idx2); gmicloud excluded by price
        assert candidates == [("deepinfra/fp8", "fp8"), ("streamlake/fp8", "fp8")]

    def test_max_price_filter(self, components):
        """Test that providers above max_price are filtered."""
        router, _, _, cache = components

        endpoints = [
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.00000008", "completion": "0.00000018", "input_cache_read": "0.000000016"}},
            {"tag": "expensive/provider", "pricing": {"prompt": "0.00000015", "completion": "0.00000030", "input_cache_read": "0.00000005"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        result = router.select_provider("deepseek/deepseek-v4-flash-0731")
        assert result is not None
        slug, _ = result
        assert slug == "deepinfra/fp8"  # expensive/provider should be filtered

    def test_session_stickiness(self, components):
        """A session is sticky to the provider that served it (sticky-first)."""
        router, _, sessions, cache = components

        endpoints = [
            {"tag": "streamlake/fp8", "pricing": {"prompt": "0.0000000786", "completion": "0.00000015719", "input_cache_read": "0.00000001572"}},
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.00000008", "completion": "0.00000018", "input_cache_read": "0.000000016"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        # No session: best = deepinfra (higher priority)
        assert router.select_provider("deepseek/deepseek-v4-flash-0731")[0] == "deepinfra/fp8"

        # Simulate streamlake having actually served this session (rebind after success)
        router.bind_session("session-123", "streamlake/fp8")

        # Now streamlake is placed first (stickiness preserved)
        candidates = router.select_candidates("deepseek/deepseek-v4-flash-0731", session_id="session-123")
        assert candidates[0] == ("streamlake/fp8", "fp8")

    def test_no_endpoints_returns_none(self, components):
        """Test that missing endpoints returns None/empty."""
        router, _, _, _ = components
        assert router.select_provider("nonexistent/model") is None
        assert router.select_candidates("nonexistent/model") == []

    def test_provider_order_priority(self, components):
        """Test that provider order is respected when prices are equal."""
        router, _, _, cache = components

        endpoints = [
            {"tag": "gmicloud/fp8", "pricing": {"prompt": "0.0000000786", "completion": "0.00000015719", "input_cache_read": "0.00000001572"}},
            {"tag": "streamlake/fp8", "pricing": {"prompt": "0.0000000786", "completion": "0.00000015719", "input_cache_read": "0.00000001572"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        result = router.select_provider("deepseek/deepseek-v4-flash-0731")
        assert result is not None
        slug, _ = result
        # streamlake should win (higher priority in config)
        assert slug == "streamlake/fp8"


class TestBackoffManager:
    """Test backoff state management."""

    def test_initial_cooldown(self):
        """Test first error triggers initial cooldown."""
        bm = BackoffManager(initial_cooldown=300)
        bm.mark_error("test-provider")
        assert bm.is_cooldown("test-provider")

    def test_success_resets_cooldown(self):
        """Test success resets cooldown."""
        bm = BackoffManager(initial_cooldown=300)
        bm.mark_error("test-provider")
        bm.mark_success("test-provider")
        assert not bm.is_cooldown("test-provider")

    def test_escalation(self):
        """Test cooldown escalation after consecutive errors."""
        bm = BackoffManager(
            initial_cooldown=300,
            consecutive_threshold=3,
            escalation_seconds=[3600],
            max_cooldown=43200,
        )
        bm.mark_error("test-provider")
        bm.mark_error("test-provider")
        bm.mark_error("test-provider")
        assert bm.is_cooldown("test-provider")
        status = bm.get_status("test-provider")
        assert status["error_count"] == 3
        assert status["in_cooldown"]

    def test_cooldown_expiration(self):
        """Test cooldown expires after timeout."""
        bm = BackoffManager(initial_cooldown=1)  # 1 second
        bm.mark_error("test-provider")
        assert bm.is_cooldown("test-provider")
        import time
        time.sleep(1.1)
        assert not bm.is_cooldown("test-provider")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
