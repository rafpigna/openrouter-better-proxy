"""Tests for router core logic."""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from config import Config
from backoff import BackoffManager
from session import SessionManager
from cache import EndpointCache
from router import Router


@pytest.fixture
def mock_config():
    """Create a mock config with test data."""
    config = MagicMock()
    config.get_model_config.return_value = {
        "quantizations": ["fp8", "fp4"],
        "providers": ["deepseek", "deepinfra", "streamlake", "gmicloud"],
        "max_price": {
            "input": 0.10,
            "completion": 0.25,
            "cache": 0.05,
        },
    }
    return config


@pytest.fixture
def components(mock_config):
    """Create router components."""
    backoff = BackoffManager()
    sessions = SessionManager()
    cache = EndpointCache(data_dir="/tmp/test-cache")
    router = Router(backoff, sessions, cache)
    router.config = mock_config  # Inject mock config
    return router, backoff, sessions, cache


class TestRouterSelection:
    """Test provider selection logic."""

    def test_select_best_fp8_provider(self, components):
        """Test selection of best fp8 provider."""
        router, _, _, cache = components

        # Mock endpoints
        endpoints = [
            {"tag": "streamlake/fp8", "pricing": {"prompt": "0.0786", "completion": "0.15719", "input_cache_read": "0.01572"}},
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.080", "completion": "0.18", "input_cache_read": "0.016"}},
            {"tag": "gmicloud/fp8", "pricing": {"prompt": "0.112", "completion": "0.224", "input_cache_read": "0.0224"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        result = router.select_provider("deepseek/deepseek-v4-flash-0731")
        assert result is not None
        slug, tier = result
        assert tier == "fp8"
        # deepinfra wins (higher priority in config: ["deepseek", "deepinfra", "streamlake", "gmicloud"])
        assert slug == "deepinfra/fp8"

    def test_fallback_to_fp4_when_fp8_exhausted(self, components):
        """Test fallback to fp4 when fp8 providers are unavailable."""
        router, backoff, _, cache = components

        # Mock endpoints with only fp4
        endpoints = [
            {"tag": "open-inference/fp4", "pricing": {"prompt": "0.065", "completion": "0.14", "input_cache_read": "0.014"}},
            {"tag": "sail-research/fp4", "pricing": {"prompt": "0.065", "completion": "0.18", "input_cache_read": "0.020"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        # Mark fp8 providers as in cooldown (simulate all down)
        backoff.mark_error("streamlake/fp8")
        backoff.mark_error("deepinfra/fp8")

        result = router.select_provider("deepseek/deepseek-v4-flash-0731")
        assert result is not None
        slug, tier = result
        assert tier == "fp4"

    def test_max_price_filter(self, components):
        """Test that providers above max_price are filtered."""
        router, _, _, cache = components

        # Mock endpoints with one above max_price
        endpoints = [
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.080", "completion": "0.18", "input_cache_read": "0.016"}},
            {"tag": "expensive/provider", "pricing": {"prompt": "0.15", "completion": "0.30", "input_cache_read": "0.05"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        result = router.select_provider("deepseek/deepseek-v4-flash-0731")
        assert result is not None
        slug, _ = result
        assert slug == "deepinfra/fp8"  # expensive/provider should be filtered

    def test_session_stickiness(self, components):
        """Test that sessions stick to their assigned provider."""
        router, _, sessions, cache = components

        endpoints = [
            {"tag": "streamlake/fp8", "pricing": {"prompt": "0.0786", "completion": "0.15719", "input_cache_read": "0.01572"}},
            {"tag": "deepinfra/fp8", "pricing": {"prompt": "0.080", "completion": "0.18", "input_cache_read": "0.016"}},
        ]
        cache.set("deepseek/deepseek-v4-flash-0731", {"endpoints": endpoints, "fetched_at": datetime.utcnow().isoformat()})

        # First request assigns session to deepinfra (higher priority in config)
        result1 = router.select_provider("deepseek/deepseek-v4-flash-0731", session_id="session-123")
        assert result1[0] == "deepinfra/fp8"

        # Second request with same session should stick
        result2 = router.select_provider("deepseek/deepseek-v4-flash-0731", session_id="session-123")
        assert result2[0] == "deepinfra/fp8"

    def test_no_endpoints_returns_none(self, components):
        """Test that missing endpoints returns None."""
        router, _, _, _ = components

        result = router.select_provider("nonexistent/model")
        assert result is None

    def test_provider_order_priority(self, components):
        """Test that provider order is respected when prices are equal."""
        router, _, _, cache = components

        # Same price, different providers
        endpoints = [
            {"tag": "gmicloud/fp8", "pricing": {"prompt": "0.0786", "completion": "0.15719", "input_cache_read": "0.01572"}},
            {"tag": "streamlake/fp8", "pricing": {"prompt": "0.0786", "completion": "0.15719", "input_cache_read": "0.01572"}},
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
        # 3 consecutive errors
        bm.mark_error("test-provider")
        bm.mark_error("test-provider")
        bm.mark_error("test-provider")
        assert bm.is_cooldown("test-provider")
        # Should be in 1h cooldown now
        status = bm.get_status("test-provider")
        assert status["error_count"] == 3
        assert status["in_cooldown"]

    def test_cooldown_expiration(self):
        """Test cooldown expires after timeout."""
        bm = BackoffManager(initial_cooldown=1)  # 1 second
        bm.mark_error("test-provider")
        assert bm.is_cooldown("test-provider")
        # Wait for cooldown to expire
        import time
        time.sleep(1.1)
        assert not bm.is_cooldown("test-provider")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
