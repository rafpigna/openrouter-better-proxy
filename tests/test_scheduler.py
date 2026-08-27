"""Tests for scheduler, migration, and price diff."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, time

from scheduler import RefreshScheduler
from migration import PriceMigration
from price_diff import PriceDiffDetector
from fetcher import EndpointFetcher
from cache import EndpointCache
from backoff import BackoffManager
from session import SessionManager
from router import Router


@pytest.fixture
def components():
    """Create components for testing."""
    backoff = BackoffManager()
    sessions = SessionManager()
    cache = EndpointCache(data_dir="/tmp/test-cache-scheduler")
    router = Router(backoff, sessions, cache)
    fetcher = EndpointFetcher(api_key="test-key", cache=cache)
    diff_detector = PriceDiffDetector(snapshot_dir="/tmp/test-snapshots")
    scheduler = RefreshScheduler(
        fetcher, cache, diff_detector, router, sessions, PriceMigration()
    )
    return scheduler, fetcher, cache, diff_detector


class TestPriceMigration:
    """Test price migration logic."""

    def test_n_star_calculation(self):
        """Test N* formula calculation."""
        migration = PriceMigration()

        provider_a = {
            "pricing": {
                "prompt": "0.22",
                "completion": "0.66",
                "input_cache_read": "0.007",
            }
        }
        provider_b = {
            "pricing": {
                "prompt": "0.0786",
                "completion": "0.15719",
                "input_cache_read": "0.01572",
            }
        }

        n_star = migration.calculate_n_star(provider_a, provider_b)
        assert n_star > 0
        # N* > 1 means it takes more than 1 turn to break even
        # With hysteresis_mult=3, need turns > 3 * N* to migrate
        assert n_star < 2  # Reasonable threshold for "close to immediate"

    def test_should_migrate_on_expensive_provider(self):
        """Test migration decision when current provider is expensive."""
        migration = PriceMigration(est_turns_per_session=50)

        provider_a = {
            "pricing": {
                "prompt": "0.22",
                "completion": "0.66",
                "input_cache_read": "0.007",
            }
        }
        provider_b = {
            "pricing": {
                "prompt": "0.0786",
                "completion": "0.15719",
                "input_cache_read": "0.01572",
            }
        }

        should_migrate, n_star = migration.should_migrate(provider_a, provider_b)
        # With est_turns=50 and hysteresis_mult=3, should migrate
        assert should_migrate is True
        assert n_star > 0

    def test_no_migration_when_same_price(self):
        """Test no migration when prices are equal."""
        migration = PriceMigration()

        provider_a = {
            "pricing": {
                "prompt": "0.08",
                "completion": "0.18",
                "input_cache_read": "0.016",
            }
        }
        provider_b = {
            "pricing": {
                "prompt": "0.08",
                "completion": "0.18",
                "input_cache_read": "0.016",
            }
        }

        should_migrate, n_star = migration.should_migrate(provider_a, provider_b)
        assert should_migrate is False


class TestPriceDiffDetector:
    """Test price change detection."""

    def test_detects_price_change(self):
        """Test detection of price changes."""
        detector = PriceDiffDetector(snapshot_dir="/tmp/test-diff")

        last_snapshot = {
            "model_id": "test/model",
            "endpoints": [
                {
                    "tag": "provider/fp8",
                    "pricing": {
                        "prompt": "0.08",
                        "completion": "0.18",
                        "input_cache_read": "0.016",
                    },
                }
            ],
        }

        new_snapshot = {
            "model_id": "test/model",
            "endpoints": [
                {
                    "tag": "provider/fp8",
                    "pricing": {
                        "prompt": "0.16",  # Doubled
                        "completion": "0.18",
                        "input_cache_read": "0.016",
                    },
                }
            ],
        }

        detector._last_snapshot["test/model"] = last_snapshot
        events = detector.detect_changes(new_snapshot)

        assert len(events) == 1
        assert events[0]["provider"] == "provider/fp8"
        assert len(events[0]["changes"]) == 1
        assert events[0]["changes"][0]["field"] == "prompt"

    def test_no_change_when_prices_same(self):
        """Test no events when prices haven't changed."""
        detector = PriceDiffDetector(snapshot_dir="/tmp/test-diff2")

        snapshot = {
            "model_id": "test/model",
            "endpoints": [
                {
                    "tag": "provider/fp8",
                    "pricing": {
                        "prompt": "0.08",
                        "completion": "0.18",
                        "input_cache_read": "0.016",
                    },
                }
            ],
        }

        detector._last_snapshot["test/model"] = snapshot
        events = detector.detect_changes(snapshot)

        assert len(events) == 0


class TestScheduler:
    """Test scheduler logic."""

    @pytest.mark.asyncio
    async def test_manual_refresh(self, components):
        """Test manual refresh trigger."""
        scheduler, fetcher, cache, diff_detector = components

        # Mock fetcher
        fetcher.refresh_all_models = AsyncMock(return_value={"test/model": "success"})

        result = await scheduler.manual_refresh()
        assert result == {"test/model": "success"}

    @pytest.mark.asyncio
    async def test_scheduler_starts_and_stops(self, components):
        """Test scheduler lifecycle."""
        scheduler, _, _, _ = components

        await scheduler.start()
        assert scheduler._running is True

        await scheduler.stop()
        assert scheduler._running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
