"""Tests for scheduler, migration, and price diff."""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, time, timezone

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


class TestNextTriggerCalendarRollover:
    """Regression (2026-08-31): explicit-time triggers used
    `replace(day=day+1)` / `replace(minute=...)`, which crash with
    'day is out of range for month' on the 31st (and minute>=56 in dense
    windows). All rollovers must go through timedelta."""

    def _scheduler(self):
        backoff = BackoffManager()
        sessions = SessionManager()
        cache = EndpointCache(data_dir="data/test-cache-sched-cal")
        router = Router(backoff, sessions, cache)
        fetcher = EndpointFetcher(api_key="test-key", cache=cache)
        diff_detector = PriceDiffDetector(snapshot_dir="data/test-snapshots-cal")
        return RefreshScheduler(fetcher, cache, diff_detector, router, sessions, PriceMigration())

    def _with_times(self, monkeypatch, times, tz="UTC"):
        # NOTE: default_timezone="UTC" keeps these regression tests
        # machine-independent (they predate the multi-timezone feature and
        # were written under the old implicit-UTC semantics).
        from config import config
        monkeypatch.setitem(config.raw, "refresh",
                            {"interval_minutes": 30, "price_change_threshold": 10.0,
                             "default_timezone": tz,
                             "times": times})

    @pytest.mark.asyncio
    async def test_explicit_time_rolls_over_month(self, monkeypatch):
        s = self._scheduler()
        self._with_times(monkeypatch, ["09:00"])
        # Aug 31 10:00 UTC -> next trigger must be Sep 1 09:00 (not day=32!)
        now = datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc)
        real_now = datetime.now(timezone.utc)

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return now if tz else now.replace(tzinfo=None)
        import scheduler as sched_mod
        monkeypatch.setattr(sched_mod, "datetime", _Frozen)
        trig = await s._wait_for_next_trigger()
        assert (trig.year, trig.month, trig.day, trig.hour) == (2026, 9, 1, 9), trig
        del real_now

    @pytest.mark.asyncio
    async def test_dense_window_minute_rollover(self, monkeypatch):
        s = self._scheduler()
        self._with_times(monkeypatch, [{"from": "00:00", "to": "23:59", "step": "05m"}])
        # 23:57 -> rounding up must land 00:00 of the NEXT day, not minute=60
        now = datetime(2026, 8, 31, 23, 57, tzinfo=timezone.utc)

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return now if tz else now.replace(tzinfo=None)
        import scheduler as sched_mod
        monkeypatch.setattr(sched_mod, "datetime", _Frozen)
        trig = await s._wait_for_next_trigger()
        assert (trig.month, trig.day, trig.hour, trig.minute) == (9, 1, 0, 0), trig

    @pytest.mark.asyncio
    async def test_explicit_time_future_today(self, monkeypatch):
        s = self._scheduler()
        self._with_times(monkeypatch, ["09:00"])
        now = datetime(2026, 8, 31, 7, 0, tzinfo=timezone.utc)

        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return now if tz else now.replace(tzinfo=None)
        import scheduler as sched_mod
        monkeypatch.setattr(sched_mod, "datetime", _Frozen)
        trig = await s._wait_for_next_trigger()
        assert (trig.month, trig.day, trig.hour) == (8, 31, 9), trig


class TestTimezoneAwareTimes:
    """Multi-timezone refresh times (2026-08-31): entries accept an explicit
    suffix (Z = UTC, +HH:MM = fixed offset) and refresh.default_timezone
    resolves the plain ones. Machine-local semantics = backwards compatible
    default; UTC semantics = what the pre-feature code did."""

    def _scheduler(self):
        backoff = BackoffManager()
        sessions = SessionManager()
        cache = EndpointCache(data_dir="data/test-cache-sched-cal")
        router = Router(backoff, sessions, cache)
        fetcher = EndpointFetcher(api_key="test-key", cache=cache)
        diff_detector = PriceDiffDetector(snapshot_dir="data/test-snapshots-cal")
        return RefreshScheduler(fetcher, cache, diff_detector, router, sessions, PriceMigration())

    def _freeze(self, monkeypatch, now):
        class _Frozen(datetime):
            @classmethod
            def now(cls, tz=None):
                return now if tz else now.replace(tzinfo=None)
        import scheduler as sched_mod
        monkeypatch.setattr(sched_mod, "datetime", _Frozen)

    def _with_times(self, monkeypatch, times, tz="local"):
        from config import config
        monkeypatch.setitem(config.raw, "refresh",
                            {"interval_minutes": 30, "price_change_threshold": 10.0,
                             "default_timezone": tz,
                             "times": times})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tz,entry,expected_utc", [
        # "12:01" with default UTC == old behaviour
        ("UTC", "12:01", datetime(2026, 8, 31, 12, 1, tzinfo=timezone.utc)),
        # "12:01Z" legacy alias accepted
        ("local", "12:01Z", datetime(2026, 8, 31, 12, 1, tzinfo=timezone.utc)),
        # "19:05UTC" canonical explicit suffix
        ("local", "19:05UTC", datetime(2026, 8, 31, 19, 5, tzinfo=timezone.utc)),
        # "12:01+02:00" explicit offset == 10:01 UTC
        ("local", "12:01+02:00", datetime(2026, 8, 31, 10, 1, tzinfo=timezone.utc)),
        # "12:01-05:30" negative offset == 17:31 UTC
        ("UTC", "12:01-05:30", datetime(2026, 8, 31, 17, 31, tzinfo=timezone.utc)),
    ])
    async def test_entry_timezones(self, monkeypatch, tz, entry, expected_utc):
        s = self._scheduler()
        self._with_times(monkeypatch, [entry], tz=tz)
        self._freeze(monkeypatch, datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc))
        trig = await s._wait_for_next_trigger()
        assert trig == expected_utc, f"entry={entry!r} default={tz!r} -> {trig}"

    @pytest.mark.asyncio
    async def test_invalid_entries_skipped_not_fatal(self, monkeypatch):
        s = self._scheduler()
        self._with_times(monkeypatch, ["25:00", "abc", 12, "09:00"], tz="UTC")
        self._freeze(monkeypatch, datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc))
        trig = await s._wait_for_next_trigger()
        # only the valid "09:00" survives (note: bare int 12 is skipped as neither str nor dict)
        assert (trig.hour, trig.minute) == (9, 0), trig

    @pytest.mark.asyncio
    async def test_no_times_falls_back_to_interval(self, monkeypatch):
        s = self._scheduler()
        self._with_times(monkeypatch, [], tz="UTC")
        self._freeze(monkeypatch, datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc))
        trig = await s._wait_for_next_trigger()
        assert trig == datetime(2026, 8, 31, 8, 30, tzinfo=timezone.utc), trig


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
