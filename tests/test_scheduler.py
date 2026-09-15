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


class TestMigrationAuthorizedTargets:
    """Fix 2026-09-15 (incident 2026-09-10): migration may target ONLY
    providers the request-time router would authorize right now — same
    allowlist `providers`, same max_price, same backoff health. The old
    scan ran over the WHOLE catalog and pinned sessions to unconfigured
    providers (deepinfra/fp4 for z-ai/glm-5.3-flash)."""

    MODEL = "z-ai/glm-5.3-flash"

    @pytest.fixture(autouse=True)
    def _preserve_config(self):
        """Snapshot/restore the global config mutations around each test so
        the other test modules (test_streaming etc.) still see routing_config.yaml."""
        from config import config
        saved_models = config.raw.get("models")
        saved_migration = config.raw.get("migration")
        yield
        if saved_models is None:
            config.raw.pop("models", None)
        else:
            config.raw["models"] = saved_models
        if saved_migration is None:
            config.raw.pop("migration", None)
        else:
            config.raw["migration"] = saved_migration

    # Per-token prices; z-ai/fp8 is at the DOUBLED price (incident values)
    PRICES = {
        "z-ai/fp8": (1.5e-7, 5e-7, 3e-8),
        "deepinfra/fp4": (7.5e-8, 2.5e-7, 1.5e-8),
        "novita/fp8": (1.32e-7, 4.4e-7, 2.64e-8),
        "gmicloud/fp8": (1.125e-7, 3.75e-7, 2.25e-8),
        "streamlake/fp8": (1.1235e-7, 3.745e-7, 2.247e-8),
    }

    @staticmethod
    def _ep(tag: str) -> dict:
        p, c, k = TestMigrationAuthorizedTargets.PRICES[tag]
        return {
            "tag": tag,
            "pricing": {
                "prompt": str(p),
                "completion": str(c),
                "input_cache_read": str(k),
            },
            "quantization": tag.split("/")[-1] if "/" in tag else "unknown",
        }

    def _make(self, providers, max_price, cooldown=(), migration=None):
        """Build scheduler + cache + sessions wired like main.py."""
        from config import config
        config.raw["models"] = {self.MODEL: {"providers": providers, "max_price": max_price}}
        config.raw["migration"] = migration or {
            "enabled": True, "hysteresis_mult": 4.0, "est_turns_per_session": 50,
            "r_cache_estimate": 300000, "out_per_turn_estimate": 40000,
        }
        backoff = BackoffManager()
        for t in cooldown:
            backoff.mark_error(t)
        sessions = SessionManager()
        cache = EndpointCache(data_dir="/tmp/test-cache-mig-auth")
        router = Router(backoff, sessions, cache)
        fetcher = EndpointFetcher(api_key="test-key", cache=cache)
        diff = PriceDiffDetector(snapshot_dir="/tmp/test-snapshots-mig-auth")
        scheduler = RefreshScheduler(fetcher, cache, diff, router, sessions, PriceMigration())
        return scheduler, cache, sessions

    @pytest.mark.asyncio
    async def test_unauthorized_never_migrates(self):
        """Incident replica: deepinfra/fp4 is the cheapest alternative but
        deepinfra is NOT in the allowlist -> session stays on z-ai/fp8,
        migration log stays empty (old code pinned it to deepinfra/fp4)."""
        scheduler, cache, sessions = self._make(
            providers=["z-ai/fp8", "novita/fp8", "gmicloud/fp8"],
            max_price={"input": 0.076, "completion": 0.26, "cache": 0.016},
        )
        cache.set(self.MODEL, {"endpoints": [self._ep(t) for t in self.PRICES]})
        sessions.set_provider("S1", "z-ai/fp8")

        events = [{"provider": "z-ai/fp8", "changes": []}]
        await scheduler._evaluate_migration(self.MODEL, events, cache.get(self.MODEL))

        assert sessions.get_provider("S1") == "z-ai/fp8"
        assert scheduler.get_migration_log() == []

    @pytest.mark.asyncio
    async def test_migrates_to_authorized_alternative(self):
        """deepinfra/fp4 configured AND under max_price -> session migrates to
        it: the incident would have been FIXED if deepinfra had been
        authorized (migration to an actually usable provider)."""
        scheduler, cache, sessions = self._make(
            providers=["z-ai/fp8", "deepinfra/fp4", "gmicloud/fp8"],
            max_price={"input": 0.076, "completion": 0.26, "cache": 0.016},
        )
        cache.set(self.MODEL, {"endpoints": [self._ep(t) for t in self.PRICES]})
        sessions.set_provider("S1", "z-ai/fp8")

        events = [{"provider": "z-ai/fp8", "changes": []}]
        await scheduler._evaluate_migration(self.MODEL, events, cache.get(self.MODEL))

        assert sessions.get_provider("S1") == "deepinfra/fp4"
        log = scheduler.get_migration_log()
        assert len(log) == 1 and log[0]["to_provider"] == "deepinfra/fp4"

    @pytest.mark.asyncio
    async def test_base_in_allowlist_but_over_max_price_not_chosen(self):
        """'novita' base IS in the allowlist but its price is over max_price:
        it must NOT be a migration target, and with no authorized under-cap
        alternative at all the session stays put (old code would have pinned
        it to novita/fp8 — useless, the router would reject it at request
        time)."""
        scheduler, cache, sessions = self._make(
            providers=["z-ai/fp8", "novita/fp8", "gmicloud/fp8"],
            max_price={"input": 0.076, "completion": 0.26, "cache": 0.016},
        )
        tags = ["z-ai/fp8", "novita/fp8", "gmicloud/fp8"]
        cache.set(self.MODEL, {"endpoints": [self._ep(t) for t in tags]})
        sessions.set_provider("S1", "z-ai/fp8")

        events = [{"provider": "z-ai/fp8", "changes": []}]
        await scheduler._evaluate_migration(self.MODEL, events, cache.get(self.MODEL))

        assert sessions.get_provider("S1") == "z-ai/fp8"
        assert scheduler.get_migration_log() == []

    @pytest.mark.asyncio
    async def test_cooldown_provider_never_chosen(self):
        """deepinfra/fp4 authorized + under max_price but in cooldown -> not a
        migration target; migration picks the next authorized alternative
        (streamlake/fp8) instead."""
        scheduler, cache, sessions = self._make(
            providers=["z-ai/fp8", "deepinfra/fp4", "streamlake/fp8"],
            max_price={"input": 0.16, "completion": 0.55, "cache": 0.04},
            cooldown=("deepinfra/fp4",),
        )
        tags = ["z-ai/fp8", "deepinfra/fp4", "streamlake/fp8"]
        cache.set(self.MODEL, {"endpoints": [self._ep(t) for t in tags]})
        sessions.set_provider("S1", "z-ai/fp8")

        events = [{"provider": "z-ai/fp8", "changes": []}]
        await scheduler._evaluate_migration(self.MODEL, events, cache.get(self.MODEL))

        assert sessions.get_provider("S1") == "streamlake/fp8"
        log = scheduler.get_migration_log()
        assert len(log) == 1 and log[0]["to_provider"] == "streamlake/fp8"


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

    def _with_times(self, monkeypatch, times, tz="UTC", interval_minutes=30):
        # NOTE: default_timezone="UTC" keeps these regression tests
        # machine-independent (they predate the multi-timezone feature and
        # were written under the old implicit-UTC semantics).
        from config import config
        monkeypatch.setitem(config.raw, "refresh",
                            {"interval_minutes": interval_minutes, "price_change_threshold": 10.0,
                             "default_timezone": tz,
                             "times": times})

    @pytest.mark.asyncio
    async def test_explicit_time_rolls_over_month(self, monkeypatch):
        s = self._scheduler()
        # interval huge so the fixed time is the earliest candidate
        self._with_times(monkeypatch, ["09:00"], interval_minutes=1440)
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
        self._with_times(monkeypatch, ["09:00"], interval_minutes=1440)
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

    def _with_times(self, monkeypatch, times, tz="local", interval_minutes=1440):
        # interval default huge: these tests target the FIXED-time logic,
        # not the interval candidate (which is always in the min() now)
        from config import config
        monkeypatch.setitem(config.raw, "refresh",
                            {"interval_minutes": interval_minutes, "price_change_threshold": 10.0,
                             "default_timezone": tz,
                             "times": times})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tz,entry,expected_utc", [
        # "12:01" with default UTC == old behaviour
        ("UTC", "12:01", datetime(2026, 8, 31, 12, 1, tzinfo=timezone.utc)),
        # "19:05UTC" explicit UTC regardless of default
        ("local", "19:05UTC", datetime(2026, 8, 31, 19, 5, tzinfo=timezone.utc)),
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
        # legacy "Z" and numeric offsets are REJECTED by the two-clock
        # contract (2026-08-31) — skipped with a warning, never fatal
        self._with_times(monkeypatch, ["25:00", "abc", 12, "12:01Z", "12:01+02:00", "09:00"], tz="UTC")
        self._freeze(monkeypatch, datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc))
        trig = await s._wait_for_next_trigger()
        # only the valid "09:00" survives (bare int 12 skipped: neither str nor dict)
        assert (trig.hour, trig.minute) == (9, 0), trig

    @pytest.mark.asyncio
    async def test_no_times_falls_back_to_interval(self, monkeypatch):
        s = self._scheduler()
        self._with_times(monkeypatch, [], tz="UTC", interval_minutes=30)
        self._freeze(monkeypatch, datetime(2026, 8, 31, 8, 0, tzinfo=timezone.utc))
        trig = await s._wait_for_next_trigger()
        assert trig == datetime(2026, 8, 31, 8, 30, tzinfo=timezone.utc), trig

    @pytest.mark.asyncio
    async def test_interval_fires_alongside_fixed_times(self, monkeypatch):
        """IN ADDITION semantics (2026-08-31): with fixed times present, the
        periodic interval is still the earliest candidate when it comes first."""
        s = self._scheduler()
        self._with_times(monkeypatch, ["23:59"], tz="UTC", interval_minutes=30)
        self._freeze(monkeypatch, datetime(2026, 8, 31, 10, 0, tzinfo=timezone.utc))
        trig = await s._wait_for_next_trigger()
        # interval candidate 10:30 beats the fixed 23:59 — it MUST fire
        assert trig == datetime(2026, 8, 31, 10, 30, tzinfo=timezone.utc), trig


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
