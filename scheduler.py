"""Asyncio scheduler for periodic endpoint refresh."""

import asyncio
import logging
from datetime import datetime, time, timedelta, timezone
from typing import Optional

from config import config
from fetcher import EndpointFetcher
from cache import EndpointCache
from price_diff import PriceDiffDetector
from session import SessionManager
from router import Router
from migration import PriceMigration

logger = logging.getLogger(__name__)


class RefreshScheduler:
    """Schedule periodic endpoint refreshes.

    Trigger types:
    1. Periodic: every `interval_minutes`
    2. Explicit times: list of times/windows from config
    3. Manual: triggered via /refresh endpoint
    """

    def __init__(
        self,
        fetcher: EndpointFetcher,
        cache: EndpointCache,
        diff_detector: PriceDiffDetector,
        router: Router,
        sessions: SessionManager,
        price_migration: PriceMigration,
    ):
        self.fetcher = fetcher
        self.cache = cache
        self.diff_detector = diff_detector
        self.router = router
        self.sessions = sessions
        self.price_migration = price_migration
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._migration_log: list[dict] = []  # Track migration events
        self._next_run: Optional[datetime] = None  # Next scheduled refresh

    @property
    def next_run(self) -> Optional[datetime]:
        """Next scheduled refresh datetime (UTC), or None if not running."""
        return self._next_run

    async def start(self):
        """Start the scheduler."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info("Refresh scheduler started")

    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Refresh scheduler stopped")

    async def _run(self):
        """Main scheduler loop."""
        logger.info("Scheduler loop started")

        while self._running:
            try:
                next_run = await self._wait_for_next_trigger()
                if next_run == "immediate":
                    self._next_run = None
                    await self._refresh_all()
                else:
                    # next_run is a datetime (timezone-aware UTC)
                    self._next_run = next_run
                    now = datetime.now(timezone.utc)
                    delay = (next_run - now).total_seconds()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    await self._refresh_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Backoff on error

    async def _wait_for_next_trigger(self):
        """Calculate next trigger time.

        Returns:
            'immediate' if should refresh now, or datetime for next trigger.
        """
        now = datetime.now(timezone.utc)
        current_time = now.time()

        # Check explicit times
        next_times = []
        for time_spec in config.refresh_times:
            if isinstance(time_spec, str):
                # Simple time like "10:00"
                h, m = map(int, time_spec.split(":"))
                trigger = now.replace(hour=h, minute=m, second=0, microsecond=0)
                if trigger <= now:
                    # timedelta handles month/year rollover (e.g. Jan 31 -> Feb 1)
                    trigger = trigger + timedelta(days=1)
                next_times.append(trigger)
            elif isinstance(time_spec, dict):
                # Window like {from: "18:00", to: "22:00", step: "05m"}
                # Safeguard: YAML may parse times without colon as int (e.g. 18 -> 18)
                from_str = str(time_spec.get("from", "00:00"))
                to_str = str(time_spec.get("to", "23:59"))
                step_str = str(time_spec.get("step", "05m"))

                # Skip if from/to aren't valid time strings
                if ":" not in from_str or ":" not in to_str:
                    logger.warning(f"Skipping invalid refresh time spec: {time_spec}")
                    continue

                from_h, from_m = map(int, from_str.split(":"))
                to_h, to_m = map(int, to_str.split(":"))

                # Calculate next trigger in window
                trigger = now.replace(hour=from_h, minute=from_m, second=0, microsecond=0)
                if trigger <= now:
                    # Add one day (timedelta handles month/year rollover)
                    trigger = trigger + timedelta(days=1)

                # Check if we're in a dense window (every 5 min)
                if step_str == "05m":
                    # Round up to next 5-minute mark (timedelta handles the
                    # minute>=56 -> next-hour and day rollover correctly)
                    minutes = trigger.minute
                    remaining = (5 - (minutes % 5)) % 5
                    if remaining:
                        trigger = trigger + timedelta(minutes=remaining)
                    if trigger <= now:
                        trigger = trigger + timedelta(days=1)

                next_times.append(trigger)

        if next_times:
            return min(next_times)

        # Fall back to periodic
        interval = config.refresh_interval_minutes * 60
        next_run = now.timestamp() + interval
        return datetime.fromtimestamp(next_run, timezone.utc)

    async def _refresh_all(self):
        """Refresh endpoints for all configured models."""
        logger.info("Starting endpoint refresh")

        results = await self.fetcher.refresh_all_models()

        # Detect price changes and evaluate migration
        for model_id, status in results.items():
            if status == "success":
                cache_data = self.cache.get(model_id)
                if cache_data:
                    events = self.diff_detector.detect_changes(cache_data)
                    if events:
                        logger.info(f"Price change events for {model_id}: {len(events)}")
                        await self._evaluate_migration(model_id, events, cache_data)

        logger.info(f"Endpoint refresh complete: {results}")

        # Periodic cleanup of stale sessions (every refresh cycle)
        removed = self.sessions.cleanup_stale(max_age_hours=24)

        return results

    async def _evaluate_migration(self, model_id: str, events: list[dict], cache_data: dict) -> None:
        """Evaluate migration for active sessions based on price change events."""
        if not config.migration_enabled:
            logger.debug("Migration disabled, skipping evaluation")
            return

        # Build endpoint lookup for quick access
        endpoints_by_tag = {ep["tag"]: ep for ep in cache_data.get("endpoints", [])}

        # Get all active sessions
        active_sessions = self.sessions.get_all_sessions()
        if not active_sessions:
            logger.debug("No active sessions to migrate")
            return

        for event in events:
            changed_provider = event.get("provider")
            if not changed_provider:
                continue

            # Check if this provider is sticky for any active session
            for session_id, sticky_provider in active_sessions.items():
                if sticky_provider != changed_provider:
                    continue

                logger.info(f"Session {session_id} sticky on {changed_provider} with price change")

                # Get current provider endpoint
                current_ep = endpoints_by_tag.get(changed_provider)
                if not current_ep:
                    logger.warning(f"Cannot find endpoint for {changed_provider}")
                    continue

                # Find best alternative (exclude current provider)
                best_alternative = None
                best_n_star = float("inf")

                for ep in cache_data.get("endpoints", []):
                    alt_tag = ep.get("tag")
                    if alt_tag == changed_provider:
                        continue

                    # Calculate N* for this alternative
                    n_star = self.price_migration.calculate_n_star(current_ep, ep)
                    if n_star < best_n_star:
                        best_n_star = n_star
                        best_alternative = ep

                if not best_alternative:
                    logger.debug(f"No alternative provider found for {changed_provider}")
                    continue

                # Decide whether to migrate
                should_migrate, n_star = self.price_migration.should_migrate(
                    current_ep,
                    best_alternative,
                    est_turns_remaining=config.est_turns_per_session,
                )

                if should_migrate:
                    logger.info(
                        f"Migrating session {session_id}: {changed_provider} -> {best_alternative.get('tag')} "
                        f"(N*={n_star:.2f})"
                    )
                    self.sessions.set_provider(session_id, best_alternative.get("tag"))

                    # Log migration event
                    self._migration_log.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "session_id": session_id,
                        "from_provider": changed_provider,
                        "to_provider": best_alternative.get("tag"),
                        "n_star": n_star,
                        "model": model_id,
                    })

                    # Keep only last 100 migration events
                    if len(self._migration_log) > 100:
                        self._migration_log = self._migration_log[-100:]
                else:
                    logger.debug(
                        f"No migration for session {session_id}: N*={n_star:.2f}, "
                        f"gain insufficient vs hysteresis"
                    )

    async def manual_refresh(self) -> dict:
        """Manual refresh trigger (called from /refresh endpoint)."""
        logger.info("Manual refresh triggered")
        return await self._refresh_all()

    def get_migration_log(self) -> list[dict]:
        """Get recent migration events (for /status endpoint)."""
        return list(self._migration_log)
