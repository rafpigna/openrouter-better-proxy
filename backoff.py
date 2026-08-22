"""Backoff state manager for providers."""

import time
from collections import defaultdict
from typing import Optional


class BackoffManager:
    """Track error counts and cooldowns per provider.

    States:
    - 1 error: initial_cooldown (default 5 min)
    - 3 consecutive errors: 1 hour
    - encore error after 1h: 12 hours
    - max: never more than 12 hours
    - success: reset
    """

    def __init__(
        self,
        initial_cooldown: int = 300,
        consecutive_threshold: int = 3,
        escalation_seconds: list[int] = None,
        max_cooldown: int = 43200,
    ):
        self.initial_cooldown = initial_cooldown
        self.consecutive_threshold = consecutive_threshold
        self.escalation_seconds = escalation_seconds or [3600, 43200]
        self.max_cooldown = max_cooldown

        # Per-provider state
        self._error_counts: dict[str, int] = defaultdict(int)
        self._cooldown_until: dict[str, float] = {}

    def mark_error(self, provider: str) -> None:
        """Record an error for a provider."""
        self._error_counts[provider] += 1
        count = self._error_counts[provider]

        if count == 1:
            # First error: initial cooldown
            self._cooldown_until[provider] = time.time() + self.initial_cooldown
        elif count >= self.consecutive_threshold:
            # Escalate
            idx = min(count - self.consecutive_threshold, len(self.escalation_seconds) - 1)
            cooldown = min(self.escalation_seconds[idx], self.max_cooldown)
            self._cooldown_until[provider] = time.time() + cooldown

    def mark_success(self, provider: str) -> None:
        """Reset backoff for a provider."""
        self._error_counts[provider] = 0
        self._cooldown_until.pop(provider, None)

    def is_cooldown(self, provider: str) -> bool:
        """Check if provider is currently in cooldown."""
        if provider not in self._cooldown_until:
            return False
        if time.time() < self._cooldown_until[provider]:
            return True
        # Cooldown expired, reset
        self._cooldown_until.pop(provider)
        return False

    def get_status(self, provider: str) -> dict:
        """Get backoff status for a provider."""
        return {
            "provider": provider,
            "error_count": self._error_counts.get(provider, 0),
            "in_cooldown": self.is_cooldown(provider),
            "cooldown_remaining": max(0, self._cooldown_until.get(provider, 0) - time.time())
            if provider in self._cooldown_until
            else 0,
        }
