"""Router core — provider selection logic.

Implements the selection algorithm from DESIGN.md §3:
1. Get endpoint list from cache (or fetch if missing)
2. Filter by quantization tier (tiered)
3. Filter by provider order
4. Filter by max_price
5. Filter by backoff health
6. Order by provider priority, then price
7. Apply session stickiness
"""

import logging
from typing import Optional

from config import config
from backoff import BackoffManager
from session import SessionManager
from cache import EndpointCache

logger = logging.getLogger(__name__)


class Router:
    """Main router: selects best provider for a model + session."""

    def __init__(
        self,
        backoff: BackoffManager,
        sessions: SessionManager,
        endpoint_cache: EndpointCache,
    ):
        self.backoff = backoff
        self.sessions = sessions
        self.endpoint_cache = endpoint_cache

    def select_provider(
        self,
        model_id: str,
        session_id: Optional[str] = None,
        tools: bool = False,
    ) -> Optional[tuple[str, str]]:
        """Select best provider for a model.

        Returns:
            (provider_slug, quantization_tier) or None if no candidates.
        """
        # 1. Check session stickiness first
        if session_id and self.sessions.has_session(session_id):
            sticky_provider = self.sessions.get_provider(session_id)
            if sticky_provider and not self.backoff.is_cooldown(sticky_provider):
                logger.debug(f"Session {session_id} sticky to {sticky_provider}")
                return (sticky_provider, self._get_tier_for_provider(model_id, sticky_provider))

        # 2. Get endpoint list
        endpoints = self._get_endpoints(model_id)
        if not endpoints:
            logger.warning(f"No endpoints cached for model {model_id}")
            return None

        # 3. Get model config
        model_config = config.get_model_config(model_id)
        if not model_config:
            logger.warning(f"No config for model {model_id}")
            return None

        # 4. Get quantization tiers (tiered)
        quantizations = model_config.get("quantizations", ["fp8", "fp4"])

        # 5. Get provider order
        provider_order = model_config.get("providers", [])

        # 6. Get max_price filters
        max_price = model_config.get("max_price", {})
        max_input = max_price.get("input", float("inf"))
        max_completion = max_price.get("completion", float("inf"))
        max_cache = max_price.get("cache", float("inf"))

        # 7. Filter and score candidates
        candidates = []
        for ep in endpoints:
            tag = ep.get("tag")
            if not tag:
                continue

            pricing = ep.get("pricing") or {}
            prompt_price = float(pricing.get("prompt", 0) or 0)
            completion_price = float(pricing.get("completion", 0) or 0)
            cache_price = float(pricing.get("input_cache_read", 0) or 0)

            # Filter by max_price
            if prompt_price > max_input:
                continue
            if completion_price > max_completion:
                continue
            if cache_price > max_cache:
                continue

            # Filter by backoff health
            if self.backoff.is_cooldown(tag):
                continue

            # Determine tier
            tier = self._detect_tier(tag, quantizations)

            # Only consider if tier matches our priority list
            if tier not in quantizations:
                continue

            # Get provider priority index
            try:
                provider_idx = provider_order.index(tag.split("/")[0] if "/" in tag else tag)
            except ValueError:
                provider_idx = len(provider_order)

            candidates.append({
                "slug": tag,
                "tier": tier,
                "prompt_price": prompt_price,
                "completion_price": completion_price,
                "cache_price": cache_price,
                "provider_idx": provider_idx,
            })

        if not candidates:
            logger.debug(f"No valid candidates for model {model_id}")
            return None

        # 8. Sort: by tier priority, then provider order, then price
        tier_priority = {tier: i for i, tier in enumerate(quantizations)}
        candidates.sort(key=lambda c: (
            tier_priority.get(c["tier"], 999),
            c["provider_idx"],
            c["prompt_price"],
        ))

        # 9. Pick best
        best = candidates[0]
        slug = best["slug"]

        # 10. Store session mapping if applicable
        if session_id:
            self.sessions.set_provider(session_id, slug)

        logger.debug(f"Selected {slug} (tier={best['tier']}) for model {model_id}")
        return (slug, best["tier"])

    def _get_endpoints(self, model_id: str) -> list[dict]:
        """Get endpoint list from cache or fetch."""
        cached = self.endpoint_cache.get(model_id)
        if cached:
            return cached.get("endpoints", [])

        # Try loading from disk
        disk_data = self.endpoint_cache.load_from_disk(model_id)
        if disk_data:
            return disk_data.get("endpoints", [])

        return []

    def _detect_tier(self, tag: str, quantizations: list[str]) -> str:
        """Detect quantization tier from tag."""
        # Tags like "streamlake/fp8", "deepinfra/fp4", "open-inference/fp4"
        for q in quantizations:
            if q in tag.lower():
                return q
        return "unknown"

    def _get_tier_for_provider(self, model_id: str, provider: str) -> str:
        """Get the tier for a specific provider (for sticky sessions)."""
        endpoints = self._get_endpoints(model_id)
        for ep in endpoints:
            if ep.get("tag") == provider:
                return self._detect_tier(provider, config.get_model_config(model_id).get("quantizations", ["fp8", "fp4"]))
        return "unknown"

    def record_error(self, provider: str) -> None:
        """Record an error for a provider."""
        self.backoff.mark_error(provider)
        logger.warning(f"Error recorded for provider {provider}")

    def record_success(self, provider: str) -> None:
        """Record a success for a provider."""
        self.backoff.mark_success(provider)
        logger.debug(f"Success recorded for provider {provider}")

    def get_status(self) -> dict:
        """Get router status (for debugging)."""
        return {
            "sessions": self.sessions.get_all_sessions(),
            "backoff": {
                p: self.backoff.get_status(p)
                for p in set(list(self.backoff._error_counts.keys()) +
                           list(self.backoff._cooldown_until.keys()))
            },
            "cached_models": list(self.endpoint_cache.get_all_cached().keys()),
        }
