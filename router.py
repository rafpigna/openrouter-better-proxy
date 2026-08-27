"""Router core — provider selection logic.

Implements the selection algorithm from DESIGN.md §3:
1. Get endpoint list from cache (or fetch if missing)
2. Filter by quantization tier (tiered)
3. Filter by provider order — STRICT ALLOWLIST GATE
4. Filter by max_price
5. Filter by backoff health
6. Order by provider priority, then price
7. Apply session stickiness
8. Expose the ordered candidate LIST so routes.py can fail over
   in-process across authorized providers only.
"""

import logging
from typing import Optional

from config import config
from backoff import BackoffManager
from session import SessionManager
from cache import EndpointCache

logger = logging.getLogger(__name__)

# Prices fetched from OpenRouter API are per-token. The user-facing config
# (max_price) is expressed in dollars per MILLION tokens. This is the single
# conversion point for the routing decision.
PER_MILLION = 1_000_000


def _norm_base(provider_name: str) -> str:
    """Normalize a provider reference to its BASE name.

    Config `providers` may be written either as base names ("deepinfra") or as
    full endpoint tags ("z-ai/fp8") depending on whether the entry was hand-
    written or saved from the dashboard. Endpoint tags always arrive as
    "base[/quant]". Comparisons happen on the normalized BASE form so both
    spellings work interchangeably.
    """
    p = (provider_name or "").strip()
    return p.split("/")[0] if "/" in p else p


class Router:
    """Main router: selects the ordered list of authorized providers for a request."""

    def __init__(
        self,
        backoff: BackoffManager,
        sessions: SessionManager,
        endpoint_cache: EndpointCache,
    ):
        self.backoff = backoff
        self.sessions = sessions
        self.endpoint_cache = endpoint_cache

    def select_candidates(
        self,
        model_id: str,
        session_id: Optional[str] = None,
        tools: bool = False,
    ) -> list[tuple[str, str]]:
        """Return the ordered list of viable (provider_slug, quantization_tier).

        Strictly limited to providers listed in the model config (`providers`):
        a provider whose slug base is NOT in that list is **never** a candidate
        (allowlist gate). The list is sorted by tier priority, then provider
        priority, then price. A session's sticky provider (if still healthy and
        authorized) is placed first to preserve its cache.

        Returns [] when no authorized, usable provider exists.
        """
        endpoints = self._get_endpoints(model_id)
        if not endpoints:
            logger.warning(f"No endpoints cached for model {model_id}")
            return []

        model_config = config.get_model_config(model_id)
        if not model_config:
            logger.warning(f"No config for model {model_id}")
            return []

        quantizations = model_config.get("quantizations", ["fp8", "fp4"])
        provider_order = model_config.get("providers", [])

        candidates = self._compute_candidates(
            model_id, quantizations, provider_order,
            model_config.get("max_price", {}), endpoints,
        )

        # Build ordered result (allowlist already applied in _compute_candidates)
        ordered: list[tuple[str, str]] = []
        seen: set[str] = set()

        # Session stickiness first (preserve cache), if still healthy + authorized
        if session_id:
            sticky = self.sessions.get_provider(session_id)
            if sticky and not self.backoff.is_cooldown(sticky):
                if any(c["slug"] == sticky for c in candidates):
                    tier = self._detect_tier(sticky, quantizations)
                    ordered.append((sticky, tier))
                    seen.add(sticky)

        for c in candidates:
            if c["slug"] not in seen:
                ordered.append((c["slug"], c["tier"]))
                seen.add(c["slug"])

        return ordered

    def select_provider(
        self,
        model_id: str,
        session_id: Optional[str] = None,
        tools: bool = False,
    ) -> Optional[tuple[str, str]]:
        """Back-compat: return the single best (provider_slug, tier), or None.

        The session binding is no longer done here; routes.py binds the session
        to the provider that actually *served* the request (after a successful
        upstream response), so a failed candidate is never recorded as sticky.
        """
        cands = self.select_candidates(model_id, session_id, tools)
        if not cands:
            return None
        return cands[0]

    def bind_session(self, session_id: str, provider_slug: str) -> None:
        """Pin a session to the provider that actually served the request."""
        if session_id:
            self.sessions.set_provider(session_id, provider_slug)

    def _compute_candidates(
        self,
        model_id: str,
        quantizations: list[str],
        provider_order: list[str],
        max_price: dict,
        endpoints: list[dict],
    ) -> list[dict]:
        """Filter endpoints into an ordered candidate list.

        The allowlist gate lives here: a provider whose base slug is not in
        `provider_order` is EXCLUDED, not merely deprioritized.
        """
        max_input = max_price.get("input", float("inf"))
        max_completion = max_price.get("completion", float("inf"))
        max_cache = max_price.get("cache", float("inf"))

        candidates = []
        for ep in endpoints:
            tag = ep.get("tag")
            if not tag:
                continue

            pricing = ep.get("pricing") or {}
            prompt_price = float(pricing.get("prompt", 0) or 0)
            completion_price = float(pricing.get("completion", 0) or 0)
            cache_price = float(pricing.get("input_cache_read", 0) or 0)

            # Filter by max_price. max_price is in $/M tokens, provider prices
            # are per-token: convert the provider price to $/M for comparison.
            if prompt_price * PER_MILLION > max_input:
                continue
            if completion_price * PER_MILLION > max_completion:
                continue
            if cache_price * PER_MILLION > max_cache:
                continue

            # Filter by backoff health
            if self.backoff.is_cooldown(tag):
                continue

            # Determine tier
            tier = self._detect_tier(tag, quantizations)

            # Only consider if tier matches our priority list
            if tier not in quantizations:
                continue

            # ALLOWLIST GATE: never use a provider that isn't in the configured
            # list, no matter how cheap/healthy it is. Both sides are normalized
            # to the provider BASE so config entries written as full tags
            # ("z-ai/fp8", dashboard-saved) match endpoint tags ("z-ai/fp8")
            # exactly like hand-written base names ("z-ai").
            base = _norm_base(tag)
            cfg_bases = [_norm_base(p) for p in provider_order]
            if base not in cfg_bases:
                logger.debug(f"Provider {tag} not in allowlist {provider_order}; excluded")
                continue
            provider_idx = cfg_bases.index(base)

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
            return []

        # Sort: by tier priority, then provider order, then price
        tier_priority = {tier: i for i, tier in enumerate(quantizations)}
        candidates.sort(key=lambda c: (
            tier_priority.get(c["tier"], 999),
            c["provider_idx"],
            c["prompt_price"],
        ))
        return candidates

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
        mc = config.get_model_config(model_id) or {}
        quantizations = mc.get("quantizations", ["fp8", "fp4"])
        for ep in endpoints:
            if ep.get("tag") == provider:
                return self._detect_tier(provider, quantizations)
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
