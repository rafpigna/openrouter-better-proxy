"""Price migration logic — decides when to switch sessions to cheaper providers."""

import logging
from typing import Optional

from config import config

logger = logging.getLogger(__name__)


class PriceMigration:
    """Evaluate whether to migrate a session to a cheaper provider.

    Implements the N* formula from DESIGN.md §3.1:
    N* = (p_in(B) - p_cache(B)) * R
         -----------------------------------
         R * (p_cache(A) - p_cache(B)) + O * (p_out(A) - p_out(B))

    Where:
    - A = current sticky provider
    - B = alternative provider
    - R = estimated cache tokens in session
    - O = estimated output per turn
    """

    def __init__(
        self,
        hysteresis_mult: float = 3.0,
        est_turns_per_session: int = 50,
        r_cache_estimate: int = 300000,
        out_per_turn_estimate: int = 40000,
    ):
        self.hysteresis_mult = hysteresis_mult
        self.est_turns_per_session = est_turns_per_session
        self.r_cache_estimate = r_cache_estimate
        self.out_per_turn_estimate = out_per_turn_estimate

    def calculate_n_star(
        self,
        provider_a: dict,
        provider_b: dict,
        r_cache: Optional[int] = None,
        out_per_turn: Optional[int] = None,
    ) -> float:
        """Calculate N* (break-even turns).

        Returns:
            N* value. If <= 0, switching never makes sense.
            If < 1, switching makes sense immediately.
        """
        r = r_cache or self.r_cache_estimate
        o = out_per_turn or self.out_per_turn_estimate

        # Get prices (convert from string if needed)
        p_in_a = float(provider_a.get("pricing", {}).get("prompt", 0) or 0)
        p_cache_a = float(provider_a.get("pricing", {}).get("input_cache_read", 0) or 0)
        p_out_a = float(provider_a.get("pricing", {}).get("completion", 0) or 0)

        p_in_b = float(provider_b.get("pricing", {}).get("prompt", 0) or 0)
        p_cache_b = float(provider_b.get("pricing", {}).get("input_cache_read", 0) or 0)
        p_out_b = float(provider_b.get("pricing", {}).get("completion", 0) or 0)

        # Numerator: cost of switching (write cache on first turn)
        numerator = (p_in_b - p_cache_b) * r

        # Denominator: savings per turn
        denom = r * (p_cache_a - p_cache_b) + o * (p_out_a - p_out_b)

        if denom <= 0:
            logger.debug("Switching never makes sense (denominator <= 0)")
            return float("inf")

        n_star = numerator / denom
        logger.debug(f"N* = {n_star:.2f} (num={numerator:.6f}, denom={denom:.6f})")
        return n_star

    def should_migrate(
        self,
        current_provider: dict,
        alternative_provider: dict,
        est_turns_remaining: Optional[int] = None,
    ) -> tuple[bool, float]:
        """Decide whether to migrate to alternative provider.

        Returns:
            (should_migrate: bool, n_star: float)
        """
        n_star = self.calculate_n_star(current_provider, alternative_provider)

        if n_star == float("inf"):
            return False, n_star

        turns = est_turns_remaining or self.est_turns_per_session

        # Check hysteresis
        cost_switch = (
            float(alternative_provider.get("pricing", {}).get("prompt", 0) or 0)
            - float(alternative_provider.get("pricing", {}).get("input_cache_read", 0) or 0)
        ) * self.r_cache_estimate

        savings_per_turn = (
            self.r_cache_estimate * (
                float(current_provider.get("pricing", {}).get("input_cache_read", 0) or 0)
                - float(alternative_provider.get("pricing", {}).get("input_cache_read", 0) or 0)
            )
            + self.out_per_turn_estimate * (
                float(current_provider.get("pricing", {}).get("completion", 0) or 0)
                - float(alternative_provider.get("pricing", {}).get("completion", 0) or 0)
            )
        )

        estimated_gain = turns * savings_per_turn
        threshold = self.hysteresis_mult * cost_switch

        should_migrate = estimated_gain >= threshold
        logger.info(
            f"Migration check: N*={n_star:.2f}, turns={turns}, "
            f"gain={estimated_gain:.6f}, threshold={threshold:.6f}, "
            f"should_migrate={should_migrate}"
        )

        return should_migrate, n_star

    def get_config_summary(self) -> dict:
        """Get current migration config."""
        return {
            "hysteresis_mult": self.hysteresis_mult,
            "est_turns_per_session": self.est_turns_per_session,
            "r_cache_estimate": self.r_cache_estimate,
            "out_per_turn_estimate": self.out_per_turn_estimate,
        }
