"""OpenRouter Better Proxy provider profile.

Clone of the built-in openrouter plugin, but with base_url pointing to the
local router proxy instead of OpenRouter directly.

This preserves session_id sticky routing and all OpenRouter-specific behavior
(provider preferences, reasoning config, etc.) while routing through our proxy.
"""

import logging
import os
from pathlib import Path
from typing import Any

from agent.portal_tags import get_conversation_context
from agent.transports.codex import _cache_scope_from_session_id
from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


def _load_plugin_config() -> dict:
    """Load plugin config from config.yaml in plugin directory."""
    config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _get_proxy_url() -> str:
    """Get proxy URL from config or environment."""
    config = _load_plugin_config()
    return os.environ.get(
        "ORBP_PROXY_URL",
        config.get("proxy_url", "http://localhost:8787/v1")
    )

# Anthropic model families that still accept an explicit "disable thinking"
# request (the manual ``thinking: {type: "disabled"}`` form OpenRouter emits
# for ``reasoning: {enabled: false}``). Everything Claude 4.6 and newer —
# including future date-stamped / named models (fable, mythos-class, …) —
# mandates reasoning and returns HTTP 400 on any disable form. We therefore
# default *unknown* Anthropic models to "cannot disable" (the modern contract)
# and keep only this explicit legacy allowlist of models that can. Mirrors the
# default-to-newest philosophy in agent/anthropic_adapter._get_anthropic_max_output.
_ANTHROPIC_REASONING_OPTIONAL_SUBSTRINGS = (
    "claude-3",          # 3, 3.5, 3.7
    "claude-opus-4-0", "claude-opus-4.0", "claude-opus-4-1", "claude-opus-4.1",
    "claude-sonnet-4-0", "claude-sonnet-4.0",
    "claude-opus-4-2025", "claude-opus-4-2025",  # date-stamped 4.0 IDs
    "claude-opus-4-5", "claude-opus-4.5",
    "claude-sonnet-4-5", "claude-sonnet-4.5",
    "claude-haiku-4-5", "claude-haiku-4.5",
)


def _anthropic_reasoning_is_mandatory(model: str | None) -> bool:
    """Return True for Anthropic models that reject any disable-thinking form.

    Claude 4.6+ (adaptive thinking) and newer named models have no "off"
    switch — sending ``reasoning: {enabled: false}`` makes OpenRouter emit
    ``thinking: {type: "disabled"}``, which these models 400 on. Unknown /
    new Anthropic model names default to mandatory so the next un-numbered
    release doesn't reintroduce the 400.
    """
    m = (model or "").lower()
    if not m.startswith(("anthropic/", "claude")) and "claude" not in m:
        return False
    return not any(sub in m for sub in _ANTHROPIC_REASONING_OPTIONAL_SUBSTRINGS)


class OpenRouterBetterProxyProfile(ProviderProfile):
    """OpenRouter Better Proxy — routes through local proxy."""

    @staticmethod
    def _clamp_reasoning_to_catalog(cfg: dict[str, Any], model: str | None) -> dict[str, Any]:
        """Clamp ``cfg["effort"]`` to the model's catalog-advertised levels.

        OpenRouter's /v1/models entries publish ``reasoning.supported_efforts``
        per model (ported from PrimeIntellect-ai/prime-agent#1258). Sending an
        unsupported effort (e.g. ``ultra`` to a route that stops at ``high``)
        yields provider 4xx errors; clamp to the nearest LOWER supported level
        instead. No-op when the catalog is unreachable, the model is unlisted,
        or no supported_efforts list is published (None = all levels accepted).
        """
        effort = cfg.get("effort")
        if not effort or cfg.get("enabled") is False:
            return cfg
        try:
            from hermes_cli.models import (
                clamp_reasoning_effort_to_supported,
                openrouter_model_reasoning_capabilities,
            )
            caps = openrouter_model_reasoning_capabilities(model)
            if not caps or not caps.get("supports_reasoning"):
                return cfg
            clamped = clamp_reasoning_effort_to_supported(
                effort, caps.get("supported_efforts")
            )
        except Exception:
            return cfg
        if clamped and clamped != effort:
            logger.debug(
                "openrouter-better-proxy: clamped reasoning effort %r → %r for %s "
                "(catalog supported_efforts=%s)",
                effort, clamped, model, caps.get("supported_efforts"),
            )
            cfg = dict(cfg)
            cfg["effort"] = clamped
        return cfg

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch from proxy's /v1/models — no auth required."""
        try:
            result = super().fetch_models(api_key=None, base_url=base_url, timeout=timeout)
            return result
        except Exception as exc:
            logger.debug("fetch_models(openrouter-better-proxy): %s", exc)
            return None

    def build_extra_body(
        self, *, session_id: str | None = None, **context: Any
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        # Top-level session_id → OpenRouter's sticky routing key. Per their
        # prompt-caching docs it is used directly as the routing key instead of
        # hashing the opening messages, and it activates stickiness on the
        # first successful request rather than only after a cache hit.
        sticky_key = _cache_scope_from_session_id(get_conversation_context() or session_id)
        if sticky_key:
            body["session_id"] = sticky_key
        prefs = context.get("provider_preferences")
        if prefs:
            body["provider"] = prefs

        # Pareto Code router — model-gated.
        model = (context.get("model") or "")
        if model == "openrouter/pareto-code":
            score = context.get("openrouter_min_coding_score")
            if score is not None and score != "":
                try:
                    score_f = float(score)
                except (TypeError, ValueError):
                    score_f = None
                if score_f is not None and 0.0 <= score_f <= 1.0:
                    body["plugins"] = [
                        {"id": "pareto-router", "min_coding_score": score_f}
                    ]
        return body

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        model: str | None = None,
        session_id: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """OpenRouter passes the full reasoning_config dict as extra_body.reasoning."""
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        extra_headers: dict[str, Any] = {}
        if supports_reasoning:
            if _anthropic_reasoning_is_mandatory(model):
                cfg = reasoning_config or {}
                effort = cfg.get("effort")
                if cfg.get("enabled", True) is not False and effort and effort != "none":
                    top_level["verbosity"] = effort
            elif reasoning_config is not None:
                extra_body["reasoning"] = self._clamp_reasoning_to_catalog(
                    dict(reasoning_config), model
                )
            else:
                extra_body["reasoning"] = {"enabled": True, "effort": "medium"}

        # xAI's prompt cache is pinned per backend server via this header
        grok_conv_id = _cache_scope_from_session_id(get_conversation_context() or session_id)
        if grok_conv_id and model and model.startswith(("x-ai/grok-", "xai/grok-")):
            extra_headers["x-grok-conv-id"] = grok_conv_id
        if extra_headers:
            top_level["extra_headers"] = extra_headers

        return extra_body, top_level


openrouter_better_proxy = OpenRouterBetterProxyProfile(
    name="openrouter-better-proxy",
    aliases=("orbp",),
    env_vars=("OPENROUTER_API_KEY",),
    display_name="OpenRouter Better Proxy",
    description="OpenRouter Better Proxy — routes through local proxy",
    signup_url="",
    base_url=_get_proxy_url(),
    models_url=_get_proxy_url() + "/models",
    fallback_models=(
        "deepseek/deepseek-v4-flash-0731",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "deepseek/deepseek-chat",
        "google/gemini-3.7-flash",
        "qwen/qwen3-plus",
    ),
)

register_provider(openrouter_better_proxy)
