"""OpenRouter Router Proxy — Configuration loader."""

import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Load and validate routing_config.yaml + .env secrets."""

    def __init__(self, config_path: str = "routing_config.yaml", env_path: str = ".env"):
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self.raw: dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load config from YAML and .env."""
        # Load YAML
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.raw = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        # Load .env (simple key=value, no quotes required)
        if self.env_path.exists():
            with open(self.env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        os.environ[key.strip()] = value.strip()

    # --- Server ---
    @property
    def host(self) -> str:
        return self.raw.get("server", {}).get("host", "0.0.0.0")

    @property
    def port(self) -> int:
        return self.raw.get("server", {}).get("port", 8787)

    @property
    def dashboard_enabled(self) -> bool:
        """Dashboard abilitata di default, disabilitabile via config o env var."""
        enabled = self.raw.get("server", {}).get("dashboard", True)
        if os.environ.get("DASHBOARD_DISABLED") == "1":
            return False
        return bool(enabled)

    @property
    def sse_log_enabled(self) -> bool:
        """SSE live log: SOLO se esplicitamente abilitato (config server.sse_log: true o env SSE_LOG_ENABLED=1).

        Default OFF. Se flag/env assenti -> False, nessuna risorsa consumata.
        """
        enabled = self.raw.get("server", {}).get("sse_log", False)
        env_val = os.environ.get("SSE_LOG_ENABLED", "")
        if env_val in ("1", "true", "True", "on", "ON"):
            return True
        if env_val in ("0", "false", "False", "off", "OFF"):
            return False
        return bool(enabled)

    # --- Models ---
    @property
    def models(self) -> dict[str, Any]:
        return self.raw.get("models", {})

    def get_model_config(self, model_id: str) -> dict[str, Any]:
        """Get config for a specific model, or empty dict if not found."""
        return self.models.get(model_id, {})

    # --- Migration ---
    @property
    def migration_enabled(self) -> bool:
        return self.raw.get("migration", {}).get("enabled", True)

    @property
    def hysteresis_mult(self) -> float:
        return self.raw.get("migration", {}).get("hysteresis_mult", 3.0)

    @property
    def est_turns_per_session(self) -> int:
        return self.raw.get("migration", {}).get("est_turns_per_session", 50)

    @property
    def r_cache_estimate(self) -> int:
        return self.raw.get("migration", {}).get("r_cache_estimate", 300000)

    @property
    def out_per_turn_estimate(self) -> int:
        return self.raw.get("migration", {}).get("out_per_turn_estimate", 40000)

    # --- Refresh ---
    @property
    def refresh_interval_minutes(self) -> int:
        return self.raw.get("refresh", {}).get("interval_minutes", 30)

    @property
    def price_change_threshold(self) -> float:
        return self.raw.get("refresh", {}).get("price_change_threshold", 0.01)

    @property
    def refresh_times(self) -> list[Any]:
        return self.raw.get("refresh", {}).get("times", [])

    # --- Health ---
    @property
    def initial_cooldown_seconds(self) -> int:
        return self.raw.get("health", {}).get("initial_cooldown_seconds", 300)

    @property
    def consecutive_threshold(self) -> int:
        return self.raw.get("health", {}).get("consecutive_threshold", 3)

    @property
    def escalation_seconds(self) -> list[int]:
        return self.raw.get("health", {}).get("escalation_seconds", [3600, 43200])

    @property
    def max_cooldown_seconds(self) -> int:
        return self.raw.get("health", {}).get("max_cooldown_seconds", 43200)

    # --- Secrets ---
    @property
    def openrouter_api_key(self) -> str:
        return os.environ.get("OPENROUTER_API_KEY", "")


# Global config instance
config = Config()