"""Two-clock contract for refresh.default_timezone (2026-08-31).

Exactly two clocks exist: "local" (OS timezone of the machine) and "UTC".
IANA names are intentionally NOT accepted — a typo would silently degrade
bare entries with no UI surface to catch it. The dashboard validator must
reject anything else with a clear message.
"""

import yaml
import pytest
from pydantic import ValidationError

from web_routes import RefreshConfig
from scheduler import resolve_tz


@pytest.mark.parametrize("value,normalized", [
    ("local", "local"),
    ("LOCAL", "local"),
    (" Local ", "local"),
    ("UTC", "UTC"),
    ("utc", "UTC"),
    (None, "local"),      # default
    ("", "local"),        # empty -> local
])
def test_valid_default_timezone(value, normalized):
    kw = {} if value is None else {"default_timezone": value}
    cfg = RefreshConfig(**kw)
    assert cfg.default_timezone == normalized


@pytest.mark.parametrize("bad", [
    "Europe/Rome",
    "Europe/Moscow",
    "Rome",               # the typo case from the design discussion
    "America/New_York",
    "+02:00",             # offset makes no sense as a DEFAULT
    "CEST",
    "GMT+2",
])
def test_iana_and_other_defaults_rejected(bad):
    with pytest.raises(ValidationError) as exc:
        RefreshConfig(default_timezone=bad)
    msg = str(exc.value)
    assert "'local' or 'UTC'" in msg, msg


def test_resolve_tz_rejects_offsets_and_z():
    """Two-clock contract: offsets and legacy 'Z' are gone everywhere."""
    from datetime import timedelta
    for bad in ("+02:00", "-05:30", "+0200", "Z"):
        with pytest.raises(ValueError):
            resolve_tz(bad, "local")


def test_resolve_tz_rejects_iana():
    with pytest.raises(ValueError):
        resolve_tz("Europe/Rome", "local")


def test_example_config_is_two_clock_compliant():
    data = yaml.safe_load(open("routing_config.yaml", encoding="utf-8"))
    tz = data["refresh"]["default_timezone"]
    assert tz in ("local", "UTC"), tz
    cfg = RefreshConfig(**data["refresh"])
    assert cfg.default_timezone == cfg.default_timezone  # validator ran clean
