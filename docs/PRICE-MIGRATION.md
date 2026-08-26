# Price Migration Guide

This document explains the price migration feature of the OpenRouter Better Proxy.

## Overview

Price migration automatically switches active sessions to cheaper providers when
OpenRouter changes pricing. This minimizes costs while maintaining cache
efficiency.

## How It Works

### The N* Formula

The decision to migrate is based on the N* formula:

```
N* = (p_in(B) - p_cache(B)) * R
     -----------------------------------
     R * (p_cache(A) - p_cache(B)) + O * (p_out(A) - p_out(B))
```

Where:
- **A** = current sticky provider
- **B** = alternative provider
- **R** = estimated cached tokens in session (default: 300,000)
- **O** = estimated output per turn (default: 40,000)

### Interpretation

- **N* <= 0**: Switching never makes sense (alternative is more expensive)
- **N* < 1**: Switch immediately (savings outweigh cache loss)
- **N* >= 1**: Switch after N* turns (break-even point)

### Hysteresis

To avoid ping-ponging between providers, the migration uses hysteresis:

```
threshold = hysteresis_mult * cost_switch
estimated_gain = turns_remaining * savings_per_turn
should_migrate = estimated_gain >= threshold
```

Default `hysteresis_mult = 3.0` means savings must be 3x the cost of switching.

## Configuration

### routing_config.yaml

```yaml
migration:
  enabled: true                    # Enable/disable migration
  hysteresis_mult: 3.0            # Hysteresis multiplier (higher = less migration)
  est_turns_per_session: 50       # Estimated turns per session
  r_cache_estimate: 300000        # Estimated cached tokens (R)
  out_per_turn_estimate: 40000    # Estimated output per turn (O)
```

### Tuning Guidelines

| Parameter | Default | When to Adjust |
|-----------|---------|----------------|
| `hysteresis_mult` | 3.0 | Increase to reduce migrations, decrease to migrate more aggressively |
| `est_turns_per_session` | 50 | Adjust based on your typical session length |
| `r_cache_estimate` | 300,000 | Adjust if your sessions have significantly more/less cached tokens |
| `out_per_turn_estimate` | 40,000 | Adjust based on your typical output length |

## Migration Events

Migration events are logged and can be viewed via the status endpoint:

```bash
curl http://localhost:8787/status
```

Response includes:

```json
{
  "migration": {
    "config": {
      "enabled": true,
      "hysteresis_mult": 3.0,
      ...
    },
    "last_events": [
      {
        "timestamp": "2026-08-21T10:00:00",
        "session_id": "abc123",
        "from_provider": "deepinfra/fp8",
        "to_provider": "streamlake/fp8",
        "n_star": 0.85,
        "model": "deepseek/deepseek-v4-flash-0731"
      }
    ]
  }
}
```

## Example Scenario

### Before Price Change

| Provider | Input ($/M) | Cache Read ($/M) | Completion ($/M) |
|----------|------------|------------------|------------------|
| deepinfra/fp8 | 0.08 | 0.016 | 0.18 |
| streamlake/fp8 | 0.0786 | 0.01572 | 0.15719 |

Streamlake is cheaper but deepinfra is sticky (session started there).

### After Price Change

DeepInfra raises prices to on-peak rates:

| Provider | Input ($/M) | Cache Read ($/M) | Completion ($/M) |
|----------|------------|------------------|------------------|
| deepinfra/fp8 | 0.16 | 0.032 | 0.36 |
| streamlake/fp8 | 0.0786 | 0.01572 | 0.15719 |

### Migration Decision

1. Scheduler detects price change on refresh
2. Calculates N* for each active session on deepinfra
3. If N* < threshold and hysteresis passes, migrates session to streamlake
4. Logs migration event

## Disabling Migration

To disable migration entirely, set `migration.enabled` to `false` in
`routing_config.yaml`:

```yaml
migration:
  enabled: false
```

(There is no separate environment variable — migration is controlled only via
the config file.)

## Manual Refresh

After a known price change, trigger an immediate refresh:

```bash
curl -X POST http://localhost:8787/refresh
```

This forces price detection and migration evaluation without waiting for the next scheduled refresh.
