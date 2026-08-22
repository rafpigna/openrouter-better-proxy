# OpenRouter Better Proxy

A local proxy that sits between Hermes Agent and OpenRouter, taking control of provider selection for models like `deepseek/deepseek-v4-flash-0731`. It applies custom routing logic that OpenRouter doesn't offer natively — or offers in a limited or hard-to-maintain way.

---

## Overview

### What it does

The proxy intercepts requests from Hermes (or any OpenAI-compatible client) and routes them to the best available OpenRouter provider based on:

- **Quantization tier** — prefers higher quality (fp8 over fp4)
- **Current price** — respects a `max_price` cap per model
- **Provider health** — backs off providers that return errors
- **Session stickiness** — keeps the same provider for the duration of a session to preserve prompt cache

### Why it exists

OpenRouter's default load-balancing has several issues when used for cost-sensitive, long-running conversations:

1. **Low-quality quantization selected first.** The default is price-based. For DeepSeek V4 Flash, the cheapest endpoint is often fp4 (e.g., `open-inference/fp4` at $0.065/M) while DeepSeek's own fp8 endpoint is $0.22/M. The router picks fp4 — cheaper, but noticeably worse quality.

2. **Cache lost on provider switch.** Prompt cache is per-provider. If a request switches provider mid-session, the new provider has a cold cache — you pay full price for the prefill again.

3. **Unstable provider slugs.** A provider slug can change over time (e.g., `deepseek/fp8` → `deepseek`), breaking static pins.

4. **Dynamic pricing.** DeepSeek introduced peak/off-peak pricing (prices double during peak hours). Some providers offer temporary discounts that expire without warning. OpenRouter won't switch providers if cache already exists, even when the price difference makes switching worthwhile.

The proxy solves all four by making routing decisions locally, with full visibility of current prices and session state.

---

## Requirements

### For the proxy server

| Resource | Minimum | Recommended |
|---|---|---|
| OS | Linux (Debian 12+, Ubuntu 22.04+) or Windows | Linux (Debian 12+) |
| RAM | 256 MB | 512 MB |
| Disk | 500 MB | 1 GB |
| CPU | 0.5 core | 1 core |
| Python | 3.11+ | 3.11+ |
| OpenRouter API key | Required | — |

The proxy is designed to run on a always-on machine — a VM, LXC container, or a machine that stays on alongside Hermes. It uses minimal resources (well under 100 MB RAM at steady state).

### For Hermes (client)

- Hermes Agent v0.20+ installed on any machine that can reach the proxy over the network.

---

## Installation

### Step 1: Deploy the proxy

Choose one of the following deployment options:

#### Option A: Clone + Python install (Linux)

```bash
# Clone the repository
git clone https://github.com/rafpigna/openrouter-better-proxy.git
cd openrouter-better-proxy

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create configuration files
cp .env.example .env
cp routing_config.yaml .

# Edit .env and add your OpenRouter API key
nano .env
# Add: OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx

# Edit routing_config.yaml if needed
nano routing_config.yaml
```

#### Option B: Docker (coming soon)

A Docker image is planned for future releases.

### Step 2: Configure the proxy

Edit `routing_config.yaml` to match your needs. See the [Configuration](#configuration) section below for a full explanation of every parameter.

At minimum, set your OpenRouter API key in `.env`:

```bash
OPENROUTER_API_KEY=sk-or-v1-REPLACE_WITH_YOUR_KEY
```

### Step 3: Start the proxy

#### Manual start (testing)

```bash
cd /opt/openrouter-better-proxy
source venv/bin/activate
python main.py
```

The server will start on `http://0.0.0.0:8787` by default.

#### Systemd service (production)

```bash
# Copy the service file
sudo cp openrouter-better-proxy.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable openrouter-better-proxy
sudo systemctl start openrouter-better-proxy

# Check status
sudo systemctl status openrouter-better-proxy
```

### Step 4: Install the Hermes plugin

The proxy is a drop-in replacement for OpenRouter. To use it with Hermes, install the custom plugin.

#### On Linux/macOS

```bash
# From the repository root
chmod +x scripts/hermes-install-plugin.sh
./scripts/hermes-install-plugin.sh http://192.168.1.251:8787/v1
```

#### On Windows

```powershell
# From the repository root
.\scripts\hermes-install-plugin.ps1 -ProxyUrl "http://192.168.1.251:8787/v1"
```

#### Manual installation

The plugin can be installed globally (default profile) or per-profile.

**Global installation (recommended)** — installs the plugin for all Hermes profiles:

```bash
# Linux/macOS
mkdir -p ~/.hermes/plugins/model-providers/openrouter-better-proxy
cp -r hermes-plugin/openrouter-better-proxy/* ~/.hermes/plugins/model-providers/openrouter-better-proxy/

# Windows (PowerShell, run as your user — no sudo needed)
New-Item -ItemType Directory -Path "$env:LOCALAPPDATA\hermes\plugins\model-providers\openrouter-better-proxy" -Force
Copy-Item -Path "hermes-plugin\openrouter-better-proxy\*" -Destination "$env:LOCALAPPDATA\hermes\plugins\model-providers\openrouter-better-proxy" -Force
```

**Per-profile installation** — installs the plugin only for a specific Hermes profile. Replace `<profile>` with your profile name (e.g., `test-proxy`):

```bash
# Linux/macOS
mkdir -p ~/.hermes/profiles/<profile>/plugins/model-providers/openrouter-better-proxy
cp -r hermes-plugin/openrouter-better-proxy/* ~/.hermes/profiles/<profile>/plugins/model-providers/openrouter-better-proxy/

# Windows (PowerShell)
New-Item -ItemType Directory -Path "$env:LOCALAPPDATA\hermes\profiles\<profile>\plugins\model-providers\openrouter-better-proxy" -Force
Copy-Item -Path "hermes-plugin\openrouter-better-proxy\*" -Destination "$env:LOCALAPPDATA\hermes\profiles\<profile>\plugins\model-providers\openrouter-better-proxy" -Force
```

> **Note:** Installing the plugin in the default (global) profile does not disable the built-in OpenRouter provider. You must still explicitly set `provider: openrouter-better-proxy` in your Hermes `config.yaml` to use the proxy. If you want to use the proxy only for a specific profile, install it per-profile instead.

After installing the plugin, configure the proxy URL:

1. Edit `config.yaml` inside the plugin directory to point to your proxy instance:
   ```yaml
   proxy_url: "http://192.168.1.251:8787/v1"
   ```

2. Update your Hermes `config.yaml` to use the custom provider:
   ```yaml
   model:
     default: deepseek/deepseek-v4-flash-0731
     provider: openrouter-better-proxy
   ```

---

## Configuration

The proxy is configured via `routing_config.yaml`. All paths are relative to the proxy working directory.

### Server settings

```yaml
server:
  host: "0.0.0.0"   # Bind address (use "127.0.0.1" for local-only)
  port: 8787         # Listening port
```

### Model configuration

Each model gets its own section. The proxy supports multiple models in a single config file.

```yaml
models:
  "deepseek/deepseek-v4-flash-0731":
    quantizations: ["fp8", "fp4"]        # Priority order (tiered selection)
    providers: ["deepseek", "deepinfra", "streamlake", "gmicloud"]  # Priority order
    max_price:
      input: 0.10                        # $/M token — max prompt price
      completion: 0.25                   # $/M token — max completion price
      cache: 0.05                        # $/M token — max cache read price
```

**`quantizations`** — List of quantization tiers in priority order. The router selects the highest-priority tier that has at least one healthy, affordable provider. If no fp8 provider is available, it falls back to fp4.

**`providers`** — List of provider slugs (the part before `/` in tags like `deepinfra/fp8`) in priority order. A provider listed here but without an endpoint in the selected tier is skipped.

**`max_price`** — Price caps in $/M token. An endpoint is excluded if ANY of its prices exceed the cap. Use `0` or omit to disable a cap.

### Global settings

#### Price migration

Controls whether the proxy automatically switches active sessions to a cheaper provider when prices change significantly.

```yaml
migration:
  enabled: true                        # Set false to disable migration entirely
  hysteresis_mult: 3.0                 # Switch only if estimated savings >= 3x switch cost
  est_turns_per_session: 50            # Estimated turns remaining per session
  r_cache_estimate: 300000             # Estimated cached tokens in a session
  out_per_turn_estimate: 40000         # Estimated output tokens per turn
```

**How migration works:** When a price change is detected (see [Refresh](#refresh-settings)), the proxy calculates `N*` — the break-even number of turns. If the estimated remaining turns in the session exceed `N*` (adjusted by `hysteresis_mult`), the session is migrated to the cheaper provider on the next request.

The formula:

```
N* = (p_in(B) - p_cache(B)) * R
     -----------------------------------
     R * (p_cache(A) - p_cache(B)) + O * (p_out(A) - p_out(B))

A = current sticky provider
B = best alternative provider
R = cached tokens (r_cache_estimate)
O = output per turn (out_per_turn_estimate)
```

- `N* <= 0`: switching never makes sense
- `N* < 1`: switch immediately
- `N*` small (1–5): depends on remaining turns + hysteresis

#### Refresh settings

Controls how often the proxy fetches updated pricing from OpenRouter.

```yaml
refresh:
  interval_minutes: 30                 # Periodic refresh interval
  price_change_threshold: 0.01         # Relative threshold (1%) to detect price changes
  times:                               # Explicit refresh times (can be dense)
    - "10:00"
    - "10:05"
    - "10:10"
    - "16:00"
    - {from: "18:00", to: "22:00", step: "05m"}  # Window with step
```

The scheduler combines periodic and explicit triggers. During dense windows (e.g., peak-hour transitions), it refreshes more frequently to catch price changes quickly.

#### Health / backoff settings

Controls how the proxy handles provider errors.

```yaml
health:
  initial_cooldown_seconds: 300        # 5 min after first error
  consecutive_threshold: 3             # Escalate after this many consecutive errors
  escalation_seconds: [3600, 43200]    # 1h, then 12h
  max_cooldown_seconds: 43200          # Never more than 12h
```

| Errors | Cooldown |
|---|---|
| 1 | 5 minutes |
| 3 consecutive | 1 hour |
| Error after 1h cooldown | 12 hours |
| Success | Reset |

If all providers in a tier are in cooldown, the router falls back to the next lower quantization tier.

---

## Usage

### With Hermes Agent

After installing the plugin (see [Installation](#step-4-install-the-hermes-plugin)), configure your Hermes profile:

```yaml
# In ~/.hermes/profiles/<profile>/config.yaml
model:
  default: deepseek/deepseek-v4-flash-0731
  provider: openrouter-better-proxy
```

The proxy handles everything automatically. Use `session_id` in your requests to preserve cache across turns:

```python
# Example: Hermes chat with session stickiness
response = client.chat.completions.create(
    model="deepseek/deepseek-v4-flash-0731",
    messages=[{"role": "user", "content": "Hello"}],
    stream=True,
    session_id="my-conversation-123",  # Same ID = same provider = cached prefill
)
```

### Direct API usage

The proxy is OpenAI-compatible. You can use it with any OpenAI SDK or HTTP client:

```bash
# Health check
curl http://localhost:8787/health

# List configured models
curl http://localhost:8787/v1/models

# Chat completion (streaming)
curl http://localhost:8787/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek/deepseek-v4-flash-0731",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "session_id": "test-session-1"
  }'

# Manual price refresh
curl -X POST http://localhost:8787/refresh

# Status (debug)
curl http://localhost:8787/status
```

---

## Debug

### HTTP endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/v1/models` | GET | List configured models |
| `/v1/chat/completions` | POST | Chat completion (forwarded to best provider) |
| `/refresh` | POST | Manually trigger price cache refresh |
| `/status` | GET | Router status (sessions, backoff, migration log) |

### Logs

Logs are written to both stdout and `logs/app.log` (relative to the proxy working directory).

```bash
# Follow logs in real-time
tail -f /opt/openrouter-better-proxy/logs/app.log

# Check systemd journal
sudo journalctl -u openrouter-better-proxy -f
```

Log levels: `INFO` (default), `DEBUG` (set `LOG_LEVEL=debug` in environment).

### Status endpoint

The `/status` endpoint returns detailed routing state:

```json
{
  "sessions": {
    "my-session-1": "deepinfra/fp8"
  },
  "backoff": {
    "deepinfra/fp8": {
      "provider": "deepinfra/fp8",
      "error_count": 0,
      "in_cooldown": false,
      "cooldown_remaining": 0
    }
  },
  "cached_models": ["deepseek/deepseek-v4-flash-0731"],
  "migration": {
    "config": {
      "enabled": true,
      "hysteresis_mult": 3.0,
      "est_turns_per_session": 50,
      "r_cache_estimate": 300000,
      "out_per_turn_estimate": 40000
    },
    "last_events": [
      {"timestamp": "2026-08-21T10:05:00Z", "action": "migrate", "session": "my-session-1", "from": "deepseek", "to": "deepinfra/fp8", "n_star": 0.8}
    ]
  }
}
```

### Common troubleshooting

#### "No valid provider found for model X"

- Check that the model ID in your request matches a model in `routing_config.yaml`
- Check that `OPENROUTER_API_KEY` is set in `.env`
- Run `curl -X POST http://localhost:8787/refresh` to force a price fetch
- Check `/status` to see if any endpoints are cached
- Verify your OpenRouter API key has access to the model

#### Provider always in cooldown

- Check logs for repeated errors from a provider
- The provider may be genuinely unhealthy — the proxy will auto-retry after the cooldown expires
- Adjust `health.initial_cooldown_seconds` if cooldowns are too aggressive

#### Cache not being preserved

- Ensure you're passing the same `session_id` across requests
- Check `/status` to verify the session is mapped to a provider
- If the provider enters cooldown, the session stickiness is released and a new provider is selected

#### Migration not triggering

- Check `/status` for migration events in `last_events`
- Verify `migration.enabled` is `true`
- Increase `hysteresis_mult` if migrations are too aggressive, or decrease if they're too conservative
- Check that price changes are being detected (look for `price_change` events in logs)

---

## Price Migration

### What it is

Price migration automatically switches an active session from its current provider to a cheaper alternative when the price difference becomes significant enough to justify the switch cost.

### When it triggers

Migration is evaluated only when a price change is detected — not on every request. Price changes are detected during scheduled refreshes by comparing the newly fetched prices against the previous snapshot.

Events that trigger migration evaluation:
- Provider enters or exits peak pricing
- A temporary discount expires
- Any price change exceeding `refresh.price_change_threshold` (default 1%)

### How it works

1. A price change is detected for the provider currently serving a session
2. The proxy calculates `N*` — the break-even number of turns
3. If `estimated_turns_remaining * savings_per_turn >= hysteresis_mult * switch_cost`, the session is migrated
4. On the next request, the session is routed to the new provider
5. The new provider writes the prompt cache on the first turn (cost: `N*` input tokens at the new provider's price)

### Configuration tips

| Setting | Effect | Typical value |
|---|---|---|
| `migration.enabled` | Disable entirely to disable | `true` |
| `hysteresis_mult` | Higher = fewer migrations, more conservative | `2.0` – `5.0` |
| `est_turns_per_session` | Higher = more likely to migrate | `30` – `100` |
| `r_cache_estimate` | Higher = migration more likely (larger cache to preserve) | `100000` – `500000` |
| `out_per_turn_estimate` | Higher = more savings per turn, more likely to migrate | `20000` – `80000` |

### Example scenario

DeepSeek goes on-peak (input $0.22 → $0.44, completion $0.66 → $1.32). The proxy detects this during a refresh. For a session currently on DeepSeek:

- `N*` is calculated comparing DeepSeek on-peak vs. DeepInfra fp8
- With typical values (R=300k, O=40k), `N*` ≈ 0.5
- Since `N* < 1`, the session migrates immediately
- The first turn on DeepInfra writes the cache (~$0.02), but all subsequent turns are much cheaper

---

## FAQ

### Does the proxy store my API key?

No. The OpenRouter API key is read from `.env` at startup and kept in memory. It is never written to logs, cache files, or transmitted to any service other than OpenRouter.

### What happens if the proxy crashes?

All state is in-memory. On restart, the proxy:
- Reloads `routing_config.yaml` and `.env`
- Restores endpoint cache from disk (if available)
- Starts with empty session and backoff state
- Sessions will be re-routed on the next request (no cache loss on the client side — the client's conversation context is preserved)

### Can I use the proxy without Hermes?

Yes. The proxy is a standard OpenAI-compatible server. Any client that speaks the OpenAI API format can use it.

### Does the proxy support non-streaming requests?

Yes. Both streaming (`stream: true`) and non-streaming requests are supported.

### What happens if all providers are down?

The router returns a 400 error with a message indicating no valid provider was found. The client should retry after the backoff period.

### Can I add more models?

Yes. Add a new section under `models:` in `routing_config.yaml`. Each model is configured independently with its own quantization tiers, provider order, and price caps.

### How do I update the proxy?

If you installed the proxy by cloning the repository, update both the source code and dependencies:

```bash
cd /path/to/openrouter-better-proxy   # your clone directory
git pull origin main                    # or your current branch
source venv/bin/activate
pip install -r requirements.txt --upgrade
sudo systemctl restart openrouter-better-proxy
```

If you installed the Hermes plugin manually, re-copy the plugin files after pulling the latest repository:

```bash
# Linux/macOS
cp -r hermes-plugin/openrouter-better-proxy/* ~/.hermes/plugins/model-providers/openrouter-better-proxy/

# Windows (PowerShell)
Copy-Item -Path "hermes-plugin\openrouter-better-proxy\*" -Destination "$env:LOCALAPPDATA\hermes\plugins\model-providers\openrouter-better-proxy" -Force
```

### Where are cache files stored?

Endpoint price caches are stored in `data/or_endpoints_<model>.json` (relative to the proxy working directory). Price snapshots for diff detection are stored in `data/price_snapshot_<model>.json`.

---

## License

This project is licensed under AGPL-3.0. See [LICENSE](LICENSE) for details.

For commercial licensing, see [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL).

---

## Acknowledgments

Built for [Hermes Agent](https://github.com/NousResearch/hermes-agent) by [Nous Research](https://nousresearch.com).
